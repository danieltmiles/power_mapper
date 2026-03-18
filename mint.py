import json
import uuid
from json import JSONDecodeError
from pathlib import Path

import aio_pika
import argparse
import asyncio

from aio_pika.abc import AbstractIncomingMessage, AbstractRobustConnection
from transformers import AutoTokenizer, Qwen2Tokenizer

import serialization
import wire_formats
from cached_iterator import CachedMessageIterator
from logger import get_logger
from utils import load_config, get_answer, dial_rabbit_from_config, dial_redis_from_config
from wire_formats import Metaparams

logger = get_logger("mint")


def create_transcript_metadata_prompt(filename: str, tokenizer: Qwen2Tokenizer) -> str:
    consider_template = (
        "Consider the following file name of a transcript: `{filename}`\n"
        "From that filename, can you infer a meeting title, session type, date and video ID? "
        "Hint: The video ID is between the square brackets. "
        "Please round timestamps to the nearest day and format in ISO format."
        "Keep your thinking brief and concise."
    )
    conversation = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that extracts structured data from meeting transcript filenames. "
                "You respond with JSON only, never with any explanations. "
                "If you cannot infer a field, respond with null for that field."
            )
        },
        {
            "role": "user",
            "content": consider_template.format(
                filename="Portland City Council - Regular Session - 2024-01-15 [ABC123].txt",
            ),
        },
        {
            "role": "assistant",
            "content": (
                "```json\n{\n"
                '  "filename": "Portland City Council - Regular Session - 2024-01-15 [ABC123].txt",\n'
                '  "meeting_title": "Portland City Council",\n'
                '  "session_type": "Regular Session",\n'
                '  "date": "2024-01-15T00:00:00Z",\n'
                '  "video_id": "ABC123"\n'
                "}\n```"
            )
        },
        {
            "role": "user",
            "content": consider_template.format(filename=filename),
        }
    ]
    return tokenizer.apply_chat_template(conversation, tokenize=False)

async def get_transcript_metadata_from_llm(
    message: AbstractIncomingMessage,
    connection: AbstractRobustConnection,
    tokenizer: Qwen2Tokenizer,
) -> wire_formats.LLMPromptResponse | None:
    body = json.loads(message.body.decode())
    filename = body.get("filename")
    prompt = create_transcript_metadata_prompt(filename, tokenizer)
    reply_to = f"mint/reply/{str(uuid.uuid4())}"
    job_desc = wire_formats.LLMPromptJob(
        job_id=str(uuid.uuid4()),
        filename=Path(filename).name,
        reply_to=reply_to,
        prompt=prompt,
        meta_params=Metaparams(
            max_tokens=4096,
            temperature=0.1,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.2,
        )
    )
    async with await connection.channel() as channel:
        async def consume_response():
            response_queue = await channel.declare_queue(reply_to, durable=True)
            async with response_queue.iterator() as queue_iter:
                async for response_message in queue_iter:
                    async with response_message.process():
                        response: wire_formats.LLMPromptResponse = serialization.load(
                            response_message.body.decode(),
                            cls=wire_formats.LLMPromptResponse,
                        )
                        return response
            return None
        await channel.default_exchange.publish(
            aio_pika.Message(
                body=serialization.dumps(job_desc).encode("utf-8"),
            ),
            routing_key="llm/qwen32",
        )
        resp = await(consume_response())
        # remove the queue
        await channel.queue_delete(reply_to)
        return resp



async def process_message(
    message: AbstractIncomingMessage,
    tokenizer: Qwen2Tokenizer,
    result_queue: str,
    config: dict,
):
    """Process a single message"""
    try:
        logger.info("process debug 1")
        # Dial a fresh connection for publishing
        async with await dial_rabbit_from_config(config) as rabbitmq_connection:
            logger.info("process debug 2")
            resp: wire_formats.LLMPromptResponse = await get_transcript_metadata_from_llm(message, rabbitmq_connection, tokenizer)
            logger.info("process debug 3")
            answer_str = get_answer(resp.generated_text, start_delim="```json", end_delim="```")
            logger.info("process debug 4")
            try:
                logger.info("process debug 6")
                answer = wire_formats.TranscriptMetadata(**json.loads(answer_str))
                logger.info("process debug 7")
            except TypeError:
                logger.error(f"Failed to parse answer JSON: {answer_str}")
                return
            except JSONDecodeError as jde:
                logger.error(f"Invalid JSON format in answer: {answer_str}\nError: {jde}")
                return

            logger.info("process debug 8")
            async with await rabbitmq_connection.channel() as pub_channel:
                logger.info("process debug 9")
                await pub_channel.default_exchange.publish(
                    aio_pika.Message(
                        body=serialization.dumps(answer).encode("utf-8"),
                    ),
                    routing_key=result_queue,
                )
                logger.info("process debug 10")
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)


async def main(config):
    """
    Main function to start the RabbitMQ consumer with reconnection logic.
    Handles connection failures and automatically reconnects with exponential backoff.
    """
    logger.info("Initializing RabbitMQ consumer...")

    # Retry configuration
    max_retries = 10
    base_retry_delay = 2  # seconds
    max_retry_delay = 60  # seconds

    logger.info("Loading tokenizer...")
    tokenizer: Qwen2Tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")
    
    retry_count = 0
    
    while True:
        try:
            logger.info(f"Connecting to RabbitMQ at {config['host']}:{config['port']}...")
            connection = await dial_rabbit_from_config(config)
            logger.info("debug 1")
            redis_client = await dial_redis_from_config(config)
            logger.info("debug 2")

            async with connection:
                logger.info("debug 3")
                async with await connection.channel() as channel:
                    logger.info("debug 4")
                    # Declare the work queue
                    work_queue = config['work_queue']
                    result_queue = config["result_queue"]
                    logger.info(f"debug 5 {work_queue=} {result_queue=}")
                    await asyncio.gather(
                        channel.declare_queue(work_queue, durable=True),
                        channel.declare_queue(result_queue, durable=True),
                    )
                    logger.info("debug 6")

                logger.info(f"Successfully connected! Listening for jobs on queue: {work_queue}")
                logger.info("Waiting for messages. To exit press CTRL+C")
                
                # Start consuming messages
                async with CachedMessageIterator(
                        rabbitmq_connection=connection,
                        redis_client=redis_client,
                        queue_name=work_queue,
                        redis_key_prefix="backup:mint",
                        config=config,
                ) as queue_iter:
                    logger.info("debug 7")
                    async for message in queue_iter:
                        logger.info("debug 8")
                        async with queue_iter.processing(message):
                            logger.info("debug 9")
                            await process_message(message, tokenizer, result_queue, config)
                            logger.info("debug 10")

        except (Exception,) as conn_error:
            retry_count += 1
            if retry_count > max_retries:
                logger.error(f"Max retries ({max_retries}) exceeded. Giving up.")
                raise

            delay = min(base_retry_delay * (2 ** (retry_count - 1)), max_retry_delay)
            logger.error(f"Connection error: {conn_error}")
            logger.info(f"Reconnection attempt {retry_count}/{max_retries} in {delay} seconds...")
            await asyncio.sleep(delay)

        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generic RabbitMQ consumer boilerplate - connects to queue and processes messages',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Configuration file format (JSON):
{
    "work_queue": "queue/name",
    "host": "localhost",
    "port": 5672,
    "username": "guest",
    "password": "guest"
}

Message format (JSON):
{
    "job_id": "uuid",
    "data": "message data"
}
        '''
    )
    parser.add_argument(
        'config_file',
        type=str,
        help='Path to the JSON configuration file'
    )

    args = parser.parse_args()
    config = load_config(args.config_file)

    logger.info(f"Loaded configuration from: {args.config_file}")
    logger.info(f"Work queue: {config['work_queue']}")
    logger.info(f"RabbitMQ host: {config['host']}:{config['port']}")
    logger.info(f"Username: {config['username']}")

    try:
        asyncio.run(main(config))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
