import json
import uuid
from json import JSONDecodeError
from pathlib import Path

import aio_pika
import argparse
import asyncio

from aio_pika.abc import AbstractIncomingMessage, AbstractRobustChannel, AbstractRobustConnection
from transformers import AutoTokenizer, Qwen2Tokenizer

import serialization
import wire_formats
from utils import load_config, create_ssl_context, get_answer
from wire_formats import Metaparams


def create_solr_prompt(filename: str, tokenizer: Qwen2Tokenizer) -> str:
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

async def process_message(
    message: AbstractIncomingMessage,
    connection: AbstractRobustConnection,
    tokenizer: Qwen2Tokenizer,
) -> wire_formats.LLMPromptResponse | None:
    # Parse the incoming message
    body = json.loads(message.body.decode())
    filename = body.get("filename")
    prompt = create_solr_prompt(filename, tokenizer)
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



async def main(config):
    """
    Main function to start the RabbitMQ consumer with reconnection logic.
    Handles connection failures and automatically reconnects with exponential backoff.
    Processes up to 3 messages concurrently using a semaphore.
    """
    print("Initializing RabbitMQ consumer...")
    
    ssl_context = create_ssl_context()
    
    print(f"Connecting to RabbitMQ at {config['host']}:{config['port']}...")
    tokenizer: Qwen2Tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")
    connection = await aio_pika.connect_robust(
        host=config['host'],
        port=config['port'],
        login=config['username'],
        password=config['password'],
        ssl=True,
        ssl_context=ssl_context,
    )

    async with connection:
        channel: AbstractRobustChannel = await connection.channel()
        await channel.set_qos(prefetch_count=3)

        work_queue = config['work_queue']
        result_queue = config["result_queue"]
        work_queue = await channel.declare_queue(work_queue, durable=True)
        await channel.declare_queue(result_queue, durable=True)

        print(f"Successfully connected! Listening for jobs on queue: {work_queue}")
        print("Waiting for messages (processing up to 3 concurrently). To exit press CTRL+C")

        # Create a semaphore to limit concurrent message processing to 3
        semaphore = asyncio.Semaphore(3)
        tasks = set()

        async def process_with_semaphore(message: AbstractIncomingMessage):
            """Process a single message with semaphore control"""
            async with semaphore:
                try:
                    resp: wire_formats.LLMPromptResponse = await process_message(message, connection, tokenizer)
                    answer_str = get_answer(resp.generated_text, start_delim="```json", end_delim="```")
                    try:
                        answer = wire_formats.TranscriptMetadata(**json.loads(answer_str))
                    except TypeError:
                        print(f"Failed to parse answer JSON: {answer_str}")
                        return
                    except JSONDecodeError as jde:
                        print(f"Invalid JSON format in answer: {answer_str}\nError: {jde}")
                        await message.nack()
                        return
                    
                    async with await connection.channel() as pub_channel:
                        await pub_channel.default_exchange.publish(
                            aio_pika.Message(
                                body=serialization.dumps(answer).encode("utf-8"),
                            ),
                            routing_key=result_queue,
                        )
                    await message.ack()
                except Exception as e:
                    print(f"Error processing message: {e}")
                    await message.nack(requeue=True)

        async with work_queue.iterator() as queue_iter:
            async for message in queue_iter:
                # Create a task for processing the message
                task = asyncio.create_task(process_with_semaphore(message))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
                
                # Clean up completed tasks
                tasks = {t for t in tasks if not t.done()}


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

    print(f"Loaded configuration from: {args.config_file}")
    print(f"Work queue: {config['work_queue']}")
    print(f"RabbitMQ host: {config['host']}:{config['port']}")
    print(f"Username: {config['username']}")

    try:
        asyncio.run(main(config))
    except KeyboardInterrupt:
        print("\nInterrupted by user")
