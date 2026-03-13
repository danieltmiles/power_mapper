import argparse
import asyncio
import uuid

import aio_pika
from aio_pika.abc import AbstractIncomingMessage, AbstractRobustConnection
from aiormq import AMQPError, ChannelInvalidStateError, ChannelClosed

import serialization
from cached_iterator import CachedMessageIterator
from logger import get_logger
from utils import load_config, get_answer, dial_rabbit_from_config, dial_redis_from_config
from wire_formats import WhisperResult, LLMPromptJob, LLMPromptResponse, Metaparams, CleanedWhisperResult

logger = get_logger("clean")


def create_cleanup_prompt(text: str) -> str:
    """
    Create a prompt for the journalistic courtesy (cleanup) task.

    Args:
        text: Raw transcript text to clean up

    Returns:
        str: Formatted prompt for the LLM
    """
    prompt_template = """<transcript_to_correct>
{text}
</transcript_to_correct>

You are a transcript editor. Your job is to fix ASR (speech recognition) errors while preserving how people actually speak.

PRESERVE these natural speech features:
- False starts and self-corrections ("I think, uh, I mean...")
- Filler words (um, uh, like, you know)
- Repetitions for emphasis ("very very important")
- Incomplete sentences and run-ons
- Grammatical errors that occur in natural speech

ONLY FIX these ASR errors:
- Misheard words that don't make sense in context
- Randomly inserted words that break sentence flow
- Missing words that make sentences incomprehensible
- Obvious word substitutions (homophones, similar-sounding words)

Do not make the speech more formal or grammatically correct. The goal is an accurate transcript of what was said, not cleaned-up prose.

Return ONLY the corrected transcript without explanations, stop when finished. End with a text block containing the corrected transcript, formatted like this:
```corrected_transcript
[corrected transcript here]
```
"""
    text = text.replace("...", "")
    return prompt_template.format(text=text.strip())

async def send_to_llm_queue(
    connection: AbstractRobustConnection,
    prompt: str,
    filename: str,
) -> str | None:
    """
    Send a prompt to the llm/qwen32 queue and wait for the response.
    Returns the generated text or None if failed.
    """
    reply_to = f"clean/reply/{str(uuid.uuid4())}"
    job = LLMPromptJob(
        job_id=str(uuid.uuid4()),
        filename=filename,
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
        # Declare reply queue
        response_queue = await channel.declare_queue(reply_to, durable=True)
        
        # Send job to llm/qwen32 queue
        await channel.default_exchange.publish(
            aio_pika.Message(
                body=serialization.dumps(job).encode("utf-8"),
            ),
            routing_key="llm/qwen32",
        )
        
        # Wait for response
        async with response_queue.iterator() as queue_iter:
            async for response_message in queue_iter:
                async with response_message.process():
                    response: LLMPromptResponse = serialization.load(
                        response_message.body.decode(),
                        cls=LLMPromptResponse,
                    )
                    # Clean up the reply queue
                    await channel.queue_delete(reply_to)
                    return response.generated_text
        
        # Clean up if no response received
        await channel.queue_delete(reply_to)
        return None


async def process_message(
    message: AbstractIncomingMessage,
    results_queue: str,
    config: dict,
):
    """
    Process an LLM cleanup job message from RabbitMQ.
    """
    logger.info("Received message")
    whisper_result: WhisperResult = serialization.load(message.body.decode())

    # Normal segment processing
    segment_count = whisper_result.segment_count
    total_segments = whisper_result.total_segments
    text = whisper_result.transcript.get("text", "")

    logger.info(f"Processing job, segment {segment_count}/{total_segments}")
    logger.info(f"length of transcript text: {len(text)}")

    if len(text) < 50:
        logger.info("Transcript text is very short, skipping cleanup and sending original text")
        cleaned_text = text
    else:
        # Create prompt and send to LLM queue - dial fresh connection for this
        async with await dial_rabbit_from_config(config) as rabbitmq_connection:
            prompt = create_cleanup_prompt(text)
            generated = await send_to_llm_queue(
                rabbitmq_connection,
                prompt,
                filename=f"segment_{segment_count}"
            )

            if generated is None:
                logger.error(f"No response from LLM queue for segment {segment_count}")
                return

            logger.info(f"generated: {generated}")

            cleaned_text = get_answer(generated, "```corrected_transcript", "```")

    if cleaned_text:
        logger.info(f"cleaned_text={cleaned_text}")
    else:
        logger.info("Could not extract cleaned text, using original")
        cleaned_text = text

    # Prepare response and publish with fresh connection
    response = CleanedWhisperResult(cleaned_transcript=cleaned_text, whisper_result=whisper_result)
    async with await dial_rabbit_from_config(config) as rabbitmq_connection:
        async with await rabbitmq_connection.channel() as channel:
            await channel.default_exchange.publish(
                aio_pika.Message(
                    body=serialization.dumps(response).encode(),
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                ),
                routing_key=results_queue,
            )

    logger.info(f"segment {segment_count} completed and response sent to {results_queue}")


async def main(config, concurrent: int = 3):
    """
    Main function to start the LLM cleanup consumer with reconnection logic.
    Handles connection failures and automatically reconnects with exponential backoff.
    
    Args:
        config: Configuration dictionary
        concurrent: Number of concurrent message processors
    """
    logger.info("Initializing LLM cleanup consumer...")
    logger.info(f"Concurrent processors: {concurrent}")
    
    # Retry configuration
    max_retries = 10
    base_retry_delay = 2  # seconds
    max_retry_delay = 60  # seconds
    
    retry_count = 0
    
    while True:
        try:
            # Connect to RabbitMQ with TLS
            logger.info(f"Connecting to RabbitMQ at {config['host']}:{config['port']}...")
            connection = await dial_rabbit_from_config(config)
            redis_client = await dial_redis_from_config(config)
            
            # Reset retry count on successful connection
            retry_count = 0
            
            async with connection:
                async with await connection.channel() as channel:
                    # Declare the work queue
                    work_queue = config['work_queue']
                    results_queue = config["results_queue"]
                    await asyncio.gather(
                        channel.declare_queue(work_queue, durable=True),
                        channel.declare_queue(results_queue, durable=True),
                    )
                
                logger.info(f"Successfully connected! Listening for LLM cleanup jobs on queue: {work_queue}")
                logger.info("Waiting for cleanup jobs. To exit press CTRL+C")
                
                # Track active processing tasks
                active_tasks = set()

                async def process_and_mark(msg, queue_iter):
                    """Process a message and mark it as processed."""
                    async with queue_iter.processing(msg):
                        await process_message(msg, results_queue, config)

                # Start consuming messages
                async with CachedMessageIterator(
                        rabbitmq_connection=connection,
                        redis_client=redis_client,
                        queue_name=work_queue,
                        redis_key_prefix="backup:clean",
                        config=config,
                ) as queue_iter:
                    async for message in queue_iter:
                        # Wait if we're a/t capacity
                        while len(active_tasks) >= concurrent:
                            # Wait for at least one task to complete
                            done, active_tasks = await asyncio.wait(
                                active_tasks,
                                return_when=asyncio.FIRST_COMPLETED
                            )
                        
                        # Create task for concurrent processing
                        task = asyncio.create_task(process_and_mark(message, queue_iter))
                        active_tasks.add(task)
                    
                    # Wait for all remaining tasks to complete
                    if active_tasks:
                        await asyncio.gather(*active_tasks, return_exceptions=True)
                            
        except (AMQPError, ChannelInvalidStateError, ChannelClosed, ConnectionError) as conn_error:
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
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}", exc_info=True)

            retry_count += 1
            if retry_count > max_retries:
                logger.error(f"Max retries ({max_retries}) exceeded. Giving up.")
                raise

            delay = min(base_retry_delay * (2 ** (retry_count - 1)), max_retry_delay)
            logger.info(f"Retrying in {delay} seconds...")
            await asyncio.sleep(delay)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='RabbitMQ consumer for LLM text cleanup jobs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Configuration file format (JSON):
{
    "work_queue": "clean/whisper_results",
    "results_queue": "clean/results",
    "host": "localhost",
    "port": 5672,
    "username": "guest",
    "password": "guest"
}

Note: This service sends prompts to the llm/qwen32 queue for processing
instead of loading its own model.
        '''
    )
    parser.add_argument(
        'config_file',
        type=str,
        help='Path to the JSON configuration file'
    )
    parser.add_argument(
        '--concurrent',
        type=int,
        default=3,
        help='Number of concurrent message processors (default: 3)'
    )

    args = parser.parse_args()

    # Load configuration from file
    config = load_config(args.config_file)

    logger.info(f"Loaded configuration from: {args.config_file}")
    logger.info(f"Work queue: {config['work_queue']}")
    logger.info(f"Results queue: {config.get('results_queue', 'clean/results')}")
    logger.info(f"RabbitMQ host: {config['host']}:{config['port']}")
    logger.info(f"Username: {config['username']}")
    logger.info("Note: Sending prompts to llm/qwen32 queue for processing")

    try:
        asyncio.run(main(config, args.concurrent))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
