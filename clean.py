import json
import uuid
import aio_pika
import argparse
import asyncio

from aio_pika.abc import AbstractIncomingMessage, AbstractRobustConnection
from aiormq import AMQPError, ChannelInvalidStateError, ChannelClosed
from pamqp.commands import Basic
from transformers import AutoTokenizer, Qwen2Tokenizer

import serialization
from utils import load_config, create_ssl_context, get_answer
from wire_formats import WhisperResult, LLMPromptJob, LLMPromptResponse, Metaparams, CleanedWhisperResult


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
    connection: AbstractRobustConnection,
    results_queue: str,
    semaphore: asyncio.Semaphore,
):
    """
    Process an LLM cleanup job message from RabbitMQ.
    """
    try:
        print(f"Received message")
        whisper_result: WhisperResult = serialization.load(message.body.decode())

        # Normal segment processing
        segment_count = whisper_result.segment_count
        total_segments = whisper_result.total_segments
        text = whisper_result.transcript.get("text", "")

        print(f"Processing job, segment {segment_count}/{total_segments}")
        print(f"length of transcript text: {len(text)}")

        if len(text) < 50:
            print(f"Transcript text is very short, skipping cleanup and sending original text")
            cleaned_text = text
        else:
            # Create prompt and send to LLM queue
            prompt = create_cleanup_prompt(text)
            generated = await send_to_llm_queue(
                connection,
                prompt,
                filename=f"segment_{segment_count}"
            )

            if generated is None:
                print(f"Error: No response from LLM queue for segment {segment_count}")
                await message.nack(requeue=True)
                return

            print(f"generated: {generated}")

            cleaned_text = get_answer(generated, "```corrected_transcript", "```")

        if cleaned_text:
            print(f"cleaned_text={cleaned_text}")
        else:
            print(f"Warning: Could not extract cleaned text, using original")
            cleaned_text = text

        # Prepare response
        response = CleanedWhisperResult(cleaned_transcript=cleaned_text, whisper_result=whisper_result)
        await message.channel.basic_publish(
            body=serialization.dumps(response).encode(),
            exchange="",
            routing_key=results_queue,
            properties=Basic.Properties(
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            ),
        )

        print(f"segment {segment_count} completed and response sent to {results_queue}")

        # Acknowledge successful processing
        await message.ack()
    finally:
        # Always release the semaphore when done
        semaphore.release()


async def main(config):
    """
    Main function to start the LLM cleanup consumer with reconnection logic.
    Handles connection failures and automatically reconnects with exponential backoff.
    """
    print("Initializing LLM cleanup consumer...")
    
    # Retry configuration
    max_retries = 10
    base_retry_delay = 2  # seconds
    max_retry_delay = 60  # seconds

    ssl_context = create_ssl_context()
    retry_count = 0
    
    while True:
        try:
            # Connect to RabbitMQ with TLS
            print(f"Connecting to RabbitMQ at {config['host']}:{config['port']}...")
            
            connection = await aio_pika.connect_robust(
                host=config['host'],
                port=config['port'],
                login=config['username'],
                password=config['password'],
                ssl=True,
                ssl_context=ssl_context,
            )
            
            # Reset retry count on successful connection
            retry_count = 0
            
            async with connection:
                # Create channel
                channel = await connection.channel()
                
                # Set QoS to process three messages concurrently
                await channel.set_qos(prefetch_count=3)
                
                # Declare the work queue
                work_queue = config['work_queue']
                results_queue = config["results_queue"]
                queue, _ = await asyncio.gather(
                    channel.declare_queue(work_queue, durable=True),
                    channel.declare_queue(results_queue, durable=True),
                )
                
                print(f"Successfully connected! Listening for LLM cleanup jobs on queue: {work_queue}")
                print(f"Processing up to 3 messages concurrently")
                print("Waiting for cleanup jobs. To exit press CTRL+C")
                
                # Semaphore to limit concurrent processing to 3
                semaphore = asyncio.Semaphore(3)
                
                # Start consuming messages
                async with queue.iterator() as queue_iter:
                    async for message in queue_iter:
                        # Acquire semaphore before creating task (blocks if 3 tasks already running)
                        await semaphore.acquire()
                        
                        # Create task for concurrent processing
                        # Task will release semaphore in its finally block
                        asyncio.create_task(process_message(message, connection, results_queue, semaphore))
                            
        except (AMQPError, ChannelInvalidStateError, ChannelClosed, ConnectionError) as conn_error:
            retry_count += 1
            if retry_count > max_retries:
                print(f"Max retries ({max_retries}) exceeded. Giving up.")
                raise
            
            # Calculate exponential backoff delay
            delay = min(base_retry_delay * (2 ** (retry_count - 1)), max_retry_delay)
            print(f"Connection error: {conn_error}")
            print(f"Reconnection attempt {retry_count}/{max_retries} in {delay} seconds...")
            await asyncio.sleep(delay)
            
        except KeyboardInterrupt:
            print("\nShutting down gracefully...")
            break
        except Exception as e:
            print(f"Unexpected error in main loop: {e}")
            import traceback
            traceback.print_exc()
            
            retry_count += 1
            if retry_count > max_retries:
                print(f"Max retries ({max_retries}) exceeded. Giving up.")
                raise
            
            delay = min(base_retry_delay * (2 ** (retry_count - 1)), max_retry_delay)
            print(f"Retrying in {delay} seconds...")
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

    args = parser.parse_args()

    # Load configuration from file
    config = load_config(args.config_file)

    print(f"Loaded configuration from: {args.config_file}")
    print(f"Work queue: {config['work_queue']}")
    print(f"Results queue: {config.get('results_queue', 'clean/results')}")
    print(f"RabbitMQ host: {config['host']}:{config['port']}")
    print(f"Username: {config['username']}")
    print(f"Note: Sending prompts to llm/qwen32 queue for processing")

    # Run main with config
    try:
        asyncio.run(main(config))
    except KeyboardInterrupt:
        print("\nInterrupted by user")
