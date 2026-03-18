import numpy as np
import aio_pika
import argparse
import torch
import time
import asyncio

from aiormq import ChannelInvalidStateError, ChannelClosed, AMQPError
from pamqp.commands import Basic

import serialization
from logger import get_logger
from utils import load_config, create_ssl_context, publish_event
from wire_formats import WhisperJobDescription, WhisperResult, WhisperTimings

logger = get_logger("whisper")

def load_whisper_model(device):
    """
    Load the Whisper model.
    
    Returns:
        Whisper model instance
    """
    try:
        import whisper
    except AttributeError:
        raise ImportError(
            "Please install the correct Whisper package using: pip install openai-whisper\n"
            "If you have the 'whisper' package installed, first uninstall it with: pip uninstall whisper"
        )
    
    logger.info(f"Loading Whisper large model on device: {device}")
    start_time = time.time()
    model = whisper.load_model("large", device=device)
    end_time = time.time()
    logger.info(f"Whisper model loaded in {end_time - start_time:.2f} seconds")
    
    return model


def perform_transcription(audio_data, whisper_model, temperature=0.2, language='en', 
                         initial_prompt=None, word_timestamps=True):
    """
    Perform transcription on the audio data.
    
    Args:
        audio_data: Numpy array containing the audio waveform
        whisper_model: Loaded Whisper model
        temperature: Temperature for sampling (default: 0.2)
        language: Language code (default: 'en')
        initial_prompt: Optional initial prompt for the model
        word_timestamps: Whether to include word-level timestamps (default: True)
    
    Returns:
        Transcription result dictionary
    """
    logger.info(f"Starting transcription on audio with shape {audio_data.shape}")
    start_time = time.time()
    
    # Build transcription parameters
    transcribe_params = {
        'temperature': temperature,
        'word_timestamps': word_timestamps,
        'fp16': False,  # MPS doesn't support fp16; avoids NaN logits
    }
    
    if language:
        transcribe_params['language'] = language
    
    if initial_prompt:
        transcribe_params['initial_prompt'] = initial_prompt
    
    # Perform transcription
    result = whisper_model.transcribe(audio_data, **transcribe_params)
    
    end_time = time.time()
    logger.info(f"Transcription completed in {end_time - start_time:.2f} seconds")
    
    return result


async def process_message(message: aio_pika.abc.AbstractIncomingMessage, whisper_model, destination_queue, config: dict):
    """
    Process a transcription job message from RabbitMQ.
    """
    logger.info("Received message")
    job_desc: WhisperJobDescription = serialization.load(message.body.decode())
    filename = job_desc.transcript_metadata.filename
    segment_count = job_desc.segment_count
    total_segments = job_desc.total_segments
    # Convert audio data from list back to numpy array
    audio_data = np.array(job_desc.audio_segment.audio, dtype=np.float32)

    if len(audio_data) == 0:
        raise ValueError("Empty audio data received")

    logger.info(f"Audio data shape: {audio_data.shape}, duration: ~{len(audio_data) / 16000:.2f}s")

    # Run transcription in thread pool to prevent blocking the event loop
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        perform_transcription,
        audio_data,
        whisper_model,
        job_desc.temperature,
        job_desc.language,
        None,
        # job_desc.word_timestamps
        False,
    )

    # Prepare response
    response = WhisperResult(
        transcript=result,
        speaker=job_desc.audio_segment.speaker,
        timings=WhisperTimings(start=job_desc.audio_segment.start, end=job_desc.audio_segment.end),
        transcript_metadata=job_desc.transcript_metadata,
        segment_count=job_desc.segment_count,
        total_segments=job_desc.total_segments,
    )

    # Get channel from message with error handling for invalid state
    publish_succeeded = False
    try:
        channel = message.channel
        await channel.basic_publish(
            body=serialization.dumps(response).encode(),
            exchange="",
            routing_key=destination_queue,
            properties=Basic.Properties(
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            ),
        )
        publish_succeeded = True
    except (ChannelInvalidStateError, ChannelClosed) as channel_error:
        logger.error(f"Channel error while sending response: {channel_error}")
        logger.info("Message will be re-queued for retry")
        await publish_event(
            config,
            f"WHISPER_PUBLISH_FAILED_NACK: {filename} segment {segment_count}/{total_segments} "
            f"publish to {destination_queue} failed ({channel_error}), nacking with requeue. "
            f"No duplicate expected since publish did not succeed."
        )
        # Nack the message so it gets requeued
        await message.nack(requeue=True)
        return

    logger.info(f"Job completed and response sent to {destination_queue}")

    # Acknowledge successful processing - result was already published above
    try:
        await message.ack()
    except (ChannelInvalidStateError, ChannelClosed, Exception) as ack_error:
        # CRITICAL: Result was already published to destination_queue but ack failed.
        # RabbitMQ will redeliver this message, causing a DUPLICATE downstream.
        await publish_event(
            config,
            f"WHISPER_ACK_FAILED_AFTER_PUBLISH: {filename} segment {segment_count}/{total_segments} "
            f"result was ALREADY PUBLISHED to {destination_queue}, but ack failed ({ack_error}). "
            f"RabbitMQ will redeliver this message, causing DUPLICATE processing downstream."
        )
        raise  # let outer handler reconnect

async def main(config):
    """
    Main function to start the whisper transcription consumer with reconnection logic.
    Handles connection failures and automatically reconnects with exponential backoff.
    """
    logger.info("Initializing Whisper transcription consumer...")

    # Retry configuration
    max_retries = 10
    base_retry_delay = 2  # seconds
    max_retry_delay = 60  # seconds

    # Load the Whisper model once at startup
    logger.info("Loading Whisper model...")
    whisper_model = load_whisper_model(device)
    
    ssl_context = create_ssl_context()
    # If using self-signed certificates, uncomment:
    # ssl_context = create_ssl_context(verify=False)
    
    retry_count = 0
    
    while True:
        try:
            # Connect to RabbitMQ with TLS
            logger.info(f"Connecting to RabbitMQ at {config['host']}:{config['port']}...")
            
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
                
                # Set QoS to process one message at a time
                await channel.set_qos(prefetch_count=1)
                
                # Declare the work queue
                work_queue = config['work_queue']
                results_queue = config["results_queue"]
                queue, _ = await asyncio.gather(
                    channel.declare_queue(work_queue, durable=True),
                    channel.declare_queue(results_queue, durable=True)
                )
                
                logger.info(f"Successfully connected! Listening for transcription jobs on queue: {work_queue}")
                logger.info("Waiting for transcription jobs. To exit press CTRL+C")
                
                # Start consuming messages
                async with queue.iterator() as queue_iter:
                    async for message in queue_iter:
                        try:
                            await process_message(message, whisper_model, results_queue, config)
                        except (ChannelInvalidStateError, ChannelClosed) as channel_err:
                            logger.error(f"Channel error during message processing: {channel_err}")
                            logger.info("Will attempt to reconnect...")
                            # Break out of the message loop to reconnect
                            raise
                        except Exception as e:
                            logger.error(f"Unexpected error processing message: {e}", exc_info=True)
                            # Continue processing other messages
                            
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
        description='RabbitMQ consumer for Whisper transcription jobs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Configuration file format (JSON):
{
    "work_queue": "whisper/large",
    "host": "localhost",
    "port": 5672,
    "username": "guest",
    "password": "guest"
}
        '''
    )
    parser.add_argument(
        'config_file',
        type=str,
        help='Path to the JSON configuration file'
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to run the Whisper model on (e.g. 'cpu', 'cuda', 'cuda:0'). If not specified, will auto-detect.",
    )
    
    args = parser.parse_args()
    logger.info(f"{args=}")
    logger.info(f"args.device = {args.device}")
    device_str = args.device
    if not device_str:
        device_str = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    logger.info(f"{device_str=}")
    device = torch.device(device_str)
    
    # Load configuration from file
    config = load_config(args.config_file)
    
    logger.info(f"Loaded configuration from: {args.config_file}")
    logger.info(f"Work queue: {config['work_queue']}")
    logger.info(f"RabbitMQ host: {config['host']}:{config['port']}")
    logger.info(f"Username: {config['username']}")

    try:
        asyncio.run(main(config))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
