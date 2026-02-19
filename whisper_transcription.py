import numpy as np
import aio_pika
import argparse
import torch
import time
import asyncio

from aiormq import ChannelInvalidStateError, ChannelClosed, AMQPError
from pamqp.commands import Basic

import serialization
from utils import load_config, create_ssl_context
from wire_formats import WhisperJobDescription, WhisperResult, WhisperTimings

def load_whisper_model(device: torch.Device):
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
    
    print(f"Loading Whisper large model on device: {device}")
    start_time = time.time()
    model = whisper.load_model("large", device=device)
    end_time = time.time()
    print(f"Whisper model loaded in {end_time - start_time:.2f} seconds")
    
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
    print(f"Starting transcription on audio with shape {audio_data.shape}")
    start_time = time.time()
    
    # Build transcription parameters
    transcribe_params = {
        'temperature': temperature,
        'word_timestamps': word_timestamps,
    }
    
    if language:
        transcribe_params['language'] = language
    
    if initial_prompt:
        transcribe_params['initial_prompt'] = initial_prompt
    
    # Perform transcription
    result = whisper_model.transcribe(audio_data, **transcribe_params)
    
    end_time = time.time()
    print(f"Transcription completed in {end_time - start_time:.2f} seconds")
    
    return result


async def process_message(message: aio_pika.abc.AbstractIncomingMessage, whisper_model, destination_queue):
    """
    Process a transcription job message from RabbitMQ.
    """
    print(f"Received message")
    job_desc: WhisperJobDescription = serialization.load(message.body.decode())
    # Convert audio data from list back to numpy array
    audio_data = np.array(job_desc.audio_segment.audio, dtype=np.float32)

    if len(audio_data) == 0:
        raise ValueError("Empty audio data received")

    print(f"Audio data shape: {audio_data.shape}, duration: ~{len(audio_data) / 16000:.2f}s")

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
    except (ChannelInvalidStateError, ChannelClosed) as channel_error:
        print(f"Channel error while sending response: {channel_error}")
        print(f"Message will be re-queued for retry")
        # Nack the message so it gets requeued
        await message.nack(requeue=True)
        return

    print(f"Job completed and response sent to {destination_queue}")

    # Acknowledge successful processing
    await message.ack()

async def main(config):
    """
    Main function to start the whisper transcription consumer with reconnection logic.
    Handles connection failures and automatically reconnects with exponential backoff.
    """
    print("Initializing Whisper transcription consumer...")
    
    # Retry configuration
    max_retries = 10
    base_retry_delay = 2  # seconds
    max_retry_delay = 60  # seconds
    
    # Load the Whisper model once at startup
    print("Loading Whisper model...")
    whisper_model = load_whisper_model(device)
    
    ssl_context = create_ssl_context()
    # If using self-signed certificates, uncomment:
    # ssl_context = create_ssl_context(verify=False)
    
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
                
                # Set QoS to process one message at a time
                await channel.set_qos(prefetch_count=1)
                
                # Declare the work queue
                work_queue = config['work_queue']
                results_queue = config["results_queue"]
                queue, _ = await asyncio.gather(
                    channel.declare_queue(work_queue, durable=True),
                    channel.declare_queue(results_queue, durable=True)
                )
                
                print(f"Successfully connected! Listening for transcription jobs on queue: {work_queue}")
                print("Waiting for transcription jobs. To exit press CTRL+C")
                
                # Start consuming messages
                async with queue.iterator() as queue_iter:
                    async for message in queue_iter:
                        try:
                            await process_message(message, whisper_model, results_queue)
                        except (ChannelInvalidStateError, ChannelClosed) as channel_err:
                            print(f"Channel error during message processing: {channel_err}")
                            print("Will attempt to reconnect...")
                            # Break out of the message loop to reconnect
                            raise
                        except Exception as e:
                            print(f"Unexpected error processing message: {e}")
                            import traceback
                            traceback.print_exc()
                            # Continue processing other messages
                            
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
    device_str = args.device or "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    
    # Load configuration from file
    config = load_config(args.config_file)
    
    print(f"Loaded configuration from: {args.config_file}")
    print(f"Work queue: {config['work_queue']}")
    print(f"RabbitMQ host: {config['host']}:{config['port']}")
    print(f"Username: {config['username']}")
    
    # Run main with config
    try:
        asyncio.run(main(config))
    except KeyboardInterrupt:
        print("\nInterrupted by user")
