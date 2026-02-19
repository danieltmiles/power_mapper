import json
import pickle
import base64
import traceback
from pathlib import Path
from typing import Any

import aio_pika
import argparse
import torch
import time
import asyncio


import pyannote.audio
from aio_pika.abc import AbstractIncomingMessage, AbstractRobustChannel
from aiormq import ChannelInvalidStateError, ChannelClosed, AMQPError
from pyannote.audio.tasks import SpeakerDiarization

import serialization
from wire_formats import TranscriptMetadata, DiarizationResponse

torch.serialization.add_safe_globals([pyannote.audio.core.task.Specifications])
torch.serialization.safe_globals([pyannote.audio.core.task.Problem])
torch.serialization.add_safe_globals([pyannote.audio.core.task.Problem])
torch.serialization.add_safe_globals([pyannote.audio.core.task.Resolution])
from pamqp.commands import Basic
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_diarization import DiarizeOutput
from pyannote.audio.core.task import Specifications

import shared_disks
from utils import normalize_audio, load_config, create_ssl_context, load_hf_token

# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

def perform_diarization(waveform, sample_rate, pipeline):
    """
    Perform speaker diarization on the audio waveform.
    
    Args:
        waveform: PyTorch tensor containing the audio waveform (should be 2D: channel, time)
        sample_rate: Sample rate of the audio
        pipeline: Pyannote diarization pipeline
    
    Returns:
        DiarizeOutput object containing diarization results
    """
    print(f"Starting diarization on waveform with shape {waveform.shape}, sample_rate={sample_rate}")
    start_time = time.time()
    
    # Ensure waveform is 2D (channel, time) as required by pyannote
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    # Perform diarization
    diarization: DiarizeOutput = pipeline({"waveform": waveform, "sample_rate": sample_rate})
    
    end_time = time.time()
    print(f"Diarization completed in {end_time - start_time:.2f} seconds")
    
    return diarization


def serialize_diarization(diarization):
    """
    Serialize DiarizeOutput object to a format that can be sent via RabbitMQ.
    
    Returns:
        Base64-encoded pickle of the diarization object
    """
    pickled = pickle.dumps(diarization)
    encoded = base64.b64encode(pickled).decode('utf-8')
    return encoded


async def process_message(
    message: AbstractIncomingMessage,
    pipeline: SpeakerDiarization,
    channel: AbstractRobustChannel,
    destination_queue_name: str,
    config: dict[str, Any],
):
    """
    Process a diarization job message from RabbitMQ.
    """
    print(f"Received message")
    transcript_metadata: TranscriptMetadata = serialization.load(message.body.decode())

    storage_info = config.get("storage_info")
    remote_storage: shared_disks.RemoteStorage = shared_disks.factory(
        remote_file_type=storage_info.get("type"),
        info=storage_info,
    )
    local_filename = f"/tmp/{transcript_metadata.filename}"
    print("retrieving remote file")
    remote_storage.retrieve(Path(transcript_metadata.filename).name, local_filename)
    print("retrieved remote file")
    signal, sr = normalize_audio(local_filename)

    # Move waveform to appropriate device
    if isinstance(signal, torch.Tensor):
        signal = signal.to(torch.device(device))

    # Run diarization in thread pool to prevent blocking the event loop
    loop = asyncio.get_event_loop()
    diarization = await loop.run_in_executor(
        None,
        perform_diarization,
        signal,
        sr,
        pipeline
    )

    # Serialize result
    diarization_encoded = serialize_diarization(diarization)

    # Prepare response
    response = DiarizationResponse(diarization_encoded, transcript_metadata)
    routing_key = destination_queue_name

    # Get channel from message with error handling for invalid state
    try:
        await channel.declare_queue(routing_key, durable=True)
        await channel.default_exchange.publish(
            aio_pika.Message(
                body=serialization.dumps(response).encode("utf-8"),
            ),
            routing_key=routing_key,
        )
    except (ChannelInvalidStateError, ChannelClosed) as channel_error:
        print(f"Channel error while sending response {channel_error}")
        print(f"Message will be re-queued for retry")
        # Nack the message so it gets requeued
        await message.nack(requeue=True)
        return

    print(f"Job completed and response sent to {routing_key}")

    # Acknowledge successful processing
    await message.ack()

async def main(config):
    """
    Main function to start the diarization consumer with reconnection logic.
    Handles connection failures and automatically reconnects with exponential backoff.
    """
    print("Initializing diarization consumer...")
    print(f"Loading diarization pipeline...")
    
    # Retry configuration
    max_retries = 10
    base_retry_delay = 2  # seconds
    max_retry_delay = 60  # seconds
    
    # Load the diarization pipeline once at startup
    start = time.time()
    pipeline: SpeakerDiarization = Pipeline.from_pretrained(
        checkpoint="pyannote/speaker-diarization-community-1",
        token=load_hf_token(),
    ).to(torch.device(device))
    end = time.time()
    print(f"Pipeline loaded in {end - start:.2f} seconds")
    
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

            async with connection:
                channel: AbstractRobustChannel = await connection.channel()

                # Set QoS to process one message at a time
                await channel.set_qos(prefetch_count=1)
                
                # Declare the work queue
                work_queue = config['work_queue']
                destination_queue = config.get("destination_queue", "diarizations")
                queue, _ = await asyncio.gather(
                    channel.declare_queue(work_queue, durable=True),
                    channel.declare_queue(destination_queue, durable=True),
                )

                print(f"Successfully connected! Listening for diarization jobs on queue: {work_queue}")
                print("Waiting for diarization jobs. To exit press CTRL+C")
                
                # Start consuming messages
                async with queue.iterator() as queue_iter:
                    async for message in queue_iter:
                        try:
                            await process_message(message, pipeline, channel, destination_queue, config)
                        except (ChannelInvalidStateError, ChannelClosed) as channel_err:
                            print(f"Channel error during message processing: {channel_err}")
                            raise
                        except Exception as e:
                            print(f"Unexpected error processing message: {e}")
                            import traceback
                            tb_str = traceback.format_exc()
                            print(tb_str)
                            dead_letter_key = "dead_letter"
                            dead_letter_body = {
                                "message": message.body.decode(),
                                "error": tb_str,
                                "pipeline_step": "dads",
                            }
                            await channel.declare_queue(dead_letter_key, durable=True)
                            await channel.default_exchange.publish(
                                aio_pika.Message(
                                    body=json.dumps(dead_letter_body).encode("utf-8"),
                                ),
                                routing_key=dead_letter_key,
                            )
                            if not message.processed:
                                print("Message that caused error was not acknowledged. Acknowledging to prevent reprocessing.")
                                await message.ack()

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
        description='RabbitMQ consumer for diarization jobs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Configuration file format (JSON):
{
    "work_queue": "diarization_work",
    "response_queue": "diarization_response",
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
    
    args = parser.parse_args()
    
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
