import pickle
import base64
from pathlib import Path
from typing import Any

import aio_pika
import argparse
import torch
import time
import asyncio


import pyannote.audio
from aio_pika.abc import AbstractIncomingMessage
from aiormq import ChannelInvalidStateError, ChannelClosed, AMQPError
from pyannote.audio.tasks import SpeakerDiarization

import serialization
from cached_iterator import CachedMessageIterator
from logger import get_logger
from wire_formats import TranscriptMetadata, DiarizationResponse

logger = get_logger("dads")

torch.serialization.add_safe_globals([pyannote.audio.core.task.Specifications])
torch.serialization.safe_globals([pyannote.audio.core.task.Problem])
torch.serialization.add_safe_globals([pyannote.audio.core.task.Problem])
torch.serialization.add_safe_globals([pyannote.audio.core.task.Resolution])
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_diarization import DiarizeOutput

import shared_disks
from utils import normalize_audio, load_config, create_ssl_context, load_hf_token, dial_rabbit_from_config, \
    dial_redis_from_config

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
    logger.info(f"Starting diarization on waveform with shape {waveform.shape}, sample_rate={sample_rate}")
    start_time = time.time()
    
    # Ensure waveform is 2D (channel, time) as required by pyannote
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    # Perform diarization
    diarization: DiarizeOutput = pipeline({"waveform": waveform, "sample_rate": sample_rate})
    
    end_time = time.time()
    logger.info(f"Diarization completed in {end_time - start_time:.2f} seconds")
    
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
    destination_queue_name: str,
    config: dict[str, Any],
):
    """
    Process a diarization job message from RabbitMQ.
    """
    logger.info("Received message")
    transcript_metadata: TranscriptMetadata = serialization.load(message.body.decode())

    storage_info = config.get("storage_info")
    remote_storage: shared_disks.RemoteStorage = shared_disks.factory(
        remote_file_type=storage_info.get("type"),
        info=storage_info,
    )
    local_filename = f"/tmp/{transcript_metadata.filename}"
    logger.info("retrieving remote file")
    remote_storage.retrieve(Path(transcript_metadata.filename).name, local_filename)
    logger.info("retrieved remote file")
    signal, sr = normalize_audio(local_filename)

    # Move waveform to appropriate device
    if isinstance(signal, torch.Tensor):
        signal = signal.to(torch.device(device))

    diarization = perform_diarization(signal, sr, pipeline)

    # Serialize result
    diarization_encoded = serialize_diarization(diarization)

    # Prepare response
    response = DiarizationResponse(diarization=diarization_encoded, transcript_metadata=transcript_metadata)
    routing_key = destination_queue_name

    async with await dial_rabbit_from_config(config) as rabbitmq_connection:
        async with await rabbitmq_connection.channel() as channel:
            resp = await channel.default_exchange.publish(
                aio_pika.Message(
                    body=serialization.dumps(response).encode("utf-8"),
                ),
                routing_key=routing_key,
            )
            logger.info(f"response from default exchange publish: {resp}")

    logger.info(f"Job completed and response sent to {routing_key}")

async def main(config):
    """
    Main function to start the diarization consumer with reconnection logic.
    Handles connection failures and automatically reconnects with exponential backoff.
    """
    logger.info("Initializing diarization consumer...")
    logger.info("Loading diarization pipeline...")
    
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
    logger.info(f"Pipeline loaded in {end - start:.2f} seconds")
    
    ssl_context = create_ssl_context()
    # If using self-signed certificates, uncomment:
    # ssl_context = create_ssl_context(verify=False)
    
    retry_count = 0
    
    while True:
        try:
            # Connect to RabbitMQ with TLS
            logger.info(f"Connecting to RabbitMQ at {config['host']}:{config['port']}...")
            connection = await dial_rabbit_from_config(config)
            redis_client = await dial_redis_from_config(config)
            async with connection:
                async with await connection.channel() as channel:
                    # Declare the work queue
                    work_queue = config['work_queue']
                    destination_queue = config.get("destination_queue", "diarizations")
                    await asyncio.gather(
                        channel.declare_queue(work_queue, durable=True),
                        channel.declare_queue(destination_queue, durable=True),
                    )

                logger.info(f"Successfully connected! Listening for diarization jobs on queue: {work_queue}")
                logger.info("Waiting for diarization jobs. To exit press CTRL+C")
                
                # Start consuming messages
                async with CachedMessageIterator(
                        rabbitmq_connection=connection,
                        redis_client=redis_client,
                        queue_name=work_queue,
                        redis_key_prefix="backup:dads",
                        config=config,
                ) as queue_iter:
                    async for message in queue_iter:
                        async with queue_iter.processing(message):
                            await process_message(message, pipeline, destination_queue, config)

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
    
    logger.info(f"Loaded configuration from: {args.config_file}")
    logger.info(f"Work queue: {config['work_queue']}")
    logger.info(f"RabbitMQ host: {config['host']}:{config['port']}")
    logger.info(f"Username: {config['username']}")

    try:
        asyncio.run(main(config))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
