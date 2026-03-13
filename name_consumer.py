import json

import aio_pika
import argparse
import asyncio
import httpx

from aio_pika.abc import AbstractIncomingMessage
from aiormq import ChannelInvalidStateError, ChannelClosed, AMQPError

import serialization
import wire_formats
from cached_iterator import CachedMessageIterator
from logger import get_logger
from utils import load_config, get_answer, dial_rabbit_from_config, dial_redis_from_config

logger = get_logger("name_consumer")


async def update_speakers(filename: str, new_speakers: dict, config: dict):
    solr_config = config['solr']
    solr_url = f"{solr_config['url']}/transcripts"
    auth = httpx.BasicAuth(solr_config['username'], solr_config['password'])

    async with httpx.AsyncClient(auth=auth) as client:
        for speaker_id, data in new_speakers.items():
            start = 0
            page_size = 1000
            while True:
                params = {
                    'q': '*:*',
                    'fq': [f'filename:"{filename}"', f'speaker_name:"{speaker_id}"'],
                    'fl': 'id,speaker_confidence',
                    'start': start,
                    'rows': page_size,
                    'wt': 'json',
                }
                response = await client.get(f'{solr_url}/select', params=params)
                response.raise_for_status()
                result = response.json()['response']
                docs = result['docs']

                docs_to_update = [
                    {
                        'id': doc['id'],
                        'speaker_name': {'set': data['name']},
                        'speaker_confidence': {'set': data['confidence']},
                    }
                    for doc in docs
                    if doc.get('speaker_confidence', 0) < data['confidence'] and data['name'].lower().strip() != "unknown"
                ]

                if docs_to_update:
                    response = await client.post(
                        f'{solr_url}/update',
                        params={'commit': 'true'},
                        headers={'Content-Type': 'application/json'},
                        content=json.dumps(docs_to_update),
                    )
                    response.raise_for_status()
                    logger.info(f"updated {len(docs_to_update)} speaker sections from {filename} with:\n{json.dumps(new_speakers, indent=4)}")

                start += page_size
                if start >= result['numFound']:
                    break

async def process_message(message: AbstractIncomingMessage, config: dict):
    response: wire_formats.LLMPromptResponse = serialization.load(message.body.decode())
    filename = response.filename
    generated_text = response.generated_text
    sequence_number: int | None = None
    num_sequences: int | None = None
    if response.state:
        sequence_number = response.state.get("sequence_number", None)
        num_sequences = response.state.get("num_sequences", None)

    logger.info(f"Received identification response for file: {filename}")

    answer = get_answer(generated_text, "```json", "```")
    if not answer:
        logger.info(f"No JSON answer found in generated text for {filename}")
        return

    try:
        new_speakers = json.loads(answer)
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing speaker JSON: {e}", exc_info=True)
        return

    logger.info(f"Parsed {len(new_speakers)} speaker identifications for {filename}")

    await update_speakers(filename, new_speakers, config)
    logger.info(f"Successfully updated speakers for {filename}")
    if sequence_number is not None and num_sequences is not None and sequence_number == num_sequences - 1:
        # last one — notify TRAC that this file is ready for topic extraction
        async with dial_rabbit_from_config(config) as connection:
            async with await connection.channel() as channel:
                await channel.declare_queue(config["destination_queue"], durable=True)
                await channel.default_exchange.publish(
                    aio_pika.Message(filename.encode("utf-8")),
                    routing_key=config["destination_queue"],
                )


async def main(config):
    """
    Main function to start the RabbitMQ consumer with reconnection logic.
    Handles connection failures and automatically reconnects with exponential backoff.
    """
    logger.info("Initializing RabbitMQ name consumer...")

    # Retry configuration
    max_retries = 10
    base_retry_delay = 2  # seconds
    max_retry_delay = 60  # seconds
    
    retry_count = 0

    while True:
        try:
            logger.info(f"Connecting to RabbitMQ at {config['host']}:{config['port']}...")
            connection = await dial_rabbit_from_config(config)
            redis_client = await dial_redis_from_config(config)
            
            # Reset retry count on successful connection
            retry_count = 0

            async with connection:
                async with await connection.channel() as channel:
                    work_queue = await channel.declare_queue(config["work_queue"], durable=True)

                logger.info(f"Successfully connected! Listening for jobs on queue: {work_queue.name}")
                logger.info("Waiting for messages. To exit press CTRL+C")

                # Start consuming messages
                async with CachedMessageIterator(
                        rabbitmq_connection=connection,
                        redis_client=redis_client,
                        queue_name=work_queue.name,
                        redis_key_prefix="backup:name",
                        config=config,
                ) as queue_iter:
                    async for message in queue_iter:
                        async with queue_iter.processing(message):
                            await process_message(message, config)
                            
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
    parser = argparse.ArgumentParser(
        description='Name consumer - reads LLM identification responses from RabbitMQ and updates speaker names in Solr',
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
