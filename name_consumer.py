import json

import aio_pika
import argparse
import asyncio
import httpx

from aio_pika.abc import AbstractIncomingMessage, AbstractExchange
from aiormq import ChannelInvalidStateError, ChannelClosed, AMQPError

import serialization
import wire_formats
from cached_iterator import CachedMessageIterator
from utils import load_config, create_ssl_context, get_answer, dial_rabbit_from_config, dial_redis_from_config


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
                    print(f"updated {len(docs_to_update)} speaker sections from {filename} with:\n{json.dumps(new_speakers, indent=4)}")

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

    print(f"Received identification response for file: {filename}")

    answer = get_answer(generated_text, "```json", "```")
    if not answer:
        print(f"No JSON answer found in generated text for {filename}")
        return

    try:
        new_speakers = json.loads(answer)
    except json.JSONDecodeError as e:
        print(f"Error parsing speaker JSON: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"Parsed {len(new_speakers)} speaker identifications for {filename}")

    await update_speakers(filename, new_speakers, config)
    print(f"Successfully updated speakers for {filename}")
    if sequence_number is not None and num_sequences is not None and sequence_number == num_sequences - 1:
        # last one
        async with dial_rabbit_from_config(config) as connection:
            async with await connection.channel() as channel:
                exchange: AbstractExchange = await channel.get_exchange(config["file_done_exchange"])
                await exchange.publish(
                    aio_pika.Message(filename.encode("utf-8")),
                    routing_key=config["file_done_exchange"],
                )


async def main(config):
    """
    Main function to start the RabbitMQ consumer with reconnection logic.
    Handles connection failures and automatically reconnects with exponential backoff.
    """
    print("Initializing RabbitMQ name consumer...")

    # Retry configuration
    max_retries = 10
    base_retry_delay = 2  # seconds
    max_retry_delay = 60  # seconds
    
    retry_count = 0

    while True:
        try:
            print(f"Connecting to RabbitMQ at {config['host']}:{config['port']}...")
            connection = await dial_rabbit_from_config(config)
            redis_client = await dial_redis_from_config(config)
            
            # Reset retry count on successful connection
            retry_count = 0

            async with connection:
                async with await connection.channel() as channel:
                    alt_exchange, topic_exchange, unrouted_queue, work_queue = await asyncio.gather(
                        channel.declare_exchange(
                            name=f"{config['file_done_exchange']}.unrouted",
                            type=aio_pika.ExchangeType.FANOUT,
                            durable=True,
                        ),
                        channel.declare_exchange(
                            name=config["file_done_exchange"],
                            type=aio_pika.ExchangeType.TOPIC,
                            durable=True,
                            arguments = {"alternate-exchange": f"{config['file_done_exchange']}.unrouted"},
                        ),
                        channel.declare_queue("file_done_unrouted_messages", durable=True),
                        channel.declare_queue(config["work_queue"], durable=True),
                    )
                    await unrouted_queue.bind(alt_exchange)

                print(f"Successfully connected! Listening for jobs on queue: {work_queue.name}")
                print("Waiting for messages. To exit press CTRL+C")

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

    print(f"Loaded configuration from: {args.config_file}")
    print(f"Work queue: {config['work_queue']}")
    print(f"RabbitMQ host: {config['host']}:{config['port']}")
    print(f"Username: {config['username']}")

    try:
        asyncio.run(main(config))
    except KeyboardInterrupt:
        print("\nInterrupted by user")
