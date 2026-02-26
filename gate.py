import time
from typing import Any

import aio_pika
import argparse
import httpx
import asyncio

from aio_pika.abc import AbstractIncomingMessage, AbstractRobustChannel, AbstractRobustExchange
from httpx import ConnectError

import serialization
from utils import load_config, create_ssl_context, get_answer, SimilarityCalculator
from wire_formats import CleanedWhisperResult

async def publish_speaker_segment_to_solr(cleaned_result: CleanedWhisperResult, solr_config: dict) -> None:
    auth = httpx.BasicAuth(solr_config['username'], solr_config['password'])
    document = {
        "filename": cleaned_result.whisper_result.transcript_metadata.filename,
        "meeting_title": cleaned_result.whisper_result.transcript_metadata.meeting_title,
        "session_type": cleaned_result.whisper_result.transcript_metadata.session_type,
        "date": cleaned_result.whisper_result.transcript_metadata.date.isoformat(),
        "video_id": cleaned_result.whisper_result.transcript_metadata.video_id,
        "sequence_number": cleaned_result.whisper_result.segment_count,
        "count": cleaned_result.whisper_result.total_segments,
        "start_time": cleaned_result.whisper_result.timings.start,
        "end_time": cleaned_result.whisper_result.timings.end,
        "speaker_name": cleaned_result.whisper_result.speaker,
        "text": cleaned_result.cleaned_transcript,
        "cleaned_whisper_result": serialization.dumps(cleaned_result, minify=True),
    }
    base_url = solr_config['url'].rstrip('/')
    collection = "transcripts"
    update_url = f"{base_url}/{collection}/update"
    async with httpx.AsyncClient(auth=auth, timeout=90) as client:
        tries = 0
        while True:
            try:
                response = await client.post(
                    update_url,
                    json=[document],
                    params={"commit": "true"},
                )
                break
            except ConnectError:
                tries += 1
                if tries > 5:
                    print("too many errors talking to solr, bailing")
                    raise
                print("caught ConnectError trying to send document to solr, sleeping 5, then retrying")
                await asyncio.sleep(5 * tries)

        response.raise_for_status()
        result = response.json()
        print(f"✓ Document published successfully!")
        print(f"Response: {result}")


async def ensure_solr_schema(solr_config: dict) -> None:
    """Ensure the Solr collection has the proper schema for transcript documents."""

    # Construct schema API URL
    base_url = solr_config['url'].rstrip('/')
    auth = httpx.BasicAuth(solr_config['username'], solr_config['password'])
    collection = "transcripts"
    schema_url = f"{base_url}/{collection}/schema"
    # Define the fields we need
    fields_to_add = [
        {"name": "filename", "type": "string", "stored": True, "indexed": True, "multiValued": False},
        {"name": "meeting_title", "type": "text_general", "stored": True, "indexed": True, "multiValued": False},
        {"name": "session_type", "type": "string", "stored": True, "indexed": True, "multiValued": False},
        {"name": "date", "type": "pdate", "stored": True, "indexed": True, "multiValued": False},
        {"name": "video_id", "type": "string", "stored": True, "indexed": True, "multiValued": False},
        {"name": "sequence_number", "type": "pint", "stored": True, "indexed": True, "multiValued": False},
        {"name": "count", "type": "pint", "stored": True, "indexed": True, "multiValued": False},
        {"name": "start_time", "type": "pfloat", "stored": True, "indexed": True, "multiValued": False},
        {"name": "end_time", "type": "pfloat", "stored": True, "indexed": True, "multiValued": False},
        {"name": "speaker_name", "type": "text_general", "stored": True, "indexed": True, "multiValued": False},
        {"name": "speaker_confidence", "type": "pint", "stored": True, "indexed": True, "multiValued": False},
        {"name": "text", "type": "text_general", "stored": True, "indexed": True, "multiValued": False},
        {"name": "cleaned_whisper_result", "type": "text_general", "stored": True, "indexed": True, "multiValued": False},
    ]

    async with httpx.AsyncClient(auth=auth, timeout=10) as client:
        # Get existing schema fields
        try:
            response = await client.get(f"{schema_url}/fields")
            response.raise_for_status()
            existing_fields = {field['name'] for field in response.json().get('fields', [])}
            print(f"Found {len(existing_fields)} existing fields in schema")
        except Exception as e:
            print(f"Warning: Could not retrieve existing schema: {e}")
            existing_fields = set()

        # Add missing fields
        fields_added = 0
        for field in fields_to_add:
            if field['name'] not in existing_fields:
                try:
                    response = await client.post(
                        f"{schema_url}",
                        json={"add-field": field},
                    )
                    response.raise_for_status()
                    print(f"Added field: {field['name']} ({field['type']})")
                    fields_added += 1
                except Exception as e:
                    print(f"Warning: Could not add field {field['name']}: {e}")
            else:
                print(f"Field already exists: {field['name']}")

        if fields_added > 0:
            print(f"Successfully added {fields_added} new fields to schema")
        else:
            print("Schema is up to date - all required fields exist")


async def process_message(
    message: AbstractIncomingMessage,
    channel: AbstractRobustChannel,
    config: dict[str, Any],
    accepted_exchange: AbstractRobustExchange,
):
    # Parse the incoming message
    cleaned_whisper_result: CleanedWhisperResult = serialization.load(message.body.decode())
    original_text = cleaned_whisper_result.whisper_result.transcript.get("text", "")
    cleaned_text = cleaned_whisper_result.cleaned_transcript
    start = time.time()
    similarity_score = SimilarityCalculator().text_similarity(original_text, cleaned_text)
    end = time.time()
    print(f"calculated similarity score in {end-start} seconds")
    if similarity_score <= 0.7:
        print("rejected")
        cleaned_whisper_result.whisper_result.tries += 1
        if cleaned_whisper_result.whisper_result.tries < config.get("max_retries", 3):
            await channel.default_exchange.publish(
                aio_pika.Message(
                    body=serialization.dumps(cleaned_whisper_result.whisper_result).encode(),
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                ),
                routing_key=config["retry_queue"],
            )
    else:
        print("accepted")
        await asyncio.gather(
            accepted_exchange.publish(message, routing_key=config["accepted_queue"]),
            publish_speaker_segment_to_solr(cleaned_whisper_result, config["solr"])
        )
    await message.ack()


async def main(config):
    """
    Main function to start the RabbitMQ consumer with reconnection logic.
    Handles connection failures and automatically reconnects with exponential backoff.
    """
    print("Initializing RabbitMQ consumer...")
    
    ssl_context = create_ssl_context()
    
    print(f"Connecting to RabbitMQ at {config['host']}:{config['port']}...")

    connection = await aio_pika.connect_robust(
        host=config['host'],
        port=config['port'],
        login=config['username'],
        password=config['password'],
        ssl=True,
        ssl_context=ssl_context,
    )
    await ensure_solr_schema(config["solr"])

    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)

        work_queue, alt_exchange, accepted_exchange, unrouted_queue, retry_queue, accepted_queue = await asyncio.gather(
            channel.declare_queue(config["work_queue"], durable=True),
            channel.declare_exchange(
                f"{config['accepted_queue']}.unrouted",
                aio_pika.ExchangeType.FANOUT,
                durable=True
            ),
            channel.declare_exchange(
                config["accepted_queue"],
                aio_pika.ExchangeType.TOPIC,
                durable=True,
                arguments={"alternate-exchange": f"{config['accepted_queue']}.unrouted"}
            ),
            channel.declare_queue("unrouted_messages", durable=True),
            channel.declare_queue(config["retry_queue"], durable=True),
            channel.declare_queue(config["accepted_queue"], durable=True,)
        )
        await unrouted_queue.bind(alt_exchange)
        await accepted_queue.bind(accepted_exchange, routing_key=config["accepted_queue"])

        print(f"Successfully connected! Listening for jobs on queue: {work_queue}")
        print("Waiting for messages. To exit press CTRL+C")

        async with work_queue.iterator() as queue_iter:
            async for message in queue_iter:
                await process_message(message, channel, config, accepted_exchange)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generic RabbitMQ consumer boilerplate - connects to queue and processes messages',
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
