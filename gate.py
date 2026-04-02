import datetime
import time
from typing import Any

import aio_pika
import argparse
import httpx
import asyncio

from aio_pika.abc import AbstractIncomingMessage
from httpx import ConnectError

import serialization
from cached_iterator import CachedMessageIterator
from logger import get_logger
from utils import load_config, SimilarityCalculator, dial_rabbit_from_config, dial_redis_from_config, publish_event, EmbeddingModel, ensure_solr_vector_field_type
from wire_formats import CleanedWhisperResult

logger = get_logger("gate")

async def publish_speaker_segment_to_solr(cleaned_result: CleanedWhisperResult, solr_config: dict) -> None:
    auth = httpx.BasicAuth(solr_config['username'], solr_config['password'])
    transcript_filename_date = cleaned_result.whisper_result.transcript_metadata.date
    if transcript_filename_date:
        date = transcript_filename_date.isoformat()
    else:
        date = "2006-01-02T15:04:05+00:00"
    vector = EmbeddingModel().encode(cleaned_result.cleaned_transcript)
    document = {
        "filename": cleaned_result.whisper_result.transcript_metadata.filename,
        "meeting_title": cleaned_result.whisper_result.transcript_metadata.meeting_title,
        "session_type": cleaned_result.whisper_result.transcript_metadata.session_type,
        "date": date,
        "video_id": cleaned_result.whisper_result.transcript_metadata.video_id,
        "sequence_number": cleaned_result.whisper_result.segment_count,
        "count": cleaned_result.whisper_result.total_segments,
        "start_time": cleaned_result.whisper_result.timings.start,
        "end_time": cleaned_result.whisper_result.timings.end,
        "speaker_name": cleaned_result.whisper_result.speaker,
        "text": cleaned_result.cleaned_transcript,
        "cleaned_whisper_result": serialization.dumps(cleaned_result, minify=True),
        "vector_qwen": vector,
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
                    logger.error("too many errors talking to solr, bailing")
                    raise
                logger.error("caught ConnectError trying to send document to solr, sleeping 5, then retrying")
                await asyncio.sleep(5 * tries)

        response.raise_for_status()
        result = response.json()
        logger.info("Document published successfully!")
        logger.info(f"Response: {result}")


async def ensure_solr_schema(solr_config: dict) -> None:
    """Ensure the Solr collection has the proper schema for transcript documents."""

    base_url = solr_config['url'].rstrip('/')
    auth = httpx.BasicAuth(solr_config['username'], solr_config['password'])
    collection = "transcripts"
    schema_url = f"{base_url}/{collection}/schema"

    async with httpx.AsyncClient(auth=auth, timeout=10) as client:
        vector_field_type_name = await ensure_solr_vector_field_type(client, schema_url, EmbeddingModel().dimension)

        fields_to_add = [
            {"name": "filename",             "type": "string",                "stored": True, "indexed": True, "multiValued": False},
            {"name": "meeting_title",        "type": "text_general",          "stored": True, "indexed": True, "multiValued": False},
            {"name": "session_type",         "type": "string",                "stored": True, "indexed": True, "multiValued": False},
            {"name": "date",                 "type": "pdate",                 "stored": True, "indexed": True, "multiValued": False},
            {"name": "video_id",             "type": "string",                "stored": True, "indexed": True, "multiValued": False},
            {"name": "sequence_number",      "type": "pint",                  "stored": True, "indexed": True, "multiValued": False},
            {"name": "count",                "type": "pint",                  "stored": True, "indexed": True, "multiValued": False},
            {"name": "start_time",           "type": "pfloat",                "stored": True, "indexed": True, "multiValued": False},
            {"name": "end_time",             "type": "pfloat",                "stored": True, "indexed": True, "multiValued": False},
            {"name": "speaker_name",         "type": "text_general",          "stored": True, "indexed": True, "multiValued": False},
            {"name": "speaker_confidence",   "type": "pint",                  "stored": True, "indexed": True, "multiValued": False},
            {"name": "text",                 "type": "text_general",          "stored": True, "indexed": True, "multiValued": False},
            {"name": "cleaned_whisper_result","type": "text_general",         "stored": True, "indexed": True, "multiValued": False},
            {"name": "vector_qwen",          "type": vector_field_type_name,  "stored": True, "indexed": True, "multiValued": False},
        ]

        # Get existing schema fields
        try:
            response = await client.get(f"{schema_url}/fields")
            response.raise_for_status()
            existing_fields = {field['name'] for field in response.json().get('fields', [])}
            logger.info(f"Found {len(existing_fields)} existing fields in schema")
        except Exception as e:
            logger.info(f"Could not retrieve existing schema: {e}")
            existing_fields = set()

        # Add missing fields
        fields_added = 0
        for field in fields_to_add:
            if field['name'] not in existing_fields:
                try:
                    response = await client.post(f"{schema_url}", json={"add-field": field})
                    response.raise_for_status()
                    logger.info(f"Added field: {field['name']} ({field['type']})")
                    fields_added += 1
                except Exception as e:
                    logger.info(f"Could not add field {field['name']}: {e}")
            else:
                logger.info(f"Field already exists: {field['name']}")

        if fields_added > 0:
            logger.info(f"Successfully added {fields_added} new fields to schema")
        else:
            logger.info("Schema is up to date - all required fields exist")


async def process_message(
    message: AbstractIncomingMessage,
    config: dict[str, Any],
):
    # Parse the incoming message
    bodystr = message.body.decode()
    logger.info(f"Received message with body: {bodystr}")
    cleaned_whisper_result: CleanedWhisperResult = serialization.load(bodystr)
    logger.info(f"Parsed CleanedWhisperResult: {cleaned_whisper_result}")
    original_text = cleaned_whisper_result.whisper_result.transcript.get("text", "")
    cleaned_text = cleaned_whisper_result.cleaned_transcript
    start = time.time()
    similarity_score = SimilarityCalculator().text_similarity(original_text, cleaned_text)
    end = time.time()
    logger.info(f"calculated similarity score in {end-start} seconds")
    
    filename = cleaned_whisper_result.whisper_result.transcript_metadata.filename
    sequence_number = cleaned_whisper_result.whisper_result.segment_count
    total_segments = cleaned_whisper_result.whisper_result.total_segments

    if similarity_score <= 0.7:
        logger.info("rejected")
        cleaned_whisper_result.whisper_result.tries += 1
        tries = cleaned_whisper_result.whisper_result.tries
        max_retries = config.get("max_retries", 3)
        if tries < max_retries:
            await publish_event(
                config,
                f"GATE_REJECTED_REQUEUE: {filename} segment {sequence_number}/{total_segments} "
                f"rejected (similarity={similarity_score:.3f}, try {tries}/{max_retries}). "
                f"Requeuing to {config['retry_queue']} for re-cleaning. "
                f"This segment will be re-processed through CLEAN→GATE, adding to total processing count."
            )
            # Dial fresh connection for publishing rejection
            async with await dial_rabbit_from_config(config) as rabbitmq_connection:
                async with await rabbitmq_connection.channel() as channel:
                    await channel.default_exchange.publish(
                        aio_pika.Message(
                            body=serialization.dumps(cleaned_whisper_result.whisper_result).encode(),
                            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                        ),
                        routing_key=config["retry_queue"],
                    )
        else:
            await publish_event(
                config,
                f"GATE_REJECTED_EXHAUSTED: {filename} segment {sequence_number}/{total_segments} "
                f"rejected (similarity={similarity_score:.3f}) and max retries ({max_retries}) exhausted. "
                f"Segment DROPPED - will not appear in final output."
            )
    else:
        logger.info("accepted")
        # Dial fresh connection for publishing acceptance
        async with await dial_rabbit_from_config(config) as rabbitmq_connection:
            async with await rabbitmq_connection.channel() as channel:
                await asyncio.gather(
                    channel.default_exchange.publish(
                        aio_pika.Message(body=message.body),
                        routing_key=config["accepted_queue"]
                    ),
                    publish_speaker_segment_to_solr(cleaned_whisper_result, config["solr"])
                )


async def main(config):
    """
    Main function to start the RabbitMQ consumer with reconnection logic.
    Handles connection failures and automatically reconnects with exponential backoff.
    """
    logger.info("Initializing RabbitMQ consumer...")

    EmbeddingModel().init(config["embedding_model_path"])

    # Retry configuration
    max_retries = 10
    base_retry_delay = 2  # seconds
    max_retry_delay = 60  # seconds

    await ensure_solr_schema(config["solr"])
    
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
                    await asyncio.gather(
                        channel.declare_queue(config["work_queue"], durable=True),
                        channel.declare_queue(config["retry_queue"], durable=True),
                        channel.declare_queue(config["accepted_queue"], durable=True,)
                    )

                logger.info(f"Successfully connected! Listening for jobs on queue: {config['work_queue']}")
                logger.info("Waiting for messages. To exit press CTRL+C")

                # Start consuming messages
                async with CachedMessageIterator(
                        rabbitmq_connection=connection,
                        redis_client=redis_client,
                        queue_name=config["work_queue"],
                        redis_key_prefix="backup:gate",
                        config=config,
                ) as queue_iter:
                    async for message in queue_iter:
                        async with queue_iter.processing(message):
                            await process_message(message, config)

        except (Exception,) as conn_error:
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

    logger.info(f"Loaded configuration from: {args.config_file}")
    logger.info(f"Work queue: {config['work_queue']}")
    logger.info(f"RabbitMQ host: {config['host']}:{config['port']}")
    logger.info(f"Username: {config['username']}")

    try:
        asyncio.run(main(config))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
