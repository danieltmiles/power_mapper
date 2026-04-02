# SCRIBE - Summary Construction and Report Integration from Brief Extracts
# Consumes LLM responses produced by trac_consumer.py and extracts structured
# issue summaries (issue_title + description) from the generated JSON, then
# vectorizes and persists them to the Solr 'topics' core.

import asyncio
import argparse
import json

import httpx
from aiormq import AMQPError, ChannelInvalidStateError, ChannelClosed
from httpx import ConnectError

import serialization
from logger import get_logger
from utils import load_config, dial_rabbit_from_config, get_answer, EmbeddingModel, ensure_solr_vector_field_type
from wire_formats import LLMPromptResponse

logger = get_logger("scribe")

SOLR_CORE = "topics"


def extract_issue_summary(generated_text: str) -> dict | None:
    """
    Extract issue_title and description from the LLM's generated text.

    Expects the LLM to have responded with a ```json block containing at
    minimum 'issue_title' and 'description' keys, as requested by trac_consumer.py.

    Returns a dict with those keys, or None if extraction fails.
    """
    raw_json = get_answer(generated_text, "```json", "```")
    if not raw_json:
        logger.info("no ```json block found in generated text")
        return None
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as json_error:
        logger.info(f"failed to parse extracted JSON: {json_error}")
        logger.info(f"Raw extracted text: {raw_json!r}")
        return None

    issue_title = parsed.get("issue_title")
    description = parsed.get("description")

    if not issue_title:
        logger.info("issue_title missing from extracted JSON")
        return None

    return {"issue_title": issue_title, "description": description}


async def ensure_solr_core(solr_config: dict) -> None:
    """
    Create the 'topics' core if it does not already exist, then ensure the
    schema contains all required fields including the dense vector field.
    """
    base_url = solr_config["url"].rstrip("/")
    auth = httpx.BasicAuth(solr_config["username"], solr_config["password"])

    async with httpx.AsyncClient(auth=auth, timeout=30) as client:
        schema_url = f"{base_url}/{SOLR_CORE}/schema"

        vector_field_type_name = await ensure_solr_vector_field_type(client, schema_url, EmbeddingModel().dimension)

        # Ensure document fields exist
        fields_to_add = [
            {"name": "title",           "type": "text_general",        "stored": True, "indexed": True,  "multiValued": False},
            {"name": "description",     "type": "text_general",        "stored": True, "indexed": True,  "multiValued": False},
            {"name": "source_filename", "type": "string",              "stored": True, "indexed": True,  "multiValued": False},
            {"name": "vector_qwen",     "type": vector_field_type_name, "stored": True, "indexed": True, "multiValued": False},
        ]

        existing_fields_response = await client.get(f"{schema_url}/fields")
        existing_fields_response.raise_for_status()
        existing_field_names = {field["name"] for field in existing_fields_response.json().get("fields", [])}

        fields_added = 0
        for field in fields_to_add:
            if field["name"] not in existing_field_names:
                add_response = await client.post(schema_url, json={"add-field": field})
                add_response.raise_for_status()
                logger.info(f"Added field: {field['name']} ({field['type']})")
                fields_added += 1
            else:
                logger.info(f"Field already exists: {field['name']}")

        if fields_added > 0:
            logger.info(f"Successfully added {fields_added} new fields to schema")
        else:
            logger.info("Schema is up to date - all required fields exist")


async def publish_topic_to_solr(
    issue_summary: dict,
    source_filename: str,
    solr_config: dict,
) -> None:
    text = issue_summary["description"]
    vector = EmbeddingModel().encode(text)

    document = {
        "title": issue_summary["issue_title"],
        "description": issue_summary["description"],
        "source_filename": source_filename,
        "vector_qwen": vector,
    }

    base_url = solr_config["url"].rstrip("/")
    auth = httpx.BasicAuth(solr_config["username"], solr_config["password"])
    update_url = f"{base_url}/{SOLR_CORE}/update"

    async with httpx.AsyncClient(auth=auth, timeout=90) as client:
        tries = 0
        while True:
            try:
                response = await client.post(update_url, json=[document], params={"commit": "true"})
                break
            except ConnectError:
                tries += 1
                if tries > 5:
                    logger.error("Too many errors talking to Solr, giving up")
                    raise
                logger.error(f"ConnectError posting to Solr, retrying in {5 * tries}s...")
                await asyncio.sleep(5 * tries)

        response.raise_for_status()
        logger.info(f"Published topic '{issue_summary['issue_title']}' to Solr")


async def process_message(message, config: dict) -> None:
    try:
        prompt_response: LLMPromptResponse = serialization.load(message.body.decode())
        issue_summary = extract_issue_summary(prompt_response.generated_text)
        if issue_summary:
            await publish_topic_to_solr(issue_summary, prompt_response.filename, config["solr"])
        else:
            logger.info("Could not extract issue summary from response")

    except Exception as error:
        logger.error(f"Error processing message: {error}", exc_info=True)
    finally:
        await message.ack()


async def main(config: dict) -> None:
    logger.info("Initializing SCRIBE service...")

    logger.info("Loading embedding model...")
    EmbeddingModel().init(config["embedding_model_path"])
    logger.info("Embedding model loaded")

    await ensure_solr_core(config["solr"])

    max_retries = 10
    base_retry_delay = 2   # seconds
    max_retry_delay = 60   # seconds
    retry_count = 0

    while True:
        try:
            logger.info(f"Connecting to RabbitMQ at {config['host']}:{config['port']}...")
            connection = await dial_rabbit_from_config(config)
            retry_count = 0

            async with connection:
                async with await connection.channel() as channel:
                    await channel.set_qos(prefetch_count=1)
                    queue = await channel.declare_queue(config["work_queue"], durable=True)

                    logger.info(f"Listening for issue summaries on queue: {queue.name}")
                    logger.info("Waiting for messages. To exit press CTRL+C")

                    async with queue.iterator() as queue_iter:
                        async for message in queue_iter:
                            await process_message(message, config)

        except (AMQPError, ChannelInvalidStateError, ChannelClosed, ConnectionError) as connection_error:
            retry_count += 1
            if retry_count > max_retries:
                logger.error(f"Max retries ({max_retries}) exceeded. Giving up.")
                raise

            delay = min(base_retry_delay * (2 ** (retry_count - 1)), max_retry_delay)
            logger.error(f"Connection error: {connection_error}")
            logger.info(f"Reconnection attempt {retry_count}/{max_retries} in {delay} seconds...")
            await asyncio.sleep(delay)

        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
            break

        except Exception as error:
            logger.error(f"Unexpected error in main loop: {error}", exc_info=True)

            retry_count += 1
            if retry_count > max_retries:
                logger.error(f"Max retries ({max_retries}) exceeded. Giving up.")
                raise

            delay = min(base_retry_delay * (2 ** (retry_count - 1)), max_retry_delay)
            logger.info(f"Retrying in {delay} seconds...")
            await asyncio.sleep(delay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SCRIBE - consumes LLM issue-summary responses from trac_consumer.py",
    )
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the JSON configuration file",
        default="scribe_config.json",
        nargs="?",
    )

    args = parser.parse_args()
    config = load_config(args.config_file)

    logger.info(f"Loaded configuration from: {args.config_file}")
    logger.info(f"Work queue: {config['work_queue']}")
    logger.info(f"RabbitMQ host: {config['host']}:{config['port']}")

    try:
        asyncio.run(main(config))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
