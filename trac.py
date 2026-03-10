# TRAC - Topic Recognition and Classification
import asyncio
import uuid
import argparse
from functools import partial

import aio_pika
import httpx
from aiormq import ChannelInvalidStateError, ChannelClosed, AMQPError

import serialization
from sliding_window import SlidingWindow, SequencedText
from utils import load_config, dial_rabbit_from_config
from wire_formats import LLMPromptJob

MAX_PROMPT_SIZE = 10000
PAGE_SIZE = 100

class SolrDocumentWrapper(SequencedText):
    """Adapts a Solr transcript document to the SequencedText protocol."""
    def __init__(self, doc: dict):
        self.speaker = doc.get("speaker_name", "UNKNOWN")
        self.start_seconds = float(doc.get("start_time", 0.0))
        self.end_seconds = float(doc.get("end_time", 0.0))
        self.text = doc.get("text", "")
        self.sequence_number = int(doc.get("sequence_number", 0))
        self.sequence_count = int(doc.get("count", 0))


async def fetch_solr_documents_for_filename(filename: str, solr_config: dict):
    """Async generator that yields Solr transcript docs for a filename, in sequence order."""
    base_url = solr_config['url'].rstrip('/')
    select_url = f"{base_url}/transcripts/select"
    auth = httpx.BasicAuth(solr_config['username'], solr_config['password'])
    start = 0

    async with httpx.AsyncClient(auth=auth, timeout=30) as client:
        while True:
            response = await client.get(select_url, params={
                "q": "*:*",
                "fq": f"{{!term f=filename}}{filename}",
                "sort": "sequence_number asc",
                "rows": PAGE_SIZE,
                "start": start,
                "wt": "json",
            })
            response.raise_for_status()
            data = response.json()
            docs = data["response"]["docs"]
            num_found = data["response"]["numFound"]

            for doc in docs:
                yield doc

            start += len(docs)
            if start >= num_found or not docs:
                break


async def main(config):
    max_retries = 10
    base_retry_delay = 2  # seconds
    max_retry_delay = 60  # seconds
    retry_count = 0

    while True:
        try:
            llm_connection = await dial_rabbit_from_config(config)
            llm_channel = await llm_connection.channel()
            await llm_channel.declare_queue("llm/qwen32", durable=True)

            async def sliding_window_callback(filename: str, window: str):
                nonlocal llm_connection, llm_channel
                window = window.strip()
                conversation = [
                    {
                        "role": "system",
                        "content": f"You are a political analyst helping to extract information from city council meeting transcripts. Here a portion of a meeting transcript:\n<transcript>\n{window}\n</transcript>",
                    },
                    {
                        "role": "user",
                        "content": """Extract all political issues as relationships in this exact format:
```graph
| Speaker -> Supports/Opposes -> Issue |
```

Rules:
- One relationship per line
- No additional explanation
""",
                    }
                ]
                job = LLMPromptJob(
                    job_id=str(uuid.uuid4()),
                    filename=filename,
                    reply_to=config["reply_to"],
                    prompt=conversation,
                    state={"transcript_section": window},
                )
                try:
                    await llm_channel.default_exchange.publish(
                        aio_pika.Message(
                            body=serialization.dumps(job).encode(),
                            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                        ),
                        routing_key="llm/qwen32",
                    )
                except Exception:
                    llm_connection = await dial_rabbit_from_config(config)
                    llm_channel = await llm_connection.channel()
                    await llm_channel.declare_queue("llm/qwen32", durable=True)
                    await llm_channel.default_exchange.publish(
                        aio_pika.Message(
                            body=serialization.dumps(job).encode(),
                            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                        ),
                        routing_key="llm/qwen32",
                    )

            work_connection = await dial_rabbit_from_config(config)
            async with work_connection:
                async with await work_connection.channel() as work_channel:
                    work_queue = await work_channel.declare_queue(config["work_queue"], durable=True)

                    retry_count = 0
                    print(f"Listening for filenames on queue: {config['work_queue']}")
                    async with work_queue.iterator() as queue_iter:
                        async for message in queue_iter:
                            async with message.process():
                                filename = message.body.decode()
                                print(f"Processing filename: {filename}")
                                sliding_window = SlidingWindow(
                                    max_size=MAX_PROMPT_SIZE,
                                    callback=partial(sliding_window_callback, filename),
                                )
                                async for doc in fetch_solr_documents_for_filename(filename, config["solr"]):
                                    wrapper = SolrDocumentWrapper(doc)
                                    await sliding_window.append(wrapper)

        except (AMQPError, ChannelInvalidStateError, ChannelClosed, ConnectionError) as conn_error:
            retry_count += 1
            if retry_count > max_retries:
                print(f"Max retries ({max_retries}) exceeded. Giving up.")
                raise
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
        description='TRAC - Topic Recognition and Classification. Consumes filenames from RabbitMQ and sends transcript windows to an LLM.',
    )
    parser.add_argument(
        'config_file',
        type=str,
        help='Path to the JSON configuration file',
        default='trac_config.json',
        nargs='?',
    )

    args = parser.parse_args()
    config = load_config(args.config_file)

    print(f"Loaded configuration from: {args.config_file}")
    print(f"Work queue: {config['work_queue']}")
    print(f"RabbitMQ host: {config['host']}:{config['port']}")

    try:
        asyncio.run(main(config))
    except KeyboardInterrupt:
        print("\nInterrupted by user")
