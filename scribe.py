# SCRIBE - Summary Construction and Report Integration from Brief Extracts
import asyncio
import argparse
import aio_pika
from aiormq import ChannelInvalidStateError, ChannelClosed, AMQPError

import serialization
from logger import get_logger
from relationship_parser import RelationshipParser
from utils import load_config, dial_rabbit_from_config, get_answer
from wire_formats import LLMPromptResponse, LLMPromptJob

logger = get_logger("trac_consumer")


async def process_message(message, llm_channel, config):
    try:
        prompt_response: LLMPromptResponse = serialization.load(message.body.decode())
        answer = get_answer(prompt_response.generated_text, "```graph", "```")
        relationships = RelationshipParser().parse_multiple(answer)
        issues = list(set(x.issue for x in relationships))
        for issue in issues:
            new_prompt_job = LLMPromptJob(
                job_id=prompt_response.job_id,
                filename=prompt_response.filename,
                reply_to=config["reply_to"],
                prompt=[
                    prompt_response.prompt[0],
                    {
                        "role": "user",
                        "content": (
                            f"Another analyst identified an issue, `{issue}` from the given transcript section. "
                            "Using the provided transcript section please create as complete a description of the "
                            "issue as you can. If you cannot infer any additional details, simply use the given issue "
                            "text as the full description. Format your your response like so:\n```json\n{"
                            "\n    \"issue_title\": \"given issue title\",\n    \"description\": \"full description "
                            "posibly with multiple lines\"\n}\n```\n"
                        )
                    }
                ],
                request_id=prompt_response.request_id,
                state=prompt_response.state,
            )
            await llm_channel.default_exchange.publish(
                aio_pika.Message(
                    body=serialization.dumps(new_prompt_job).encode(),
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                ),
                routing_key="llm/qwen32",
            )

    except Exception as e:
        logger.error(f"Error processing message: {e}")
    finally:
        await message.ack()


async def main(config):
    """
    Main function to start the RabbitMQ consumer with reconnection logic.
    
    Handles connection failures and automatically reconnects with exponential backoff.
    """
    print("Initializing TRAC consumer...")
    
    # Retry configuration
    max_retries = 10
    base_retry_delay = 2  # seconds
    max_retry_delay = 60  # seconds
    
    retry_count = 0

    while True:
        try:
            logger.info(f"Connecting to RabbitMQ at {config['host']}:{config['port']}...")

            retry_count = 0

            llm_connection = await dial_rabbit_from_config(config)
            llm_channel = await llm_connection.channel()
            await llm_channel.declare_queue("llm/qwen32", durable=True)

            connection = await dial_rabbit_from_config(config)

            async with connection:
                async with await connection.channel() as channel:
                    await channel.set_qos(prefetch_count=1)
                    queue = await channel.declare_queue(config["work_queue"], durable=True)

                    logger.info(f"Successfully connected! Listening for jobs on queue: {queue.name}")
                    logger.info("Waiting for messages. To exit press CTRL+C")

                    async with queue.iterator() as queue_iter:
                        async for message in queue_iter:
                            await process_message(message, llm_channel, config)
                        
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
        description='TRAC Consumer - reads from a queue using JSON config',
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

    logger.info(f"Loaded configuration from: {args.config_file}")
    logger.info(f"Work queue: {config['work_queue']}")
    logger.info(f"RabbitMQ host: {config['host']}:{config['port']}")

    try:
        asyncio.run(main(config))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")