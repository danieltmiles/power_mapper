"""
Speaker Identification producer for RabbitMQ
"""
import uuid
from asyncio import Semaphore
from collections import defaultdict
from typing import Any

import aio_pika
import argparse
import asyncio
import signal
import redis.asyncio as redis

from aio_pika.abc import AbstractRobustConnection, AbstractRobustChannel

import serialization
from logger import get_logger
from sliding_window import SlidingWindow
from utils import load_config, create_ssl_context, publish_event
from wire_formats import CleanedWhisperResult, LLMPromptJob, Metaparams

logger = get_logger("name_producer")

MAX_PROMPT_TRANSCRIPT_LENGTH = 10000
SYSTEM_PROMPT = (
    "You are a transcript analyst who identifies anonymous speakers in public meetings. "
    "You reason step-by-step from conversational evidence and always respond with valid JSON "
    "— never with explanations, apologies, or filler text outside the JSON block."
)
USER_PROMPT_TEMPLATE = """The following is an excerpt from an automatically transcribed meeting:
<transcript>
{transcript}
</transcript>

The transcription software did not know who was speaking and assigned names like SPEAKER_1, SPEAKER_2, etc. Please infer speaker identities from context. The diarization was also flawed, meaning sometimes something at the start or end of a speaker's turn might actually have been from the speaker before or after.
APPROACH:
Reason through the clues step-by-step and look for patterns.

CRITICAL RULES:
1. Use full names when possible (e.g., "Joe Smith", "Mayor Jane Doe")
2. Use only the information in the transcript provided, do not use your own knowledge
3. NEVER output bare titles like "Moderator", "Chair", "Professor" - always include the person's name
4. Output ONLY valid JSON, no markdown, no explanation
5. Results of Unknown should have a confidence score of 1

KEY CLUES TO ANALYZE:
- Direct Address: If Speaker_1 uses a person's name, like, "as my colleague John Smith will tell us," and Speaker_2 responds, Speaker_2 is likely John Smith.
- Introductions: If Speaker A introduces Speaker B, and Speaker B says, "Thank you, John Smith," speaker A is likely John Smith.
- Titles in address: "Mayor Jones" or "Councilmember Garcia" spoken TO someone identifies that person
- Self-introductions: "My name is..." statements
- Role references: Opening/chairing meetings suggests leadership role
- Chronological patterns: Who speaks immediately after being addressed?

Confidence scale:
10 = Direct self-identification ("my name is", or, "I am", etc.)
8-9 = Addressed by name and responds immediately after or strong contextual evidence (multiple clues align)
4-7 = Moderate clues (role + context)
1-3 = Weak inference

If you are unable to identify a speaker, leave them out of the output. Return ONLY the speaker identities without explanations, stop when finished. End with a text block containing the corrected identities, formatted like this:
JSON_OUTPUT_START
```json
{{
  "Speaker_01": {{
    "name": "Full Name",
    "confidence": 1-10
  }},
  "Speaker_02": {{
    "name": "Full Name",
    "confidence": 1-10
  }}
}}
```
JSON_OUTPUT_END
"""


class SpeakerIdentificationProducer:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.ssl_context = create_ssl_context()
        self.sliding_windows_by_filename: dict[str, SlidingWindow] = {}
        self.redis_client = None
        self.redis_semaphore = Semaphore(10)

    async def _connect_redis(self):
        """Connect to Redis for message backup."""
        redis_config = self.config.get('redis', {})
        if not redis_config:
            logger.info("No Redis configuration found. Message backup disabled.")
            return
        
        self.redis_client = await redis.Redis(
            host=redis_config['host'],
            port=redis_config['port'],
            ssl=redis_config.get('ssl', False),
            ssl_ca_certs=redis_config.get('ssl_ca_certs'),
            ssl_certfile=redis_config.get('ssl_certfile'),
            ssl_keyfile=redis_config.get('ssl_keyfile'),
        )
        logger.info(f"Connected to Redis at {redis_config['host']}:{redis_config['port']}")

    async def _backup_message_to_redis(self, filename: str, sequence_number: int, message_body: bytes):
        """Backup message to Redis before acknowledging."""
        if self.redis_client is None:
            return

        # Use a Redis key pattern: nameproducerbackup:{filename}:{sequence_number}
        key = f"nameproducerbackup:{filename}:{sequence_number}"
        async with self.redis_semaphore:
            await self.redis_client.set(key, message_body)

    async def _remove_message_from_redis(self, filename: str, sequence_number: int):
        """Remove backed up message from Redis after processing."""
        if self.redis_client is None:
            return
        
        key = f"nameproducerbackup:{filename}:{sequence_number}"
        async with self.redis_semaphore:
            await self.redis_client.delete(key)

    async def _set_recovery_sequence_number(self, filename: str, next_sequence_number: int):
        """Persist the next sequence number to expect on recovery."""
        if self.redis_client is None:
            return
        key = f"nameproducerbackup:{filename}::next_sequence_number"
        async with self.redis_semaphore:
            await self.redis_client.set(key, next_sequence_number)

    async def _get_recovery_sequence_number(self, filename: str) -> int:
        """Return the persisted recovery starting sequence number, or 0 if none."""
        if self.redis_client is None:
            return 0
        key = f"nameproducerbackup:{filename}::next_sequence_number"
        async with self.redis_semaphore:
            value = await self.redis_client.get(key)
        return int(value) if value is not None else 0

    async def _cleanup_filename_backups(self, filename: str):
        """Remove all backup messages for a completed filename."""
        if self.redis_client is None:
            return
        
        # Find all keys matching the pattern
        pattern = f"nameproducerbackup:{filename}:*"
        cursor = 0
        while True:
            async with self.redis_semaphore:
                cursor, keys = await self.redis_client.scan(cursor, match=pattern, count=100)
                if keys:
                    await self.redis_client.delete(*keys)
                if cursor == 0:
                    break
        logger.info(f"Cleaned up Redis backups for {filename}")

    def _make_sliding_window_for(self, filename: str, channel: AbstractRobustChannel) -> SlidingWindow:
        """Create a SlidingWindow whose callback publishes an LLM prompt job for filename."""
        async def callback(prompt_text: str):
            sliding_window = self.sliding_windows_by_filename[filename]
            last_item = sliding_window.window[-1]
            sequence_number = last_item.sequence_number
            total_segments = last_item.sequence_count
            sequence_numbers = [item.sequence_number for item in sliding_window.window]

            last_number = sequence_numbers[0]
            for num in sequence_numbers[1:]:
                if num != last_number + 1:
                    logger.info(f"non-monotonic number, {num} follows {last_number}")
                last_number = num
            logger.info(f"sending prompt with texts from sequence numbers: {sequence_numbers[0]}-{sequence_numbers[-1]}")

            # Only remove items that will be truncated away after this callback.
            # Items kept in the window tail for context in the next window must
            # remain in Redis so they can be recovered after a crash.
            to_lop = int(len(sliding_window.window) * sliding_window.truncation_percentage)
            evicted_sequence_numbers = sequence_numbers[:to_lop]
            remove_coros = [self._remove_message_from_redis(filename, seq) for seq in evicted_sequence_numbers]
            await asyncio.gather(*remove_coros, self.send(channel, filename, sequence_number, total_segments, prompt_text))

            if evicted_sequence_numbers:
                await self._set_recovery_sequence_number(filename, evicted_sequence_numbers[-1] + 1)

            if sequence_number == total_segments - 1:
                await self._cleanup_filename_backups(filename)

        return SlidingWindow(MAX_PROMPT_TRANSCRIPT_LENGTH, callback, truncation_percentage=0.3, filename=filename)

    async def send(
        self,
        channel: AbstractRobustChannel,
        filename: str,
        sequence_number: int,
        total_segments: int,
        transcript_text: str,
    ):
        job_desc = LLMPromptJob(
            job_id=str(uuid.uuid4()),
            filename=filename,
            reply_to=self.config["reply_to"],
            prompt=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(transcript=transcript_text.strip())},
            ],
            meta_params=Metaparams(
                temperature=0.2,
                top_k=40,
                top_p=None,
                repetition_penalty=1.1,
                stop=["JSON_OUTPUT_END"]
            ),
            state={
                "sequence_number": sequence_number,  # sequence number is highest in block
                "num_sequences": total_segments,
            },
        )
        await channel.default_exchange.publish(
            aio_pika.Message(
                body=serialization.dumps(job_desc).encode("utf-8"),
                reply_to=self.config["reply_to"],
            ),
            routing_key=self.config["destination_queue"]
        )

    async def _recover_from_redis(self, channel: AbstractRobustChannel):
        """Recover messages from Redis after a crash or restart."""
        if self.redis_client is None:
            logger.info("No Redis connection - skipping recovery")
            return

        logger.info("Checking Redis for backed up messages...")
        pattern = "nameproducerbackup:*"
        cursor = 0

        all_keys = []
        while True:
            async with self.redis_semaphore:
                cursor, keys = await self.redis_client.scan(cursor, match=pattern, count=100)
                all_keys.extend(keys)
                if cursor == 0:
                    break
        
        if not all_keys:
            logger.info("No backed up messages found in Redis")
            return

        logger.info(f"Found {len(all_keys)} backed up messages in Redis, recovering...")

        recovered_by_filename: dict[str, list[CleanedWhisperResult]] = defaultdict(list)
        for key in all_keys:
            try:
                async with self.redis_semaphore:
                    message_body = await self.redis_client.get(key)
                if message_body is None:
                    continue
                cleaned_whisper_result: CleanedWhisperResult = serialization.load(message_body.decode())
                filename = cleaned_whisper_result.whisper_result.transcript_metadata.filename
                recovered_by_filename[filename].append(cleaned_whisper_result)
            except Exception as e:
                logger.error(f"Error recovering message from key {key}: {e}")

        recovered_count = 0
        for filename, results in recovered_by_filename.items():
            if filename not in self.sliding_windows_by_filename:
                self.sliding_windows_by_filename[filename] = self._make_sliding_window_for(filename, channel)
            sliding_window = self.sliding_windows_by_filename[filename]
            sliding_window.next_sequence_number = await self._get_recovery_sequence_number(filename)
            logger.info(f"Recovery: starting {filename} from sequence {sliding_window.next_sequence_number}")
            for cleaned_whisper_result in sorted(results, key=lambda r: r.whisper_result.segment_count):
                await sliding_window.append(cleaned_whisper_result)
                recovered_count += 1

        logger.info(f"Successfully recovered {recovered_count} messages from Redis")

    async def run(self):
        """
        Main loop: connect to Redis, then RabbitMQ, recover any backed-up messages,
        then process incoming speaker-identification jobs.
        """
        await self._connect_redis()

        def log_heap_state():
            if not self.sliding_windows_by_filename:
                logger.info("SIGUSR1: no sliding windows active")
                return
            for filename, sliding_window in self.sliding_windows_by_filename.items():
                sequence_numbers = [seq for seq, _ in sliding_window.heap]
                logger.info(f"SIGUSR1: heap for {filename}: {sorted(sequence_numbers)}")

        asyncio.get_event_loop().add_signal_handler(signal.SIGUSR1, log_heap_state)

        connection: AbstractRobustConnection = await aio_pika.connect_robust(
            host=self.config['host'],
            port=self.config['port'],
            login=self.config['username'],
            password=self.config['password'],
            ssl=True,
            ssl_context=self.ssl_context,
        )
        async with connection:
            channel = await connection.channel()
            await channel.set_qos(prefetch_count=1)

            # Recovery runs after the channel is ready so that SlidingWindow
            # callbacks can publish LLM jobs for any unfinished work from a
            # previous session.
            await self._recover_from_redis(channel)

            work_queue = self.config['work_queue']
            queue, _ = await asyncio.gather(
                channel.declare_queue(work_queue, durable=True),
                channel.declare_queue(self.config["destination_queue"], durable=True)
            )

            logger.info(f"Successfully connected! Listening for speaker identification jobs on queue: {work_queue}")
            logger.info("Waiting for jobs. To exit press CTRL+C")

            async with queue.iterator() as queue_iter:
                async for message in queue_iter:
                    cleaned_whisper_result: CleanedWhisperResult = serialization.load(message.body.decode())
                    filename = cleaned_whisper_result.whisper_result.transcript_metadata.filename
                    sequence_number = cleaned_whisper_result.whisper_result.segment_count

                    # Backup message to Redis BEFORE acknowledging or processing
                    await self._backup_message_to_redis(filename, sequence_number, message.body)

                    try:
                        if filename not in self.sliding_windows_by_filename:
                            self.sliding_windows_by_filename[filename] = self._make_sliding_window_for(filename, channel)
                        await self.sliding_windows_by_filename[filename].append(cleaned_whisper_result)
                    except TypeError as type_err:
                        if "not supported between instances" not in str(type_err):
                            raise
                        logger.info(f"duplicate sequence number for {filename}, skipping")
                        await publish_event(
                            self.config,
                            f"NAME_PRODUCER_DUPLICATE_SEQUENCE: {filename} segment {sequence_number} "
                            f"arrived at {self.config['work_queue']} queue more than once. "
                            f"This means the same segment was published to the accepted queue multiple times "
                            f"by an upstream service (likely GATE). This is direct evidence of duplicate processing."
                        )

                    # Now safe to acknowledge - message is backed up in Redis
                    await message.ack()


async def main(config):
    """
    Main function to start the speaker identification consumer with reconnection logic.
    Handles connection failures and automatically reconnects with exponential backoff.
    """
    logger.info("Initializing speaker identification consumer...")
    producer = SpeakerIdentificationProducer(config)
    await producer.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='RabbitMQ consumer for LLM speaker identification jobs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Configuration file format (JSON):
{
    "work_queue": "llm/qwen32",
    "model_path": "/path/to/model",
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
    config = load_config(args.config_file)

    logger.info(f"Loaded configuration from: {args.config_file}")
    logger.info(f"Work queue: {config['work_queue']}")
    logger.info(f"Model path: {config.get('model_path', 'default (Qwen/Qwen2.5-7B-Instruct)')}")
    logger.info(f"RabbitMQ host: {config['host']}:{config['port']}")
    logger.info(f"Username: {config['username']}")

    try:
        asyncio.run(main(config))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
