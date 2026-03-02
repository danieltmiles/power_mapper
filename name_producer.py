"""
Speaker Identification producer for RabbitMQ
"""
import heapq
import uuid
from asyncio import Semaphore
from collections import defaultdict
from typing import Any

import aio_pika
import argparse
import asyncio
import redis.asyncio as redis
import ssl as ssl_module

import httpx
from aio_pika.abc import AbstractRobustConnection, AbstractRobustChannel

import serialization
from utils import load_config, create_ssl_context
from wire_formats import CleanedWhisperResult, LLMPromptJob, Metaparams

# sequence_number (0-indexed) and filename will uniquely identify a transcript segment
# The context window that will fit on the RTX3090 is 10240 tokens. Since a rough estimate is
# 1 token ~= 4 characters and we want to leave room for the LLM's output, lets limit our
# prompt length to half the context, 5120 tokens, * 4 characters/token = 20480 characters.
# Since the prompt template is 1694 characters, that leaves us with about 18786 characters for
# the transcript window. Call it 18000 for a safety factor. So, we need an in-order sliding
# window of transcript segments from these messages which contain one segment each and may
# come out of order.

MAX_PROMPT_TRANSCRIPT_LENGTH = 10000
PROMPT_TEMPLATE = """The following is an excerpt from an automatically transcribed meeting:
<transcript>
{transcript}
</transcript>

The transcription software did not know who was speaking and assigned names like SPEAKER_1, SPEAKER_2, etc. Please infer speaker identities from context.
APPROACH:
Reason through the clues step-by-step and Look for patterns.

CRITICAL RULES:
1. Use full names when possible (e.g., "Joe Smith", "Mayor Jane Doe")
2. NEVER output bare titles like "Moderator", "Chair", "Professor" - always include the person's name
3. Output ONLY valid JSON, no markdown, no explanation
4. Results of, Unknown, should have a confidence score of 1

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

def form_transcript_chunk(cleaned_result: CleanedWhisperResult) -> str:
    # segments can be assembled like this:
    # [{start_seconds}-{end_seconds}] {speaker}:\n{cleaned_transcript}
    speaker = cleaned_result.whisper_result.speaker
    start_seconds = cleaned_result.whisper_result.timings.start
    end_seconds = cleaned_result.whisper_result.timings.end
    return f"[{start_seconds:.2f}-{end_seconds:.2f}] {speaker.strip()}:\n{cleaned_result.cleaned_transcript.strip()}\n\n"

def safe_heappop(heap: list) -> Any:
    try:
        return heapq.heappop(heap)
    except IndexError:
        return None

def safe_heappeek(heap: list) -> Any:
    resp = safe_heappop(heap)
    if resp is not None:
        heapq.heappush(heap, resp)
    return resp

class SpeakerIdentificationProducer:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.ssl_context = create_ssl_context()
        # collected_transcript_sections[filename] is a min-heap of CleanedWhisperResult objects
        # ordered by whisper_result.segment_count (lowest segment_count at index 0)
        # Each item in the heap is a tuple: (segment_count, CleanedWhisperResult)
        self.collected_transcript_sections = defaultdict(list)
        self.filename_sequence_number_indexes = defaultdict(lambda: -1)
        self.whisper_results_by_filename = defaultdict(list)
        self.redis_client = None
        self.redis_semaphore = Semaphore(10)

    async def _connect_redis(self):
        """Connect to Redis for message backup."""
        redis_config = self.config.get('redis', {})
        if not redis_config:
            print("Warning: No Redis configuration found. Message backup disabled.")
            return
        
        self.redis_client = await redis.Redis(
            host=redis_config['host'],
            port=redis_config['port'],
            ssl=redis_config.get('ssl', False),
            ssl_ca_certs=redis_config.get('ssl_ca_certs'),
            ssl_certfile=redis_config.get('ssl_certfile'),
            ssl_keyfile=redis_config.get('ssl_keyfile'),
        )
        print(f"Connected to Redis at {redis_config['host']}:{redis_config['port']}")

    async def _backup_message_to_redis(self, filename: str, sequence_number: int, message_body: bytes):
        """Backup message to Redis before acknowledging."""
        if self.redis_client is None:
            return

        # Use a Redis key pattern: backup:{filename}:{sequence_number}
        key = f"backup:{filename}:{sequence_number}"
        async with self.redis_semaphore:
            await self.redis_client.set(key, message_body)

    async def _remove_message_from_redis(self, filename: str, sequence_number: int):
        """Remove backed up message from Redis after processing."""
        if self.redis_client is None:
            return
        
        key = f"backup:{filename}:{sequence_number}"
        async with self.redis_semaphore:
            await self.redis_client.delete(key)

    async def _cleanup_filename_backups(self, filename: str):
        """Remove all backup messages for a completed filename."""
        if self.redis_client is None:
            return
        
        # Find all keys matching the pattern
        pattern = f"backup:{filename}:*"
        cursor = 0
        while True:
            async with self.redis_semaphore:
                cursor, keys = await self.redis_client.scan(cursor, match=pattern, count=100)
                if keys:
                    await self.redis_client.delete(*keys)
                if cursor == 0:
                    break
        print(f"Cleaned up Redis backups for {filename}")

    async def _recover_from_redis(self):
        """Recover messages from Redis after a crash or restart."""
        if self.redis_client is None:
            print("No Redis connection - skipping recovery")
            return
        
        print("Checking Redis for backed up messages...")
        pattern = "backup:*"
        cursor = 0
        recovered_count = 0
        
        # Collect all backup keys
        all_keys = []
        while True:
            async with self.redis_semaphore:
                cursor, keys = await self.redis_client.scan(cursor, match=pattern, count=100)
                all_keys.extend(keys)
                if cursor == 0:
                    break
        
        if not all_keys:
            print("No backed up messages found in Redis")
            return
        
        print(f"Found {len(all_keys)} backed up messages in Redis, recovering...")
        
        # Process each backed up message
        for key in all_keys:
            try:
                # Get the message body from Redis
                async with self.redis_semaphore:
                    message_body = await self.redis_client.get(key)
                
                if message_body is None:
                    continue
                
                # Deserialize the message
                cleaned_whisper_result: CleanedWhisperResult = serialization.load(message_body.decode())
                filename = cleaned_whisper_result.whisper_result.transcript_metadata.filename
                sequence_number = cleaned_whisper_result.whisper_result.segment_count
                
                # Add to collected_transcript_sections heap
                heapq.heappush(
                    self.collected_transcript_sections[filename],
                    (sequence_number, cleaned_whisper_result),
                )
                
                recovered_count += 1
                
            except Exception as e:
                print(f"Error recovering message from key {key}: {e}")
                continue
        
        print(f"Successfully recovered {recovered_count} messages from Redis")
        
        # Now populate whisper_results_by_filename by processing sequences
        for filename in self.collected_transcript_sections:
            # Sort the heap to get items in sequence order
            heap_copy = sorted(self.collected_transcript_sections[filename], key=lambda x: x[0])
            
            # Process messages in sequence to populate whisper_results_by_filename
            for sequence_number, cleaned_whisper_result in heap_copy:
                # Only add consecutive sequences to whisper_results_by_filename
                if sequence_number == self.filename_sequence_number_indexes[filename] + 1:
                    self.whisper_results_by_filename[filename].append(cleaned_whisper_result)
                    self.filename_sequence_number_indexes[filename] = sequence_number
                else:
                    # Stop at the first gap in sequence
                    break
            
            if self.whisper_results_by_filename[filename]:
                print(f"Restored {len(self.whisper_results_by_filename[filename])} sequential messages for {filename}")

    async def send(
        self,
        channel: AbstractRobustChannel,
        filename: str,
        sequence_number: int,
        cached_whisper_result: CleanedWhisperResult,
        transcript_text: str,
    ):
        job_desc = LLMPromptJob(
            job_id=str(uuid.uuid4()),
            filename=filename,
            reply_to=self.config["reply_to"],
            prompt=PROMPT_TEMPLATE.format(transcript=transcript_text.strip()),
            meta_params=Metaparams(
                temperature=0.2,
                top_k=40,
                top_p=None,
                repetition_penalty=1.1,
                stop=["JSON_OUTPUT_END"]
            ),
        )
        await channel.default_exchange.publish(
            aio_pika.Message(
                body=serialization.dumps(job_desc).encode("utf-8"),
                reply_to=self.config["reply_to"],
            ),
            routing_key=self.config["destination_queue"]
        )
        # sliding window:
        # lop off the oldest 30% of the prompts_by_filename[filename]
        # BUT only if we're not at the end of the file
        if sequence_number != cached_whisper_result.whisper_result.total_segments - 1:
            num_sections = len(self.whisper_results_by_filename[filename])
            if num_sections > 2:
                to_lop = int(num_sections * .3)
                self.whisper_results_by_filename[filename] = self.whisper_results_by_filename[filename][to_lop:]
            # Keep the window intact even if num_sections <= 2 for continuity
        else:
            # Clear only when we're done with the entire file
            self.whisper_results_by_filename[filename] = []

#     async def fetch_from_solr_by_filename(self, filename) -> list[CleanedWhisperResult]:
#         whisper_results: list[CleanedWhisperResult] = []
#         auth = httpx.BasicAuth(username=self.config["solr"]["username"], password=self.config["solr"]["password"])
#         collection = "transcripts"
#         base_uri = f"{self.config['solr']['url']}/{collection}"
#         params = {
#             "q": "*:*",
#             "fq": f"filename:\"{filename}\"",
#             "rows": 100,
#             "sort": "sequence_number asc",
#             "start": 0,
#         }
#         async with httpx.AsyncClient(auth=auth, timeout=10) as client:
#             raw_resp = await client.get(
#                 f"{base_uri}/select",
#                 params=params,
#             )
#             raw_resp.raise_for_status()
#             resp = raw_resp.json()
#             num_found = resp.get("response", {}).get("numFound")
#             num_gotten = 0
#             while num_gotten < num_found:
#                 for document in resp.get("response", {}).get("docs", []):
#                     cleaned_whisper_result: CleanedWhisperResult = serialization.load(document["cleaned_whisper_result"])
#                     whisper_results.append(cleaned_whisper_result)
#                     num_gotten += 1
#                 params["start"] = num_gotten
#                 if num_gotten >= num_found:
#                     continue  # loop will exit, don't do another HTTP call
#                 raw_resp = await client.get(
#                     f"{base_uri}/select",
#                     params=params,
#                 )
#                 raw_resp.raise_for_status()
#                 resp = raw_resp.json()
#         return whisper_results
#
    async def clear_and_send(self, filename: str, channel: AbstractRobustChannel):
        collected_transcript_sections_heap = self.collected_transcript_sections[filename]
        popped = safe_heappop(collected_transcript_sections_heap)
        if popped is None:
            return
        sequence_number: int = popped[0]
        cached_whisper_result: CleanedWhisperResult = popped[1]
        is_last_segment = False
        while sequence_number == self.filename_sequence_number_indexes[filename] + 1:
            self.whisper_results_by_filename[filename].append(cached_whisper_result)
            self.filename_sequence_number_indexes[filename] = sequence_number
            # Remove this message from Redis backup as it's been processed
            await self._remove_message_from_redis(filename, sequence_number)
            
            prompt_text = "".join(form_transcript_chunk(r) for r in self.whisper_results_by_filename[filename])
            if sequence_number == cached_whisper_result.whisper_result.total_segments - 1 or \
                    len(prompt_text) > MAX_PROMPT_TRANSCRIPT_LENGTH:
                sequence_numbers = [x.whisper_result.segment_count for x in self.whisper_results_by_filename[filename]]
                last_number = sequence_numbers[0]
                for num in sequence_numbers[1:]:
                    if num != last_number + 1:
                        print(f"non-monotonic number, {num} follows {last_number}")
                    last_number = num
                print(f"sending prompt with texts from sequence numbers: {sequence_numbers[0]}-{sequence_numbers[-1]}")
                coros = [self._remove_message_from_redis(filename, x) for x in sequence_numbers]
                coros.append(self.send(channel, filename, sequence_number, cached_whisper_result, prompt_text))
                await asyncio.gather(*coros)
                
                # Check if this is the last segment of the file
                if sequence_number == cached_whisper_result.whisper_result.total_segments - 1:
                    is_last_segment = True

            popped = safe_heappop(collected_transcript_sections_heap)
            if popped is None:
                break
            sequence_number: int = popped[0]
            cached_whisper_result: CleanedWhisperResult = popped[1]
        else:
            # repair when we couldn't use a sequence number
            if popped:
                heapq.heappush(
                    self.collected_transcript_sections[filename],
                    (sequence_number, cached_whisper_result),
                )
        
        # If we finished processing the last segment, cleanup all backups for this filename
        if is_last_segment:
            await self._cleanup_filename_backups(filename)


    async def run(self):
        # Connect to Redis for message backup
        await self._connect_redis()
        
        # Recover any messages from Redis after a crash/restart
        await self._recover_from_redis()
        
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

            work_queue = self.config['work_queue']
            queue, _ = await asyncio.gather(
                channel.declare_queue(work_queue, durable=True),
                channel.declare_queue(self.config["destination_queue"], durable=True)
            )

            print(f"Successfully connected! Listening for speaker identification jobs on queue: {work_queue}")
            print("Waiting for jobs. To exit press CTRL+C")

            async with queue.iterator() as queue_iter:
                async for message in queue_iter:
                    cleaned_whisper_result: CleanedWhisperResult = serialization.load(message.body.decode())
                    filename = cleaned_whisper_result.whisper_result.transcript_metadata.filename
                    sequence_number = cleaned_whisper_result.whisper_result.segment_count
                    
                    # Backup message to Redis BEFORE acknowledging or processing
                    await self._backup_message_to_redis(filename, sequence_number, message.body)
                    
                    try:
                        heapq.heappush(
                            self.collected_transcript_sections[filename],
                            (sequence_number, cleaned_whisper_result),
                        )
                        await self.clear_and_send(filename, channel)
                    except TypeError as type_err:
                        if "not supported between instances" not in str(type_err):
                            raise
                        print(f"duplicate sequence number for {filename}, skipping")
                    
                    # Now safe to acknowledge - message is backed up in Redis
                    await message.ack()

async def main(config):
    """
    Main function to start the speaker identification consumer with reconnection logic.
    Handles connection failures and automatically reconnects with exponential backoff.
    """
    print("Initializing speaker identification consumer...")
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

    print(f"Loaded configuration from: {args.config_file}")
    print(f"Work queue: {config['work_queue']}")
    print(f"Model path: {config.get('model_path', 'default (Qwen/Qwen2.5-7B-Instruct)')}")
    print(f"RabbitMQ host: {config['host']}:{config['port']}")
    print(f"Username: {config['username']}")

    try:
        asyncio.run(main(config))
    except KeyboardInterrupt:
        print("\nInterrupted by user")
