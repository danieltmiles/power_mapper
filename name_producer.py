"""
Speaker Identification Worker for RabbitMQ

This worker processes transcript windows to identify speaker names from context clues
using an LLM with a sliding window approach.
"""
import json
import re
import heapq
from collections import defaultdict
from typing import Any

import aio_pika
import argparse
import asyncio

from aio_pika.abc import AbstractIncomingMessage, AbstractRobustConnection, AbstractRobustChannel
from aiormq import ChannelInvalidStateError, ChannelClosed, AMQPError
from pamqp.commands import Basic

import serialization
from utils import load_config, create_ssl_context, load_quantized_llm_model, quantized_generate_from_prompt
from wire_formats import CleanedWhisperResult

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

class SpeakerIdentificationProducer:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.ssl_context = create_ssl_context()
        # collected_transcript_sections[filename] is a min-heap of CleanedWhisperResult objects
        # ordered by whisper_result.segment_count (lowest segment_count at index 0)
        # Each item in the heap is a tuple: (segment_count, CleanedWhisperResult)
        self.collected_transcript_sections = defaultdict(list)
        self.filename_sequence_number_indexes = defaultdict(lambda: -1)
        self.prompts_by_filename = defaultdict(str)

    async def clear_and_send(self, filename: str, channel: AbstractRobustChannel):
        collected_transcript_sections_heap = self.collected_transcript_sections[filename]
        popped = safe_heappop(collected_transcript_sections_heap)
        if popped is None:
            return
        sequence_number: int = popped[0]
        cached_whisper_result: CleanedWhisperResult = popped[1]
        while sequence_number == self.filename_sequence_number_indexes[filename] + 1:
            self.filename_sequence_number_indexes[filename] += 1
            self.prompts_by_filename[filename] += form_transcript_chunk(cached_whisper_result)
            if sequence_number == cached_whisper_result.whisper_result.total_segments - 1 or \
                    len(self.prompts_by_filename[filename]) > 18000:
                # TODO: somehow ack all contributing messages
                await channel.default_exchange.publish(
                    aio_pika.Message(
                        body=self.prompts_by_filename[filename].encode("utf-8"),
                    ),
                    routing_key=self.config["destination_queue"]
                )
                # TODO: this should be a sliding window, not a hard cutoff.
                self.prompts_by_filename[filename] = ""
            popped = safe_heappop(collected_transcript_sections_heap)
            if popped is None:
                return
            sequence_number: int = popped[0]
            cached_whisper_result: CleanedWhisperResult = popped[1]
        else:
            # repair when we couldn't use a sequence number
            if popped:
                heapq.heappush(
                    self.collected_transcript_sections[filename],
                    (sequence_number, cached_whisper_result),
                )
        if len(self.prompts_by_filename[filename]) > 18000:
            # TODO: send speaker identification job to rabbitmq
            # TODO: somehow ack all contributing messages
            pass


    async def run(self):
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
            queue = await channel.declare_queue(work_queue, durable=True)

            print(f"Successfully connected! Listening for speaker identification jobs on queue: {work_queue}")
            print("Waiting for jobs. To exit press CTRL+C")

            async with queue.iterator() as queue_iter:
                async for message in queue_iter:
                    # TODO: figure out how and when to ack
                    cleaned_whisper_result: CleanedWhisperResult = serialization.load(message.body.decode())
                    filename = cleaned_whisper_result.whisper_result.transcript_metadata.filename
                    sequence_number = cleaned_whisper_result.whisper_result.segment_count
                    heapq.heappush(
                        self.collected_transcript_sections[filename],
                        (sequence_number, cleaned_whisper_result),
                    )
                    await self.clear_and_send(filename, channel)
                    # sequence_number (0-indexed) and filename will uniquely identify a transcript segment
                    # The context window that will fit on the RTX3090 is 10240 tokens. Since a rough estimate is
                    # 1 token ~= 4 characters and we want to leave room for the LLM's output, lets limit our
                    # prompt length to half the context, 5120 tokens, * 4 characters/token = 20480 characters.
                    # Since the prompt template is 1694 characters, that leaves us with about 18786 characters for
                    # the transcript window. Call it 18000 for a safety factor. So, we need an in-order sliding
                    # window of transcript segments from these messages which contain one segment each and may
                    # come out of order.

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
