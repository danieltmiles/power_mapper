#!/usr/bin/env python3
"""
SLICE - Segment and Label Individual Chunks for Extraction

This microservice waits for diarization results from the diarization-reply queue,
consolidates segments by speaker, and sends them to the whisper/large queue for
transcription. Each message includes the diarization job metadata for tracking.

This service does NOT subscribe to whisper replies - transcription will be handled
by a separate service.
"""

import asyncio
import json
import os
import sys
import base64
import pickle
import uuid
import argparse
from pathlib import Path
from typing import Any

import aio_pika
import numpy as np
from aiormq import ChannelInvalidStateError, AMQPError, ChannelClosed

import serialization
import shared_disks
from utils import diarized_segment_iter, assign_speaker_to_segment, normalize_audio, create_ssl_context
from wire_formats import DiarizationResponse, WhisperJobDescription, WhisperJobAudioSegment


class AssembleDiarizationService:
    """Service that assembles diarization results and sends them for transcription."""
    
    def __init__(self, config_file: str = "rabbitmq_config.json"):
        """
        Initialize the assemble diarization service.
        
        Args:
            config_file: Path to RabbitMQ configuration file
        """
        self.config = self._load_config(config_file)
        
        # Queue configuration
        self.work_queue = self.config.get("work_queue")
        self.destination_queue = self.config.get("destination_queue")

        print(f"Assemble Diarization Service initialized")
        print(f"  Listening on: {self.work_queue}")
        print(f"  Sending to: {self.destination_queue}")

    def _load_config(self, config_file: str) -> dict[str, Any]:
        """Load RabbitMQ configuration from JSON file."""
        if not os.path.exists(config_file):
            raise FileNotFoundError(
                f"Configuration file not found: {config_file}\n"
                f"Please ensure rabbitmq_config.json exists"
            )
        
        with open(config_file, 'r') as f:
            return json.load(f)
    
    def consolidate_segments_by_speaker(
        self,
        signal: np.ndarray,
        diarization,
        sr: int
    ) -> list[dict[str, Any]]:
        """
        Consolidate consecutive segments with the same speaker.
        
        This matches the logic in ai.py's send_whisper_jobs function.
        
        Args:
            signal: Audio signal array
            diarization: Diarization output (deserialized)
            sr: Sample rate
            
        Returns:
            List of consolidated segments with audio, start, end, and speaker
        """
        consolidated_segments = []
        current_speaker = None
        accumulated_audio = []
        accumulated_start = None
        accumulated_end = None
        
        for segment in diarized_segment_iter(signal, diarization, sr):
            speaker = segment['speaker']
            
            if current_speaker != speaker:
                # Different speaker - save accumulated segment if exists
                if accumulated_audio:
                    consolidated_segments.append({
                        'audio': np.concatenate(accumulated_audio),
                        'start': accumulated_start,
                        'end': accumulated_end,
                        'speaker': current_speaker,
                    })
                
                # Start new accumulation
                accumulated_audio = [segment['audio']]
                accumulated_start = segment['start']
                accumulated_end = segment['end']
                current_speaker = speaker
            else:
                # Same speaker - accumulate audio
                accumulated_audio.append(segment['audio'])
                accumulated_end = segment['end']
        
        # Don't forget the last accumulated segment
        if accumulated_audio:
            consolidated_segments.append({
                'audio': np.concatenate(accumulated_audio),
                'start': accumulated_start,
                'end': accumulated_end,
                'speaker': current_speaker,
            })
        
        return consolidated_segments
    
    def split_large_segments(
        self,
        consolidated_segments: list[dict[str, Any]],
        sr: int
    ) -> list[dict[str, Any]]:
        """
        Split large consolidated segments to avoid RabbitMQ size limits.
        
        This matches the logic in ai.py's send_whisper_jobs function.
        
        Args:
            consolidated_segments: List of consolidated segments
            sr: Sample rate
            
        Returns:
            List of segments, with large ones split into chunks
        """
        # JSON encoding creates significant overhead: each float32 becomes ~10 chars in JSON
        # To stay under 128MB, limit to ~3M samples
        MAX_AUDIO_SAMPLES = 3_000_000
        MIN_SEGMENT_DURATION = 0.5  # Minimum segment duration in seconds
        
        segments_list = []
        for segment in consolidated_segments:
            # Skip segments that are less than half a second long
            segment_duration = segment['end'] - segment['start']
            if segment_duration < MIN_SEGMENT_DURATION:
                print(f"Skipping short segment ({segment_duration:.3f}s) for speaker {segment['speaker']}")
                continue
            
            audio_data = segment['audio']
            if len(audio_data) <= MAX_AUDIO_SAMPLES:
                # Segment is small enough, add as-is
                segments_list.append(segment)
            else:
                # Split large segment into chunks
                print(f"Splitting large segment ({len(audio_data)} samples) for speaker {segment['speaker']}")
                num_chunks = (len(audio_data) + MAX_AUDIO_SAMPLES - 1) // MAX_AUDIO_SAMPLES
                chunk_size = len(audio_data) // num_chunks
                
                for i in range(num_chunks):
                    chunk_start_idx = i * chunk_size
                    chunk_end_idx = min((i + 1) * chunk_size, len(audio_data))
                    chunk_audio = audio_data[chunk_start_idx:chunk_end_idx]
                    
                    # Calculate time offsets for this chunk
                    chunk_duration = len(chunk_audio) / sr
                    chunk_start_time = segment['start'] + (chunk_start_idx / sr)
                    chunk_end_time = chunk_start_time + chunk_duration
                    
                    segments_list.append({
                        'audio': chunk_audio,
                        'start': chunk_start_time,
                        'end': chunk_end_time,
                        'speaker': segment['speaker'],
                    })
                    print(f"  Created chunk {i+1}/{num_chunks}: {len(chunk_audio)} samples")
        
        return segments_list
    
    async def process_diarization_result(
        self,
        diarization_response: DiarizationResponse,
        channel: aio_pika.abc.AbstractRobustChannel,
    ) -> None:
        """
        Process a single diarization result and send whisper jobs.
        
        Args:
            message: Diarization result message
            channel: RabbitMQ channel for sending whisper jobs
        """

        # Deserialize diarization result
        diarization_encoded = diarization_response.diarization
        if not diarization_encoded:
            print(f"✗ No diarization data in message")
            return
        
        try:
            diarization_bytes = base64.b64decode(diarization_encoded)
            diarization = pickle.loads(diarization_bytes)
        except Exception as e:
            print(f"✗ Failed to deserialize diarization: {e}")
            return
        
        # Get the audio file from remote storage
        # We need to download it to process it

        file_basename = Path(diarization_response.transcript_metadata.filename).name
        temp_audio_path = f"/tmp/{file_basename}"
        storage_info = self.config.get("storage_info")
        storage: shared_disks.RemoteStorage = shared_disks.factory(
            remote_file_type=storage_info.get("type"),
            info=storage_info,
        )

        try:
            await asyncio.to_thread(
                storage.retrieve,
                diarization_response.transcript_metadata.filename,
                temp_audio_path,
            )
            print(f"  Downloaded to {temp_audio_path}")
        except Exception as e:
            print(f"✗ Failed to download audio: {e}")
            return

        try:
            # Normalize audio
            print("Normalizing audio...")
            signal, sr = normalize_audio(temp_audio_path)
            
            # Consolidate segments by speaker
            print("Consolidating segments by speaker...")
            consolidated_segments = self.consolidate_segments_by_speaker(
                signal.numpy() if hasattr(signal, 'numpy') else signal,
                diarization,
                sr
            )
            print(f"  Created {len(consolidated_segments)} consolidated segments")
            
            # Split large segments
            print("Splitting large segments if needed...")
            segments_list = self.split_large_segments(consolidated_segments, sr)
            print(f"  Final segment count: {len(segments_list)}")
            
            # Send whisper jobs
            await self.send_whisper_jobs(
                segments_list,
                diarization_response,
                channel
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                print(f"Cleaned up temporary file: {temp_audio_path}")
    
    async def wait_for_queue_backpressure(
        self,
        channel: aio_pika.abc.AbstractRobustChannel,
        max_messages: int = 1000,
        check_interval: float = 2.0,
    ) -> None:
        """
        Wait until the destination queue has fewer than max_messages.
        
        This implements back-pressure to prevent overwhelming the queue.
        
        Args:
            channel: RabbitMQ channel
            max_messages: Maximum number of messages to allow in queue
            check_interval: Seconds to wait between checks
        """
        while True:
            # Declare queue to get fresh statistics (declaring existing durable queue is idempotent)
            queue_info = await channel.declare_queue(
                self.destination_queue,
                durable=True,
            )
            message_count = queue_info.declaration_result.message_count

            if message_count < max_messages:
                return
            
            print(f"  Back-pressure: Queue has {message_count} messages, waiting for it to drop below {max_messages}...")
            await asyncio.sleep(check_interval)
    
    async def send_whisper_jobs(
        self,
        segments_list: list[dict[str, Any]],
        diarization_response: DiarizationResponse,
        channel: aio_pika.abc.AbstractRobustChannel,
    ) -> None:
        total_segments = len(segments_list)
        whisper_job_id = str(uuid.uuid4())
        
        print(f"\nSending {total_segments} segments to {self.destination_queue}...")
        print(f"  Whisper job ID: {whisper_job_id}")

        for i, segment in enumerate(segments_list):

            # Create job message matching ai.py's format
            job_desc = WhisperJobDescription(
                audio_segment=WhisperJobAudioSegment(
                    audio=segment['audio'].tolist(),
                    start=segment['start'],
                    end=segment['end'],
                    speaker=segment['speaker'],
                ),
                segment_count=i,
                total_segments=total_segments,
                transcript_metadata=diarization_response.transcript_metadata,
            )

            await channel.default_exchange.publish(
                aio_pika.Message(
                    body=serialization.dumps(job_desc).encode(),
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                ),
                routing_key=self.destination_queue,
            )
            
            if (i + 1) % 10 == 0 or (i + 1) == total_segments:
                print(f"  Sent {i + 1}/{total_segments} whisper jobs")
                await self.wait_for_queue_backpressure(channel)

        print(f"✓ Successfully sent all {total_segments} whisper jobs")

    async def run(self) -> None:
        """
        Run the service - consume from diarization-reply queue.
        """
        print(f"\n{'='*60}")
        print("Starting Assemble Diarization Service")
        print(f"{'='*60}\n")
        
        # Connect to RabbitMQ
        ssl_context = create_ssl_context()

        while True:
            try:
                connection = await aio_pika.connect_robust(
                    host=self.config['host'],
                    port=self.config['port'],
                    login=self.config['username'],
                    password=self.config['password'],
                    ssl=True,
                    ssl_context=ssl_context,
                )
                async with connection:
                    channel = await connection.channel()
                    await channel.set_qos(prefetch_count=1)

                    # Declare queues
                    diarization_queue, _ = await asyncio.gather(
                        channel.declare_queue(self.work_queue, durable=True),
                        channel.declare_queue(self.destination_queue, durable=True),
                    )

                    print(f"Listening for diarization results on: {self.work_queue}")
                    print("Waiting for messages... (Press Ctrl+C to exit)\n")

                    # Consume messages
                    async with diarization_queue.iterator() as queue_iter:
                        async for message in queue_iter:
                            async with message.process(requeue=True):
                                try:
                                    diarization_result: DiarizationResponse = serialization.load(message.body.decode())
                                    await self.process_diarization_result(diarization_result, channel)
                                except json.JSONDecodeError as e:
                                    print(f"✗ Failed to decode message: {e}")
                                    raise
            except (AMQPError, ChannelInvalidStateError, ChannelClosed, ConnectionError, Exception) as conn_error:
                try:
                    # re-dial
                    print(f"{conn_error}\n\nChannel closed unexpectedly, reconnecting...")
                    if connection and not connection.is_closed:
                        await connection.close()
                    if channel and not channel.is_closed:
                        await channel.close()
                except Exception:
                    pass
                # go around again





async def main():
    """Main entry point for the assemble diarization service."""
    parser = argparse.ArgumentParser(
        description="Assemble Diarization Service - Process diarization results and send for transcription"
    )
    parser.add_argument(
        'config_file',
        type=str,
        help='Path to the JSON configuration file'
    )

    args = parser.parse_args()
    
    try:
        service = AssembleDiarizationService(config_file=args.config_file)
        await service.run()
    except FileNotFoundError as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n✓ Service stopped by user")
    except Exception as e:
        print(f"✗ Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
