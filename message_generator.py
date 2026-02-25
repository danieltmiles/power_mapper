"""
Faker-based message generator for creating test messages matching CleanedWhisperResult structure.
"""
import random
from faker import Faker
from wire_formats import (
    CleanedWhisperResult,
    WhisperResult,
    WhisperTimings,
    TranscriptMetadata,
)
from typing import List


def generate_messages(num_messages: int, seed: int | None = None) -> List[CleanedWhisperResult]:
    """
    Generate a list of mock messages using Faker.
    
    Args:
        num_messages: The number of messages to generate
        seed: Optional seed for random number generator for reproducible results.
              If provided, the same seed will always produce the same messages.
        
    Returns:
        A list of CleanedWhisperResult objects with realistic fake data
    """
    if seed is not None:
        random.seed(seed)
        Faker.seed(seed)
    
    fake = Faker()
    messages = []
    
    # Use a consistent filename for all messages in a batch
    filename = fake.file_name(extension="wav")
    
    # Generate speaker names
    speakers = [f"SPEAKER_{i:02d}" for i in range(num_messages // 2 + 1)]
    
    # Track the previous speaker for potential repetition
    previous_speaker = None
    
    # Create messages with sequential segments
    for i in range(num_messages):
        # Generate timings - each segment is about 5 seconds
        start_time = i * 5.0
        end_time = (i + 1) * 5.0
        
        timings = WhisperTimings(
            start=start_time,
            end=end_time
        )
        
        metadata = TranscriptMetadata(
            filename=filename,
            meeting_title=fake.sentence(nb_words=4),
            session_type=fake.word(),
            date=fake.date_time_this_year(),
            video_id=fake.uuid4()
        )
        
        # 20% chance to repeat the previous speaker, otherwise randomly select a speaker
        if previous_speaker and random.random() < 0.2:
            speaker = previous_speaker
        else:
            speaker = random.choice(speakers)
        
        previous_speaker = speaker
        
        # Create a WhisperResult with a mock transcript dict
        whisper_result = WhisperResult(
            transcript={
                "text": fake.paragraph(nb_sentences=50),
                "chunks": []
            },
            speaker=speaker,
            timings=timings,
            transcript_metadata=metadata,
            segment_count=i,
            total_segments=num_messages,
            tries=0
        )
        
        # Create the CleanedWhisperResult
        cleaned_result = CleanedWhisperResult(
            cleaned_transcript=whisper_result.transcript["text"],
            whisper_result=whisper_result
        )
        
        messages.append(cleaned_result)
    
    # Randomize the order of messages before returning
    random.shuffle(messages)
    
    return messages
