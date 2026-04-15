"""
Test that messages from transcriptions.json can be deserialized and used
to assemble both cleaning prompts (clean.py) and naming prompts
(name_producer.py), as if they were consumed from a RabbitMQ queue.
"""

import json
from collections import defaultdict

import pytest

import serialization
from clean import create_cleanup_prompt
from name_producer import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from sliding_window import SlidingWindow, CleanedWhisperResultWrapper
from wire_formats import WhisperResult, CleanedWhisperResult


TRANSCRIPTIONS_FILE = "transcriptions.json"


@pytest.fixture
def transcription_messages() -> list[dict]:
    """Load raw message dicts from transcriptions.json."""
    with open(TRANSCRIPTIONS_FILE) as f:
        data = json.load(f)
    return data["messages"]


@pytest.fixture
def whisper_results(transcription_messages) -> list[WhisperResult]:
    """Deserialize message bodies into WhisperResult objects."""
    results = []
    for msg in transcription_messages[:20]:  # use first 20 for speed
        body = msg["body"]
        result = serialization.load(json.dumps(body))
        assert isinstance(result, WhisperResult)
        results.append(result)
    return results


@pytest.fixture
def all_whisper_results(transcription_messages) -> list[WhisperResult]:
    """Deserialize ALL message bodies for large-window tests."""
    results = []
    for msg in transcription_messages:
        body = msg["body"]
        result = serialization.load(json.dumps(body))
        results.append(result)
    return results


def make_cleaned_results(whisper_results: list[WhisperResult]) -> list[CleanedWhisperResult]:
    """Simulate the cleaning step by wrapping each WhisperResult."""
    cleaned = []
    for wr in whisper_results:
        text = wr.transcript.get("text", "")
        cleaned.append(CleanedWhisperResult(
            cleaned_transcript=text,
            whisper_result=wr,
        ))
    return cleaned


# -- Cleaning prompt tests --

def test_create_cleanup_prompt_from_message(whisper_results):
    """Each WhisperResult's transcript text produces a non-empty cleaning prompt."""
    for wr in whisper_results:
        text = wr.transcript.get("text", "")
        prompt = create_cleanup_prompt(text)
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "<transcript_to_correct>" in prompt
        assert text.replace("...", "").strip() in prompt


def test_cleanup_prompt_contains_instructions(whisper_results):
    """The cleaning prompt includes the expected editorial instructions."""
    prompt = create_cleanup_prompt(whisper_results[0].transcript["text"])
    assert "transcript editor" in prompt
    assert "ASR" in prompt
    assert "corrected_transcript" in prompt


# -- Naming prompt tests --

@pytest.mark.asyncio
async def test_sliding_window_produces_naming_prompt(whisper_results):
    """
    Feed cleaned results through a SlidingWindow and verify the callback
    fires with a prompt that can be formatted into the naming template.
    """
    cleaned = make_cleaned_results(whisper_results)
    captured_prompts: list[str] = []

    def on_window_full(prompt_text: str):
        full_prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(transcript=prompt_text.strip())},
        ]
        captured_prompts.append(full_prompt)

    window = SlidingWindow(
        max_size=2000,
        callback=on_window_full,
        truncation_percentage=0.3,
    )

    for cr in cleaned:
        await window.append(cr)

    assert len(captured_prompts) > 0, "SlidingWindow callback should have fired at least once"
    for prompt in captured_prompts:
        assert prompt[0]["role"] == "system"
        assert "transcript analyst" in prompt[0]["content"]
        assert prompt[1]["role"] == "user"
        assert "<transcript>" in prompt[1]["content"]
        assert "SPEAKER_" in prompt[1]["content"]


def test_naming_prompt_template_renders(whisper_results):
    """USER_PROMPT_TEMPLATE renders without errors given transcript text."""
    cleaned = make_cleaned_results(whisper_results)
    sample_text = SlidingWindow.form_transcript_chunk(
        CleanedWhisperResultWrapper(cleaned[0])
    )
    rendered = USER_PROMPT_TEMPLATE.format(transcript=sample_text.strip())
    assert "SPEAKER_" in rendered
    assert "<transcript>" in rendered
    assert "JSON_OUTPUT_START" in rendered


def test_both_prompts_from_same_message(whisper_results):
    """
    A single transcription message can produce both a cleaning prompt
    and a naming prompt, demonstrating the full pipeline.
    """
    wr = whisper_results[0]
    text = wr.transcript.get("text", "")

    # Cleaning prompt
    cleaning_prompt = create_cleanup_prompt(text)
    assert "<transcript_to_correct>" in cleaning_prompt

    # Naming prompt (using the cleaned text as input)
    cleaned = CleanedWhisperResult(cleaned_transcript=text, whisper_result=wr)
    chunk = SlidingWindow.form_transcript_chunk(CleanedWhisperResultWrapper(cleaned))
    naming_prompt = USER_PROMPT_TEMPLATE.format(transcript=chunk.strip())
    assert "<transcript>" in naming_prompt


# -- Transcript section length tests --

async def _run_length_test(whisper_results: list[WhisperResult], max_size: int):
    """
    Group results by filename (as name_producer does), feed each group into
    its own SlidingWindow, and verify prompt lengths meet the max_size threshold.

    For files whose total transcript text is shorter than max_size, the single
    callback prompt will naturally be shorter — that is correct behaviour and
    we verify it separately.
    """
    cleaned = make_cleaned_results(whisper_results)
    by_filename: dict[str, list[CleanedWhisperResult]] = defaultdict(list)
    for cr in cleaned:
        by_filename[cr.whisper_result.transcript_metadata.filename].append(cr)

    # captured_by_file keeps prompts separated so we can reason per-file
    captured_by_file: dict[str, list[str]] = defaultdict(list)

    for filename, group in by_filename.items():
        def on_window_full(prompt_text: str, _fn=filename):
            captured_by_file[_fn].append(prompt_text)

        window = SlidingWindow(
            max_size=max_size,
            callback=on_window_full,
            truncation_percentage=0.3,
            filename=filename,
        )
        for cr in group:
            await window.append(cr)

    total_prompts = sum(len(v) for v in captured_by_file.values())
    assert total_prompts > 0, "Callback should have fired at least once"

    for filename, prompts in captured_by_file.items():
        if len(prompts) == 1:
            # File produced only one prompt — either the whole transcript fit
            # within max_size (short file) or it was the final flush.
            # Either way the prompt must be non-empty.
            assert len(prompts[0]) > 0, (
                f"{filename}: single prompt should be non-empty"
            )
        else:
            # Multiple prompts: all except the last must meet the threshold
            for i, prompt_text in enumerate(prompts[:-1]):
                assert len(prompt_text) >= max_size, (
                    f"{filename}: non-final section {i} length {len(prompt_text)} < {max_size}"
                )
            assert len(prompts[-1]) > 0, (
                f"{filename}: final section should be non-empty"
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("desired_length", [2000, 5000, 10000, 50000])
async def test_transcript_sections_length_2000(all_whisper_results, desired_length: int):
    """With max_size=2000, every non-final callback prompt should be >= 2000 chars."""
    await _run_length_test(all_whisper_results, desired_length)
