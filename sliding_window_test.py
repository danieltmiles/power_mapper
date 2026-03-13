"""
Pytest test suite for the SlidingWindow class in sliding_window.py
"""
import pytest
import pytest_mock
from dataclasses import dataclass

from sliding_window import SlidingWindow, CleanedWhisperResultWrapper
from wire_formats import (
    CleanedWhisperResult,
    WhisperResult,
    WhisperTimings,
    TranscriptMetadata,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class FakeSequencedText:
    """A minimal concrete implementation of the SequencedText protocol."""
    speaker: str
    start_seconds: float
    end_seconds: float
    text: str
    sequence_number: int
    sequence_count: int
    truncation_percentage: float = 0.0


def make_segment(
    speaker: str = "Speaker A",
    start: float = 0.0,
    end: float = 5.0,
    text: str = "Hello world.",
    sequence_number: int = 0,
    sequence_count: int = 10,
) -> FakeSequencedText:
    """Helper that returns a FakeSequencedText with sensible defaults."""
    return FakeSequencedText(
        speaker=speaker,
        start_seconds=start,
        end_seconds=end,
        text=text,
        sequence_number=sequence_number,
        sequence_count=sequence_count,
    )


def make_cleaned_whisper_result(
    speaker: str = "Speaker B",
    start: float = 1.0,
    end: float = 3.0,
    cleaned_transcript: str = "Cleaned text.",
    segment_count: int = 0,
    total_segments: int = 5,
) -> CleanedWhisperResult:
    """Helper that returns a CleanedWhisperResult."""
    metadata = TranscriptMetadata(filename="test.mp3")
    timings = WhisperTimings(start=start, end=end)
    whisper_result = WhisperResult(
        transcript={"text": "raw text"},
        speaker=speaker,
        timings=timings,
        transcript_metadata=metadata,
        segment_count=segment_count,
        total_segments=total_segments,
    )
    return CleanedWhisperResult(
        cleaned_transcript=cleaned_transcript,
        whisper_result=whisper_result,
    )


# ---------------------------------------------------------------------------
# Initialisation tests
# ---------------------------------------------------------------------------

def test_valid_construction(mocker: pytest_mock.MockerFixture):
    callback = mocker.MagicMock()
    sliding_window = SlidingWindow(max_size=100, callback=callback)
    assert sliding_window.max_size == 100
    assert sliding_window.callback is callback
    assert sliding_window.window == []
    assert sliding_window.truncation_percentage == 0.3  # default


def test_invalid_truncation_percentage_negative(mocker: pytest_mock.MockerFixture):
    with pytest.raises(ValueError, match="truncation percentage must be a float value from 0-1"):
        SlidingWindow(max_size=100, callback=mocker.MagicMock(), truncation_percentage=-0.1)


def test_invalid_truncation_percentage_above_one(mocker: pytest_mock.MockerFixture):
    with pytest.raises(ValueError, match="truncation percentage must be a float value from 0-1"):
        SlidingWindow(max_size=100, callback=mocker.MagicMock(), truncation_percentage=1.1)


# ---------------------------------------------------------------------------
# form_transcript_chunk tests
# ---------------------------------------------------------------------------

def test_form_transcript_chunk_basic_formatting():
    segment = make_segment(speaker="Alice", start=1.5, end=4.75, text="Hello there.")
    result = SlidingWindow.form_transcript_chunk(segment)
    assert result == "[1.50-4.75] Alice:\nHello there.\n\n"


def test_form_transcript_chunk_speaker_is_stripped():
    segment = make_segment(speaker="  Bob  ", text="Testing.")
    result = SlidingWindow.form_transcript_chunk(segment)
    assert "Bob:" in result
    assert "  Bob  " not in result


def test_form_transcript_chunk_text_is_stripped():
    segment = make_segment(text="  some text  ")
    result = SlidingWindow.form_transcript_chunk(segment)
    assert "some text" in result
    assert "  some text  " not in result


def test_form_transcript_chunk_trailing_newlines():
    segment = make_segment()
    result = SlidingWindow.form_transcript_chunk(segment)
    assert result.endswith("\n\n")


def test_form_transcript_chunk_float_formatting_two_decimal_places():
    segment = make_segment(start=0.0, end=10.0)
    result = SlidingWindow.form_transcript_chunk(segment)
    assert "[0.00-10.00]" in result


# ---------------------------------------------------------------------------
# prompt_text property tests
# ---------------------------------------------------------------------------

def test_prompt_text_empty_window(mocker: pytest_mock.MockerFixture):
    sliding_window = SlidingWindow(max_size=10_000, callback=mocker.MagicMock())
    assert sliding_window.prompt_text == ""


def test_prompt_text_single_segment(mocker: pytest_mock.MockerFixture):
    sliding_window = SlidingWindow(max_size=10_000, callback=mocker.MagicMock())
    seg = make_segment(speaker="Alice", start=0.0, end=1.0, text="Hi.")
    sliding_window.window.append(seg)
    assert sliding_window.prompt_text == SlidingWindow.form_transcript_chunk(seg)


def test_prompt_text_multiple_segments_concatenated(mocker: pytest_mock.MockerFixture):
    sliding_window = SlidingWindow(max_size=10_000, callback=mocker.MagicMock())
    seg1 = make_segment(speaker="Alice", start=0.0, end=1.0, text="Hi.")
    seg2 = make_segment(speaker="Bob", start=1.0, end=2.0, text="Hello.")
    sliding_window.window.extend([seg1, seg2])
    expected = (
        SlidingWindow.form_transcript_chunk(seg1)
        + SlidingWindow.form_transcript_chunk(seg2)
    )
    assert sliding_window.prompt_text == expected


# ---------------------------------------------------------------------------
# __len__ tests
# ---------------------------------------------------------------------------

def test_len_empty(mocker: pytest_mock.MockerFixture):
    sliding_window = SlidingWindow(max_size=10_000, callback=mocker.MagicMock())
    assert len(sliding_window) == 0


def test_len_matches_prompt_text_length(mocker: pytest_mock.MockerFixture):
    sliding_window = SlidingWindow(max_size=10_000, callback=mocker.MagicMock())
    seg = make_segment(speaker="Alice", start=0.0, end=1.0, text="Hello world.")
    sliding_window.window.append(seg)
    assert len(sliding_window) == len(sliding_window.prompt_text)


def test_len_delegates_to_prompt_text_length(mocker: pytest_mock.MockerFixture):
    sliding_window = SlidingWindow(max_size=10_000, callback=mocker.MagicMock())
    seg = make_segment(speaker="Z", start=10.0, end=20.0, text="Some text here.")
    sliding_window.window.append(seg)
    assert len(sliding_window) == len(sliding_window.prompt_text)
    assert len(sliding_window) > 0


# ---------------------------------------------------------------------------
# CleanedWhisperResultWrapper tests
# ---------------------------------------------------------------------------

def test_wrapper_attributes():
    cwr = make_cleaned_whisper_result(
        speaker="Charlie",
        start=2.0,
        end=6.0,
        segment_count=3,
        total_segments=10,
    )
    wrapper = CleanedWhisperResultWrapper(cwr)
    assert wrapper.speaker == "Charlie"
    assert wrapper.start_seconds == 2.0
    assert wrapper.end_seconds == 6.0
    assert wrapper.sequence_number == 3
    assert wrapper.sequence_count == 10
    assert wrapper.cleaned_whisper_result is cwr


def test_wrapper_stored_in_window(mocker: pytest_mock.MockerFixture):
    """A CleanedWhisperResultWrapper placed in the window is retrievable."""
    sliding_window = SlidingWindow(max_size=10_000, callback=mocker.MagicMock())
    cwr = make_cleaned_whisper_result()
    wrapped = CleanedWhisperResultWrapper(cwr)
    sliding_window.window.append(wrapped)
    assert isinstance(sliding_window.window[0], CleanedWhisperResultWrapper)


# ---------------------------------------------------------------------------
# append – sync callback tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_callback_when_below_max_size(mocker: pytest_mock.MockerFixture):
    callback = mocker.MagicMock()
    sliding_window = SlidingWindow(max_size=10_000, callback=callback)
    seg = make_segment(text="short", sequence_number=0, sequence_count=100)
    await sliding_window.append(seg)
    callback.assert_not_called()


@pytest.mark.asyncio
async def test_callback_when_max_size_reached(mocker: pytest_mock.MockerFixture):
    callback = mocker.MagicMock()
    sliding_window = SlidingWindow(max_size=1, callback=callback, truncation_percentage=0.0)
    seg = make_segment(text="a", sequence_number=0, sequence_count=100)
    await sliding_window.append(seg)
    callback.assert_called_once()


@pytest.mark.asyncio
async def test_callback_receives_prompt_text():
    received = []

    def callback(text):
        received.append(text)

    sliding_window = SlidingWindow(max_size=1, callback=callback, truncation_percentage=0.0)
    seg = make_segment(speaker="Alice", start=0.0, end=1.0, text="Hi.")
    expected_text = SlidingWindow.form_transcript_chunk(seg)
    await sliding_window.append(seg)
    assert received[0] == expected_text


@pytest.mark.asyncio
async def test_callback_on_last_sequence(mocker: pytest_mock.MockerFixture):
    """Callback fires when the last item in the sequence is drained into the window.
    Items must arrive in order (or all predecessors must have already been drained)
    before the last-sequence trigger fires."""
    callback = mocker.MagicMock()
    sliding_window = SlidingWindow(max_size=10_000, callback=callback)
    # Append items 0-3 first so the window is ready to accept item 4
    for i in range(4):
        await sliding_window.append(make_segment(sequence_number=i, sequence_count=5))
    callback.assert_not_called()
    # Now appending the last item should trigger the callback
    seg = make_segment(sequence_number=4, sequence_count=5)
    await sliding_window.append(seg)
    callback.assert_called_once()


@pytest.mark.asyncio
async def test_no_callback_on_non_last_sequence_below_max_size(mocker: pytest_mock.MockerFixture):
    callback = mocker.MagicMock()
    sliding_window = SlidingWindow(max_size=10_000, callback=callback)
    seg = make_segment(sequence_number=2, sequence_count=5)
    await sliding_window.append(seg)
    callback.assert_not_called()


@pytest.mark.asyncio
async def test_window_truncated_after_callback(mocker: pytest_mock.MockerFixture):
    """After the callback fires, truncation_percentage controls how many items
    are removed from the front of the window.  Build up a window of 4 items
    via the proper append() API, then trigger the callback with a 5th item and
    verify that 50% of the combined 5-item window is removed."""
    callback = mocker.MagicMock()
    sliding_window = SlidingWindow(max_size=1, callback=callback, truncation_percentage=0.5)
    segs = [
        make_segment(text=f"segment {i}", sequence_number=i, sequence_count=100)
        for i in range(5)
    ]
    # Append the first 4 items at a large max_size so no callback fires yet
    sliding_window.max_size = 10_000
    for seg in segs[:4]:
        await sliding_window.append(seg)
    sliding_window.max_size = 1

    initial_count = len(sliding_window.window)  # 4
    # The 5th append exceeds max_size=1 → callback fires, 50% truncated
    await sliding_window.append(segs[4])
    expected_remaining = initial_count + 1 - int((initial_count + 1) * 0.5)
    assert len(sliding_window.window) == expected_remaining


@pytest.mark.asyncio
async def test_truncation_percentage_zero_keeps_all(mocker: pytest_mock.MockerFixture):
    """With truncation_percentage=0 all items remain after callback."""
    callback = mocker.MagicMock()
    sliding_window = SlidingWindow(max_size=1, callback=callback, truncation_percentage=0.0)
    segs = [
        make_segment(text=f"seg {i}", sequence_number=i, sequence_count=100)
        for i in range(3)
    ]
    sliding_window.window.extend(segs)
    await sliding_window.append(make_segment(text="x", sequence_number=3, sequence_count=100))
    # to_lop = int(4 * 0.0) = 0 → nothing removed
    assert len(sliding_window.window) == 4


@pytest.mark.asyncio
async def test_truncation_percentage_one_clears_all(mocker: pytest_mock.MockerFixture):
    """With truncation_percentage=1.0 all items are removed after callback."""
    callback = mocker.MagicMock()
    sliding_window = SlidingWindow(max_size=1, callback=callback, truncation_percentage=1.0)
    segs = [
        make_segment(text=f"seg {i}", sequence_number=i, sequence_count=100)
        for i in range(3)
    ]
    sliding_window.window.extend(segs)
    await sliding_window.append(make_segment(text="x", sequence_number=3, sequence_count=100))
    assert len(sliding_window.window) == 0


@pytest.mark.asyncio
async def test_multiple_callbacks_accumulate(mocker: pytest_mock.MockerFixture):
    """Multiple appends each exceeding max_size each fire the callback."""
    callback = mocker.MagicMock()
    sliding_window = SlidingWindow(max_size=1, callback=callback, truncation_percentage=1.0)
    for i in range(3):
        seg = make_segment(text="x", sequence_number=i, sequence_count=100)
        await sliding_window.append(seg)
    assert callback.call_count == 3


@pytest.mark.asyncio
async def test_append_cleaned_whisper_result_wraps_object(mocker: pytest_mock.MockerFixture):
    """Appending a CleanedWhisperResult stores a CleanedWhisperResultWrapper in the window."""
    sliding_window = SlidingWindow(max_size=10_000, callback=mocker.MagicMock())
    cwr = make_cleaned_whisper_result(segment_count=0, total_segments=10)
    await sliding_window.append(cwr)
    assert len(sliding_window.window) == 1
    assert isinstance(sliding_window.window[0], CleanedWhisperResultWrapper)


@pytest.mark.asyncio
async def test_append_cleaned_whisper_result_prompt_text_contains_cleaned_transcript(
    mocker: pytest_mock.MockerFixture,
):
    """After appending a CleanedWhisperResult the cleaned_transcript text appears
    in prompt_text, confirming that CleanedWhisperResultWrapper.text is set
    correctly from cleaned_result.cleaned_transcript."""
    sliding_window = SlidingWindow(max_size=10_000, callback=mocker.MagicMock())
    cleaned = "This is the cleaned transcript."
    cwr = make_cleaned_whisper_result(
        speaker="Dana",
        start=5.0,
        end=10.0,
        cleaned_transcript=cleaned,
        segment_count=0,
        total_segments=10,
    )
    await sliding_window.append(cwr)
    pt = sliding_window.prompt_text
    assert cleaned in pt
    assert "Dana" in pt
    assert "[5.00-10.00]" in pt


@pytest.mark.asyncio
async def test_append_sequenced_text_stored_directly(mocker: pytest_mock.MockerFixture):
    """Appending a plain SequencedText stores the object without wrapping."""
    sliding_window = SlidingWindow(max_size=10_000, callback=mocker.MagicMock())
    seg = make_segment()
    await sliding_window.append(seg)
    assert sliding_window.window[0] is seg


# ---------------------------------------------------------------------------
# append – async callback tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_callback_is_awaited(mocker: pytest_mock.MockerFixture):
    async_callback = mocker.AsyncMock()
    sliding_window = SlidingWindow(max_size=1, callback=async_callback, truncation_percentage=0.0)
    seg = make_segment(text="trigger", sequence_number=0, sequence_count=100)
    await sliding_window.append(seg)
    async_callback.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_callback_receives_correct_text():
    received = []

    async def async_callback(text):
        received.append(text)

    sliding_window = SlidingWindow(max_size=1, callback=async_callback, truncation_percentage=0.0)
    seg = make_segment(speaker="Eve", start=3.0, end=7.0, text="Async test.")
    expected = SlidingWindow.form_transcript_chunk(seg)
    await sliding_window.append(seg)
    assert received[0] == expected


@pytest.mark.asyncio
async def test_async_callback_not_called_when_below_max(mocker: pytest_mock.MockerFixture):
    async_callback = mocker.AsyncMock()
    sliding_window = SlidingWindow(max_size=10_000, callback=async_callback)
    seg = make_segment(sequence_number=0, sequence_count=100)
    await sliding_window.append(seg)
    async_callback.assert_not_awaited()


@pytest.mark.asyncio
async def test_async_callback_on_last_segment(mocker: pytest_mock.MockerFixture):
    """Async callback fires when the last item in the sequence is drained into
    the window.  Preceding items 0-8 must be appended first so that item 9 can
    be drained."""
    async_callback = mocker.AsyncMock()
    sliding_window = SlidingWindow(max_size=10_000, callback=async_callback)
    for i in range(9):
        await sliding_window.append(make_segment(sequence_number=i, sequence_count=10))
    async_callback.assert_not_awaited()
    seg = make_segment(sequence_number=9, sequence_count=10)
    await sliding_window.append(seg)
    async_callback.assert_awaited_once()


# ---------------------------------------------------------------------------
# Out-of-order / heap buffering tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_out_of_order_simple():
    """Appending sequence 1 before sequence 0 should buffer item 1 in the heap
    until item 0 arrives, at which point both appear in the window in order."""
    received = []

    def callback(text):
        received.append(text)

    # Use a large max_size so the callback doesn't fire, and sequence_count=100
    # so we're never on the last item.
    sliding_window = SlidingWindow(max_size=10_000, callback=callback, truncation_percentage=0.0)

    seg1 = make_segment(speaker="A", text="Second item.", sequence_number=1, sequence_count=100)
    seg0 = make_segment(speaker="B", text="First item.", sequence_number=0, sequence_count=100)

    # Append out of order: 1 first, then 0
    await sliding_window.append(seg1)

    # seq 1 cannot be added to the window yet (waiting for seq 0), so window is empty
    assert len(sliding_window.window) == 0, "seq 1 should be held in heap until seq 0 arrives"

    await sliding_window.append(seg0)

    # Now both should have been drained into the window in the correct order
    assert len(sliding_window.window) == 2, "both items should be in the window after seq 0 arrives"
    assert sliding_window.window[0].sequence_number == 0, "first window item must be seq 0"
    assert sliding_window.window[1].sequence_number == 1, "second window item must be seq 1"
    assert "First item." in sliding_window.prompt_text
    assert "Second item." in sliding_window.prompt_text
    # First item should appear before second item in the prompt text
    assert sliding_window.prompt_text.index("First item.") < sliding_window.prompt_text.index("Second item.")


@pytest.mark.asyncio
async def test_out_of_order_triggers_multiple_callbacks_on_single_append():
    """Scenario: items 1, 2, and 3 arrive before item 0.  Once item 0 is
    appended all four items drain from the heap into the window.  If max_size
    is set so that every individual chunk exceeds it, the callback should fire
    once for each chunk that pushes the window over the size limit – i.e.
    multiple times as a result of that single append call."""

    received = []

    def callback(text):
        received.append(text)

    # Build four segments, each with a known chunk length
    segs = [
        make_segment(speaker="S", start=float(i), end=float(i + 1),
                     text=f"Item {i}.", sequence_number=i, sequence_count=100)
        for i in range(4)
    ]
    chunk_len = len(SlidingWindow.form_transcript_chunk(segs[0]))

    # max_size is just below one chunk so every chunk addition triggers the
    # callback; use truncation_percentage=1.0 so the window is cleared each
    # time, allowing subsequent chunks to trigger the callback again.
    sliding_window = SlidingWindow(max_size=chunk_len - 1, callback=callback, truncation_percentage=1.0)

    # Append 1, 2, 3 first (all buffered in the heap)
    await sliding_window.append(segs[1])
    await sliding_window.append(segs[2])
    await sliding_window.append(segs[3])

    assert len(received) == 0, "callback should not fire while seq 0 is missing"

    # Now append seq 0 – this should drain all four items and trigger the
    # callback multiple times
    await sliding_window.append(segs[0])

    assert len(received) == len(segs), (
        f"expected multiple callback invocations after draining heap, got {len(received)}"
    )


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_single_item_sequence_fires_callback(mocker: pytest_mock.MockerFixture):
    """sequence_number=0, sequence_count=1 → also the last item → callback fires."""
    callback = mocker.MagicMock()
    sliding_window = SlidingWindow(max_size=10_000, callback=callback)
    seg = make_segment(sequence_number=0, sequence_count=1)
    await sliding_window.append(seg)
    callback.assert_called_once()


@pytest.mark.asyncio
async def test_window_grows_across_multiple_appends(mocker: pytest_mock.MockerFixture):
    """Items accumulate in the window when max_size is not exceeded."""
    callback = mocker.MagicMock()
    sliding_window = SlidingWindow(max_size=10_000, callback=callback)
    for i in range(5):
        seg = make_segment(text=f"item {i}", sequence_number=i, sequence_count=100)
        await sliding_window.append(seg)
    assert len(sliding_window.window) == 5
    callback.assert_not_called()


@pytest.mark.asyncio
async def test_prompt_text_contains_all_appended_segments(mocker: pytest_mock.MockerFixture):
    sliding_window = SlidingWindow(max_size=10_000, callback=mocker.MagicMock())
    seg1 = make_segment(speaker="Alice", text="First.", sequence_number=0, sequence_count=10)
    seg2 = make_segment(speaker="Bob", text="Second.", sequence_number=1, sequence_count=10)
    await sliding_window.append(seg1)
    await sliding_window.append(seg2)
    prompt_text = sliding_window.prompt_text
    assert "Alice" in prompt_text
    assert "First." in prompt_text
    assert "Bob" in prompt_text
    assert "Second." in prompt_text


@pytest.mark.asyncio
async def test_callback_fires_at_exact_max_size_boundary():
    """Callback fires when prompt_text length >= max_size."""
    fired = []

    def callback(text):
        fired.append(text)

    seg = make_segment(speaker="A", start=0.0, end=1.0, text="x")
    chunk_len = len(SlidingWindow.form_transcript_chunk(seg))

    # max_size one less than chunk → callback fires on first append
    sliding_window = SlidingWindow(max_size=chunk_len - 1, callback=callback, truncation_percentage=0.0)
    await sliding_window.append(seg)
    assert len(fired) == 1


@pytest.mark.asyncio
async def test_no_callback_when_below_max_size_and_not_last_sequence():
    """No callback when prompt_text < max_size and not on last sequence."""
    fired = []

    def callback(text):
        fired.append(text)

    seg = make_segment(speaker="A", start=0.0, end=1.0, text="x")
    chunk_len = len(SlidingWindow.form_transcript_chunk(seg))

    # max_size one more than chunk → prompt_text < max_size; not last seq
    sliding_window = SlidingWindow(max_size=chunk_len + 1, callback=callback, truncation_percentage=0.0)
    await sliding_window.append(seg)
    assert len(fired) == 0

# from llm, it should probably go in its own test file
def test_tok():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")
    from llm import find_begin_thinking_token, find_end_thinking_token
    tok = find_begin_thinking_token(tokenizer)
    print(tok)
    tok = find_end_thinking_token(tokenizer)
    print(tok)