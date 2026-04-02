import heapq
import inspect
from typing import Protocol, Callable

from logger import get_logger
from wire_formats import CleanedWhisperResult

logger = get_logger("name_producer")

class SequencedText(Protocol):
    speaker: str
    start_seconds: float
    end_seconds: float
    text: str
    sequence_number: int
    sequence_count: int
    truncation_percentage: float

class CleanedWhisperResultWrapper(SequencedText):
    def __init__(self, cleaned_result: CleanedWhisperResult):
        self.speaker = cleaned_result.whisper_result.speaker
        self.start_seconds = cleaned_result.whisper_result.timings.start
        self.end_seconds = cleaned_result.whisper_result.timings.end
        self.text = cleaned_result.cleaned_transcript
        self.sequence_number = cleaned_result.whisper_result.segment_count
        self.sequence_count = cleaned_result.whisper_result.total_segments
        self.cleaned_whisper_result = cleaned_result

class SlidingWindow:
    def __init__(self, max_size: int, callback: Callable, truncation_percentage: float = 0.3, filename: str | None = None):
        self.max_size = max_size
        self.callback = callback
        self.heap: list[SequencedText] = []
        self.window: list[SequencedText] = []
        if truncation_percentage < 0 or truncation_percentage > 1:
            raise ValueError("truncation percentage must be a float value from 0-1")
        self.truncation_percentage = truncation_percentage
        # Tracks the sequence_number of the next item we expect to drain from
        # the heap into the window.  Stored as instance state so that it
        # survives across append() calls even when the window is empty (e.g.
        # after a full truncation).
        self.next_sequence_number: int = 0
        self.filename: str | None = filename

    async def append(self, sequenced_text: SequencedText | CleanedWhisperResult):
        if isinstance(sequenced_text, CleanedWhisperResult):
            item = CleanedWhisperResultWrapper(sequenced_text)
        else:
            item = sequenced_text
        heapq.heappush(self.heap, (item.sequence_number, item))
        seq_nums = sorted(x[0] for x in self.heap)
        filename = self.filename + " " if self.filename else ""
        logger.info(f"{filename}{seq_nums}")
        logger.info(f"{filename}{self.next_sequence_number=}")
        # Sync next_sequence_number with the window tail if the window is
        # non-empty.  This handles cases where items were inserted into
        # self.window directly (e.g. test setup) or after a partial truncation
        # that left some items behind.  When the window is empty because all
        # items were truncated we keep the persisted value so the drain loop
        # continues from the correct position rather than restarting from 0.
        if self.window:
            self.next_sequence_number = self.window[-1].sequence_number + 1
        while True:
            try:
                sequence_number, heap_item = heapq.heappop(self.heap)
            except IndexError:
                break
            if heap_item.sequence_number == self.next_sequence_number:
                self.window.append(heap_item)
                self.next_sequence_number += 1  # advance before callback so truncation recalc is correct
                prompt_text = self.prompt_text
                logger.info(f"prompt text length {len(prompt_text)}/{self.max_size}")
                if len(prompt_text) >= self.max_size or heap_item.sequence_number == heap_item.sequence_count - 1:
                    # Fire callback inside the drain loop so that a single append
                    # that releases many buffered items can trigger the callback
                    # (and truncate) multiple times.
                    if inspect.iscoroutinefunction(self.callback):
                        await self.callback(prompt_text)
                    else:
                        self.callback(prompt_text)
                    to_lop = int(len(self.window) * self.truncation_percentage)
                    self.window = self.window[to_lop:]
                    # After truncation keep next_sequence_number consistent with
                    # the surviving tail of the window (if any).
                    if self.window:
                        self.next_sequence_number = self.window[-1].sequence_number + 1
                    # If the window was fully cleared we intentionally leave
                    # next_sequence_number pointing past the item we just processed
                    # so subsequent appends continue from the right position.
            else:
                heapq.heappush(self.heap, (sequence_number, heap_item))  # return the non-matching item
                break

    @staticmethod
    def form_transcript_chunk(sequenced_text: SequencedText) -> str:
        return f"[{sequenced_text.start_seconds:.2f}-{sequenced_text.end_seconds:.2f}] {sequenced_text.speaker.strip()}:\n{sequenced_text.text.strip()}\n\n"

    @property
    def prompt_text(self) -> str:
        """Format the text in a way that is suitable for prompting an LLM"""
        return "".join(self.form_transcript_chunk(r) for r in self.window)

    def __len__(self):
        return len(self.prompt_text)


