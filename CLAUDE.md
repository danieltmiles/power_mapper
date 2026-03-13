# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Power Mapper is an AI pipeline that ingests unstructured data (YouTube videos of city council meetings, news articles, etc.) and produces a "power map" — a graph of which people care about which issues, their stances, and their connections to other people and organizations.

## Running Tests

```bash
# Run all unit tests
pytest

# Run a specific test file
pytest sliding_window_test.py

# Run a specific test
pytest sliding_window_test.py::test_out_of_order_simple

# Run integration tests (requires running Redis and RabbitMQ)
cd integration_tests && behave
```

## Running Services

Each service is a standalone Python script that reads a JSON config file:

```bash
python llm.py config.json
python mint.py config.json
python dads.py config.json
python slice.py config.json
python clean.py config.json
python gate.py config.json
python name_producer.py config.json
```

Config file format:
```json
{
    "work_queue": "queue_name",
    "host": "rabbitmq_host",
    "port": 5671,
    "username": "user",
    "password": "pass",
    "model_path": "/path/to/model.gguf",
    "hf_model_name": "Qwen/Qwen3-32B",
    "redis": {
        "host": "redis_host",
        "port": 6380,
        "ssl": true
    }
}
```

## Architecture

The pipeline processes audio through these stages, communicating entirely via **RabbitMQ queues**:

1. **MINT** (`mint.py`) — infers metadata (date, title) from audio filenames via LLM
2. **DADS** (`dads.py`) — speaker diarization; produces speaker-labeled audio segments
3. **SLICE** (`slice.py`) — splits diarization results into segments for Whisper
4. **Whisper** (`whisper_transcription.py`) — transcribes audio segments
5. **CLEAN** (`clean.py`) — LLM-powered transcript cleanup
6. **GATE** (`gate.py`) — quality check; rejects bad transcripts back to CLEAN
7. **NAME** (`name_producer.py` + `name_consumer.py`) — identifies speaker names from transcript context using a sliding window
8. **Parallel extraction** (`get_topics_producer.py`, and similar) — extracts topics, people, and organizations from transcripts

The **LLM worker** (`llm.py`) is a shared, dumb inference service. Producers send `LLMPromptJob` messages (with a `reply_to` queue), and the LLM worker sends back `LLMPromptResponse` messages. All LLM business logic lives in the producers, not in `llm.py`.

## Key Modules

- **`wire_formats.py`** — All inter-service message types as dataclasses (`LLMPromptJob`, `LLMPromptResponse`, `WhisperResult`, `CleanedWhisperResult`, `DiarizationResponse`, etc.). All implement `asdict()`.
- **`serialization.py`** — JSON serialization with embedded `__class__` metadata for polymorphic deserialization. Use `serialization.dumps(obj)` and `serialization.load(json_str)`.
- **`cached_iterator.py`** — `CachedMessageIterator`: an async context manager that wraps RabbitMQ queue consumption with Redis-backed crash recovery. Messages are backed up to Redis on receipt and removed after `mark_processed()`. Services use `async with iterator.processing(message):` to ensure cleanup.
- **`sliding_window.py`** — `SlidingWindow`: buffers out-of-order transcript segments (using a heap), re-orders them, and fires a callback when the accumulated text exceeds `max_size` or the last segment arrives. Used by NAME service to bundle context for speaker identification.
- **`utils.py`** — Shared utilities: `load_config()`, `dial_rabbit_from_config()`, `dial_redis_from_config()`, `load_quantized_llm_model()` (llama-cpp GGUF), `quantized_generate_from_prompt()`, `get_answer()` (parses delimited text from LLM output), `publish_event()` (fire-and-forget observability to `events` queue).

## Infrastructure

- **RabbitMQ** (with TLS on port 5671) — all inter-service messaging
- **Redis** (with TLS) — message backup for crash recovery; each service uses a key prefix like `backup:service_name`
- **llama-cpp-python** with GGUF models (default: `Qwen3-32B-Q4_K_M.gguf`) — LLM inference, supporting CUDA/MPS/CPU
- HuggingFace token required for tokenizer loading; stored in `hf_token.txt`
- TLS certificates expected as `server_certificate.pem` in the working directory

## Patterns

- Services follow producer/consumer splits where needed (e.g., `name_producer.py` / `name_consumer.py`)
- LLM prompts support both plain strings and chat-format lists of dicts; `llm.py` applies the chat template automatically
- `encourage_thinking=True` on `LLMPromptJob` appends `<think>` to prompt; `False` appends `<think></think>` to suppress CoT
- `get_answer(text, start_delim, end_delim)` in `utils.py` is the standard way to extract structured output from LLM responses

## Running Code

Always activate the virtual environment before running any Python command:

```bash
source ./venv/bin/activate
```

If a command is not found after activation, do not attempt to install it. Ask the user to locate it instead — PATH may not resolve correctly in all environments (e.g. PyCharm terminals).

## Code Style

### Clarity over cleverness

Prefer code that reads like plain prose. Variable and function names should communicate intent without requiring a reader to look up the definition. Spell things out: `callback` not `cb`, `sliding_window` not `sw`, `async_callback` not `async_cb`. Short abbreviations are only acceptable inside a tight loop where the meaning is obvious from immediate context.

### Simplicity of structure

Prefer plain data (dataclasses) and standalone functions over classes that bundle data and logic together. A module-level function that takes a dataclass as an argument is simpler than a method operating on `self`. Prefer pure functions — output depends only on input, no side effects — over methods that read or mutate object state.

Introduce a class when it genuinely pays for itself: when naming the concept as an object clarifies the design and makes it easier to reason about, not just because it is a familiar pattern. Stateful classes (mutable state + data + logic combined) carry the highest burden of justification. When in doubt, start with a function and promote to a class only when the argument list becomes unwieldy *and* the arguments naturally belong together as a named concept.

Avoid unnecessary indirection. If a helper only saves one line at the call site, it is not worth having.

### Testing

Do not create pytest fixtures for things that can be expressed in fewer than three lines — inline them directly in the test function. Do not create factory fixtures (fixtures that return factory functions); construct objects directly in each test. Helpers and abstractions are only justified when their body contains three or more meaningful lines of logic.