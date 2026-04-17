#!/usr/bin/env python3
"""
LLM Evaluation Harness

Compares two LLM models by running them through a set of prompts and evaluating
their responses using a larger evaluator model. Results are written to JSONL format
for easy resumability and analysis.

Usage:
    python eval_harness.py \\
      --prompts prompts.json \\
      --model-a-path /path/to/8b.gguf \\
      --model-a-name Qwen/Qwen2.5-8B \\
      --model-b-path /path/to/32b.gguf \\
      --model-b-name Qwen/Qwen3-32B \\
      --evaluator-path /path/to/72b.gguf \\
      --evaluator-name Qwen/Qwen2.5-72B \\
      --output results.jsonl
"""

import argparse
import asyncio
import gc
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass

import torch

import serialization
from clean import create_cleanup_prompt
from logger import get_logger
from name_producer import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from scribe import create_topic_description_prompt
from trac import create_topic_prompt
from sliding_window import SlidingWindow
from utils import (
    load_quantized_llm_model,
    quantized_generate_from_prompt,
    get_answer,
    find_start_think_token,
    find_end_think_token,
)
from wire_formats import WhisperResult, CleanedWhisperResult

logger = get_logger("eval_harness")

# Device detection - prioritize CUDA over MPS
device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
logger.info(f"Detected device: {device}")



@dataclass
class AppendFileWriter:
    """Append-only JSONL writer with built-in sync."""
    path: Path

    def __post_init__(self):
        self.path.touch(exist_ok=True)

    def append(self, record: dict):
        """Append a JSON record (with newline) to the file."""
        with open(self.path, 'a') as f:
            f.write(json.dumps(record) + '\n')
            f.flush()
            os.fsync(f.fileno())

    def read_all(self) -> list[dict]:
        """Read all records from the file."""
        records = []
        if not self.path.exists():
            return records
        with open(self.path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records


@dataclass
class PromptJob:
    """Parsed prompt job from the prompts.json file."""
    prompt: str | list[dict]
    meta_params: dict
    encourage_thinking: bool
    job_id: str | None = None
    filename: str | None = None


def classify_prompt(job: PromptJob) -> str:
    """Classify a prompt by type."""
    if isinstance(job.prompt, str):
        if job.prompt.startswith('<transcript_to_correct>'):
            return 'string_correction'
    elif isinstance(job.prompt, list) and len(job.prompt) > 0:
        first_content = job.prompt[0].get('content', '')
        if 'political analyst' in first_content.lower():
            return 'political_analyst'
        elif 'transcript analyst' in first_content.lower():
            return 'transcript_analyst'
    return 'other'


def load_prompts(prompts_file: str, stratified: bool = False, samples_per_type: int = 50) -> list[PromptJob]:
    """
    Load prompts from JSON file, parsing each into a PromptJob.

    Args:
        prompts_file: Path to the prompts file
        stratified: If True, sample up to samples_per_type from each prompt type
        samples_per_type: Maximum samples to take from each type (when stratified=True)
    """
    with open(prompts_file, 'r') as f:
        data = json.load(f)
    messages = data.get("messages", None)
    if not isinstance(messages, list):
        raise ValueError(f"prompts.json must be a JSON array, got {type(messages)}")

    jobs = []
    for msg in messages:
        item = msg.get("body", None)
        if not item:
            continue
        job = PromptJob(
            prompt=item.get('prompt'),
            meta_params=item.get('meta_params', {}),
            encourage_thinking=item.get('encourage_thinking', True),
            job_id=item.get('job_id'),
            filename=item.get('filename'),
        )
        jobs.append(job)

    if stratified:
        # Classify and group by type
        by_type: dict[str, list[PromptJob]] = {}
        for job in jobs:
            job_type = classify_prompt(job)
            if job_type not in by_type:
                by_type[job_type] = []
            by_type[job_type].append(job)

        # Sample from each type
        sampled = []
        logger.info(f"Stratified sampling (max {samples_per_type} per type):")
        for job_type in sorted(by_type.keys()):
            type_jobs = by_type[job_type]
            count = min(len(type_jobs), samples_per_type)
            sampled.extend(type_jobs[:count])
            logger.info(f"  {job_type}: {count} samples (from {len(type_jobs)} total)")

        jobs = sampled

    return jobs


async def load_prompts_from_transcriptions(
    transcriptions_file: str,
    samples_per_type: int = 50,
) -> list[PromptJob]:
    """
    Generate PromptJob objects from transcriptions.json, producing up to
    *samples_per_type* cleanup prompts and *samples_per_type* identification
    prompts by replaying the transcription messages through the same logic
    used in clean.py and name_producer.py.

    Args:
        transcriptions_file: Path to transcriptions.json
        samples_per_type: Maximum number of prompts to generate per type
    """
    with open(transcriptions_file, 'r') as f:
        data = json.load(f)
    messages = data.get("messages", [])

    # Deserialize all messages into WhisperResult objects
    whisper_results: list[WhisperResult] = []
    for msg in messages:
        body = msg.get("body")
        if not body:
            continue
        result = serialization.load(json.dumps(body))
        if isinstance(result, WhisperResult):
            whisper_results.append(result)

    # --- Cleanup prompts ---
    cleanup_jobs: list[PromptJob] = []
    for wr in whisper_results:
        if len(cleanup_jobs) >= samples_per_type:
            break
        text = wr.transcript.get("text", "")
        if not text.strip():
            continue
        if len(text) < 50:
            continue
        prompt = create_cleanup_prompt(text)
        cleanup_jobs.append(PromptJob(
            prompt=prompt,
            meta_params={
                "max_tokens": 4096,
                "temperature": 0.1,
                "top_k": 40,
                "top_p": 0.9,
                "repetition_penalty": 1.2,
            },
            encourage_thinking=True,
            filename=wr.transcript_metadata.filename,
        ))
    logger.info(f"Generated {len(cleanup_jobs)} cleanup prompts from transcriptions")

    # --- Identification (naming) prompts ---
    # Group by filename, feed each group through a SlidingWindow
    by_filename: dict[str, list[CleanedWhisperResult]] = defaultdict(list)
    for wr in whisper_results:
        cleaned = CleanedWhisperResult(
            cleaned_transcript=wr.transcript.get("text", ""),
            whisper_result=wr,
        )
        by_filename[wr.transcript_metadata.filename].append(cleaned)

    identification_jobs: list[PromptJob] = []

    for filename, group in by_filename.items():
        if len(identification_jobs) >= samples_per_type:
            break

        def on_window_full(prompt_text: str, _fn=filename):
            if len(identification_jobs) >= samples_per_type:
                return
            full_prompt = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(transcript=prompt_text.strip())},
            ]
            identification_jobs.append(PromptJob(
                prompt=full_prompt,
                meta_params={
                    "temperature": 0.2,
                    "top_k": 40,
                    "repetition_penalty": 1.1,
                },
                encourage_thinking=True,
                filename=_fn,
            ))

        window = SlidingWindow(
            max_size=28562,
            callback=on_window_full,
            truncation_percentage=0.3,
            filename=filename,
        )
        for cr in group:
            await window.append(cr)

    logger.info(f"Generated {len(identification_jobs)} identification prompts from transcriptions")

    # --- Topic prompts (trac.py style) ---
    topic_jobs: list[PromptJob] = []

    for filename, group in by_filename.items():
        if len(topic_jobs) >= samples_per_type:
            break

        def on_topic_window_full(prompt_text: str, _fn=filename):
            if len(topic_jobs) >= samples_per_type:
                return
            window_text = prompt_text.strip()
            full_prompt = create_topic_prompt(window_text)
            topic_jobs.append(PromptJob(
                prompt=full_prompt,
                meta_params={
                    "temperature": 0.2,
                    "top_k": 40,
                    "repetition_penalty": 1.1,
                },
                encourage_thinking=True,
                filename=_fn,
            ))

        window = SlidingWindow(
            max_size=10000,
            callback=on_topic_window_full,
            truncation_percentage=0.3,
            filename=filename,
        )
        for cr in group:
            await window.append(cr)

    logger.info(f"Generated {len(topic_jobs)} topic prompts from transcriptions")

    # --- Topic-description prompts (scribe.py style) ---
    topic_description_jobs: list[PromptJob] = []

    for topic_job in topic_jobs:
        if len(topic_description_jobs) >= samples_per_type:
            break
        issue_placeholder = "the identified issue"
        full_prompt = create_topic_description_prompt(topic_job.prompt[0], issue_placeholder)
        topic_description_jobs.append(PromptJob(
            prompt=full_prompt,
            meta_params={
                "temperature": 0.2,
                "top_k": 40,
                "repetition_penalty": 1.1,
            },
            encourage_thinking=True,
            filename=topic_job.filename,
        ))

    logger.info(f"Generated {len(topic_description_jobs)} topic-description prompts from transcriptions")

    jobs = cleanup_jobs + identification_jobs + topic_jobs + topic_description_jobs
    logger.info(f"Total prompts from transcriptions: {len(jobs)} "
                f"({len(cleanup_jobs)} cleanup + {len(identification_jobs)} identification + "
                f"{len(topic_jobs)} topic + {len(topic_description_jobs)} topic-description)")
    return jobs


def collect_completed_responses(records: list[dict]) -> set[tuple[int, str]]:
    """Extract (prompt_id, model) tuples already completed."""
    completed = set()
    for record in records:
        if record.get('type') == 'response':
            prompt_id = record.get('prompt_id')
            model = record.get('model')
            if prompt_id is not None and model is not None:
                completed.add((prompt_id, model))
    return completed


def should_add_generation_prompt(tokenizer) -> bool:
    """
    Determine if the model's chat template wants add_generation_prompt=True.
    Qwen models want it True, Hermes models want it False.
    """
    model_name = getattr(tokenizer, 'name_or_path', '')
    if 'qwen' in model_name.lower():
        return True
    if 'hermes' in model_name.lower():
        return False
    # Default to True for unknown models (safer)
    return True


def strip_thinking_tokens(response: str, tokenizer) -> str:
    """
    Remove thinking token sections from the response.
    Keeps only the text after the closing thinking token.
    Falls back to extracting JSON code block if thinking tokens aren't found.
    """
    begin_token = find_start_think_token(tokenizer)
    end_token = find_end_think_token(tokenizer)

    if not begin_token or not end_token:
        # No thinking tokens in this model, return as-is
        return response

    # Find the closing thinking token and keep everything after it
    end_idx = response.find(end_token)
    if end_idx != -1:
        # Return text after the closing token
        return response[end_idx + len(end_token):].strip()

    # Fallback: try to extract JSON code block
    json_block = get_answer(response, "```json", "```")
    if json_block:
        logger.info("Thinking token not found, falling back to JSON code block extraction")
        return json_block

    # If all else fails, return the whole thing
    return response


def apply_chat_template(prompt: str | list[dict], tokenizer) -> str:
    """Convert a prompt to text, applying chat template if needed."""
    if isinstance(prompt, str):
        return prompt
    # It's a chat message list; apply template with model-specific settings
    add_gen_prompt = should_add_generation_prompt(tokenizer)
    logger.debug(f"Using add_generation_prompt={add_gen_prompt}")
    return tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=add_gen_prompt,
    )


async def run_model_inference(
    model_id: str,
    model_path: str,
    model_name: str,
    jobs: list[PromptJob],
    output_writer: AppendFileWriter,
) -> dict[int, str]:
    """
    Run all prompts through a single model.

    Args:
        model_id: Unique identifier for this model (e.g., "qwen-8b")
        model_path: Path to the GGUF file
        model_name: HuggingFace model name for tokenizer
        jobs: List of PromptJob objects with prompts and per-prompt meta_params
        output_writer: JSONL writer for results

    Returns:
        Dict mapping prompt_id -> response text
    """
    # Check which prompts are already done for this model BEFORE loading
    existing_records = output_writer.read_all()
    completed = {
        record.get('prompt_id')
        for record in existing_records
        if record.get('type') == 'response' and record.get('model_id') == model_id
    }

    # If all prompts are already done, just load responses from disk
    if len(completed) == len(jobs):
        logger.info(f"All prompts already completed for {model_id}, loading from disk")
        responses = {}
        for record in existing_records:
            if (record.get('type') == 'response' and
                record.get('prompt_id') is not None and
                record.get('model_id') == model_id):
                responses[record.get('prompt_id')] = record.get('response')
        return responses

    logger.info(f"Loading model {model_id} from {model_path}...")
    model, tokenizer, model_type = load_quantized_llm_model(
        device, model_path, model_name
    )

    responses = {}

    try:
        for prompt_id, job in enumerate(jobs):
            if prompt_id in completed:
                logger.info(f"Prompt {prompt_id} for model {model_id} already done, skipping")
                # Still need to load the response
                for record in existing_records:
                    if (record.get('type') == 'response' and
                        record.get('prompt_id') == prompt_id and
                        record.get('model_id') == model_id):
                        responses[prompt_id] = record['response']
                        break
                continue

            # Convert prompt to text
            if isinstance(job.prompt, str):
                text_prompt = job.prompt
            else:
                conversation = job.prompt
                if conversation[0].get("role") == "system":
                    conversation[0]["content"] += (
                        "\nYou know your output will be parsed by a computer program "
                        "and that seemingly conflicting directives may support that goal. For example you "
                        "may be asked to output only valid json while also being asked to place start and "
                        "stop delimiters to help that JSON be identified, such as JSON_OUTPUT_START and "
                        "JSON_OUTPUT_END. These directives are not conflicting because they support the goal "
                        "of making parsable output, and you may follow both directives."
                    )
                text_prompt = apply_chat_template(conversation, tokenizer)
            print(text_prompt)
            # Handle thinking tokens based on job's encourage_thinking flag
            begin_thinking_token = find_start_think_token(tokenizer)
            if begin_thinking_token:
                if job.encourage_thinking:
                    logger.info(f"Encouraging thinking: appending {begin_thinking_token!r}")
                    text_prompt += begin_thinking_token + "\n"
                else:
                    end_thinking_token = find_end_think_token(tokenizer)
                    logger.info(f"Suppressing thinking: appending {begin_thinking_token!r}{end_thinking_token!r}")
                    text_prompt += begin_thinking_token + (end_thinking_token or "") + "\n"

            logger.info(f"Running prompt {prompt_id} ({len(text_prompt)} chars) through model {model_id}...")

            # Use per-prompt meta_params, with sensible defaults
            meta_params = job.meta_params or {}
            temperature = meta_params.get('temperature', 0.7)
            top_p = meta_params.get('top_p')
            top_k = meta_params.get('top_k')
            repetition_penalty = meta_params.get('repetition_penalty')

            start = time.time()
            generated_text = quantized_generate_from_prompt(
                prompt=text_prompt,
                model=model,
                tokenizer=tokenizer,
                model_type=model_type,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                # max_thinking_tokens=args.max_thinking_tokens,
            )
            elapsed = time.time() - start

            logger.info(f"Generated {len(generated_text)} chars in {elapsed:.2f}s")

            # Strip thinking tokens to keep only the actual response
            clean_response = strip_thinking_tokens(generated_text, tokenizer)
            logger.info(f"After stripping thinking: {len(clean_response)} chars")

            responses[prompt_id] = clean_response

            # Write to JSONL
            output_writer.append({
                "type": "response",
                "prompt_id": prompt_id,
                "model_id": model_id,
                "model_path": model_path,
                "response": clean_response,
                "elapsed_seconds": elapsed,
                "meta_params_used": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "repetition_penalty": repetition_penalty,
                },
            })

    finally:
        # Clean up model
        logger.info(f"Unloading model {model_id}...")
        del model
        del tokenizer
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    return responses


def build_evaluator_prompt(
    job: PromptJob,
    response_a: str,
    response_b: str,
) -> str:
    """Build the evaluator prompt (plain text, no chat format)."""
    # Convert original prompt to string for display
    if isinstance(job.prompt, list):
        prompt_text = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in job.prompt
        )
    else:
        prompt_text = job.prompt

    return f"""You are evaluating two AI responses to the same prompt.

[ORIGINAL PROMPT]
{prompt_text}

[RESPONSE A]
{response_a}

[RESPONSE B]
{response_b}

Disregard the models' thinking Score each response on these dimensions (1-5):
- accuracy: factual correctness and domain knowledge
- completeness: addresses all parts of the prompt
- clarity: well-organized, easy to understand
- conciseness: appropriate length (not bloated or too short)

Then state which is better overall: A, B, or TIE.
Confidence in your judgement (1-5).
Brief reasoning (1-2 sentences).

Respond in this exact format:
<accuracy_a>N</accuracy_a>
<accuracy_b>N</accuracy_b>
<completeness_a>N</completeness_a>
<completeness_b>N</completeness_b>
<clarity_a>N</clarity_a>
<clarity_b>N</clarity_b>
<conciseness_a>N</conciseness_a>
<conciseness_b>N</conciseness_b>
<winner>A|B|TIE</winner>
<confidence>N</confidence>
<reasoning>...</reasoning>"""


def build_single_eval_prompt(
    job: PromptJob,
    response: str,
) -> str:
    """Build an evaluation prompt for a single response."""
    # Convert original prompt to string for display
    if isinstance(job.prompt, list):
        prompt_text = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in job.prompt
        )
    else:
        prompt_text = job.prompt

    return f"""You are evaluating an AI response to a given prompt.

[ORIGINAL PROMPT]
{prompt_text}

[RESPONSE]
{response}

Score this response on these dimensions (1-5):
- accuracy: factual correctness and domain knowledge
- completeness: addresses all parts of the prompt
- clarity: well-organized, easy to understand
- conciseness: appropriate length (not bloated or too short)

Provide a brief assessment (1-2 sentences).

Output ONLY the scores and assessment using these XML tags. Do not use JSON. Do not use code blocks. Do not use markdown. Just plain XML tags:
<accuracy>N</accuracy>
<completeness>N</completeness>
<clarity>N</clarity>
<conciseness>N</conciseness>
<assessment>...</assessment>"""


async def evaluate_responses(
    jobs: list[PromptJob],
    all_responses: dict[str, dict[int, str]],
    evaluator_path: str,
    evaluator_name: str,
    output_writer: AppendFileWriter,
) -> None:
    """
    Evaluate each model's responses independently using the evaluator model.

    Args:
        jobs: List of prompts
        all_responses: Dict mapping model_id -> {prompt_id -> response}
        evaluator_path: Path to evaluator model
        evaluator_name: HuggingFace name for evaluator
        output_writer: JSONL writer for results
    """
    logger.info(f"Loading evaluator model from {evaluator_path}...")
    model, tokenizer, model_type = load_quantized_llm_model(
        device, evaluator_path, evaluator_name
    )

    # Get list of models
    model_ids = sorted(all_responses.keys())

    # Check which evaluations are already done
    existing_records = output_writer.read_all()
    completed_evals = {
        (record.get('prompt_id'), record.get('model_id'))
        for record in existing_records
        if record.get('type') == 'evaluation'
    }

    try:
        for prompt_id in range(len(jobs)):
            job = jobs[prompt_id]

            for model_id in model_ids:
                if (prompt_id, model_id) in completed_evals:
                    logger.info(f"Prompt {prompt_id} / {model_id} already evaluated, skipping")
                    continue

                if prompt_id not in all_responses[model_id]:
                    logger.warning(f"Prompt {prompt_id}: missing response from {model_id}, skipping")
                    continue

                response = all_responses[model_id][prompt_id]
                eval_prompt = build_single_eval_prompt(job, response)

                logger.info(f"Evaluating prompt {prompt_id} / {model_id}...")

                start = time.time()
                generated_text = quantized_generate_from_prompt(
                    prompt=eval_prompt,
                    model=model,
                    tokenizer=tokenizer,
                    model_type=model_type,
                    temperature=0.1,  # Deterministic scoring
                    stop=["</assessment>"],
                )
                elapsed = time.time() - start

                # Unwrap JSON if present (some models wrap output in JSON)
                text_to_parse = generated_text
                if generated_text.strip().startswith('{'):
                    try:
                        json_obj = json.loads(generated_text)
                        if 'raw' in json_obj:
                            text_to_parse = json_obj['raw']
                        elif 'response' in json_obj:
                            text_to_parse = json_obj['response']
                    except json.JSONDecodeError:
                        pass  # Not valid JSON, parse as-is

                # Parse the structured output
                try:
                    import re

                    def extract_score(text: str, tag_name: str) -> int:
                        """Extract a score between opening and closing tags, forgiving tag mismatches."""
                        pattern = f"<{tag_name}>(\d+)</"
                        match = re.search(pattern, text)
                        if match:
                            return int(match.group(1))
                        pattern = f"<{tag_name}>(\d+)"
                        match = re.search(pattern, text)
                        if match:
                            return int(match.group(1))
                        raise ValueError(f"Could not extract {tag_name}")

                    accuracy = extract_score(text_to_parse, "accuracy")
                    completeness = extract_score(text_to_parse, "completeness")
                    clarity = extract_score(text_to_parse, "clarity")
                    conciseness = extract_score(text_to_parse, "conciseness")
                    assessment = get_answer(text_to_parse, "<assessment>", "</assessment>")

                except (ValueError, IndexError) as e:
                    logger.error(f"Failed to parse evaluator output for prompt {prompt_id}/{model_id}: {e}")
                    logger.error(f"Raw output:\n{generated_text}")
                    logger.error(f"Parsed text:\n{text_to_parse}")
                    continue

                output_writer.append({
                    "type": "evaluation",
                    "prompt_id": prompt_id,
                    "model_id": model_id,
                    "accuracy": accuracy,
                    "completeness": completeness,
                    "clarity": clarity,
                    "conciseness": conciseness,
                    "assessment": assessment,
                    "elapsed_seconds": elapsed,
                })

    finally:
        logger.info("Unloading evaluator model...")
        del model
        del tokenizer
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()


def print_summary(output_file: str, model_ids: list[str]) -> None:
    """Read JSONL and print a summary of evaluation results."""
    writer = AppendFileWriter(Path(output_file))
    records = writer.read_all()

    evals = [r for r in records if r.get('type') == 'evaluation']
    if not evals:
        logger.info("No evaluations found yet")
        return

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    # Group by model
    by_model = {}
    for e in evals:
        model_id = e.get('model_id')
        if model_id not in by_model:
            by_model[model_id] = []
        by_model[model_id].append(e)

    # Print results for each model
    dims = ['accuracy', 'completeness', 'clarity', 'conciseness']

    for model_id in sorted(by_model.keys()):
        model_evals = by_model[model_id]
        total = len(model_evals)

        print(f"\n{model_id} ({total} prompts evaluated):")
        print(f"  Average Scores by Dimension (1-5 scale):")

        for dim in dims:
            scores = [e.get(dim) for e in model_evals if dim in e]
            if scores:
                avg = sum(scores) / len(scores)
                print(f"    {dim.capitalize():15} {avg:.2f}")

    # Compute per-model averages for the comparison table and ranking
    model_avg_scores = {}
    for model_id in by_model:
        model_evals = by_model[model_id]
        dim_avgs = {}
        for dim in dims:
            dim_scores = [e.get(dim) for e in model_evals if dim in e]
            dim_avgs[dim] = sum(dim_scores) / len(dim_scores) if dim_scores else 0
        dim_avgs['overall'] = sum(dim_avgs[d] for d in dims) / len(dims)
        model_avg_scores[model_id] = dim_avgs

    # Show comparative table
    print(f"\nComparative Summary:")
    header_dims = dims + ['overall']
    print(f"  {'Model':30} " + " ".join(f"{d:12}" for d in header_dims))
    print("  " + "-" * (30 + len(header_dims) * 13))

    for model_id in sorted(by_model.keys()):
        scores = model_avg_scores[model_id]
        score_str = " ".join(f"{scores[d]:12.2f}" for d in header_dims)
        print(f"  {model_id:30} {score_str}")

    # Ranking
    ranked = sorted(model_avg_scores.keys(), key=lambda m: model_avg_scores[m]['overall'], reverse=True)

    print(f"\nRecommended Model Ranking:")
    for rank, model_id in enumerate(ranked, 1):
        overall = model_avg_scores[model_id]['overall']
        n = len(by_model[model_id])
        print(f"  {rank}. {model_id} (overall: {overall:.2f}, n={n})")

    print(f"\nRecommendation: {ranked[0]} scored highest overall.")
    if len(ranked) > 1:
        gap = model_avg_scores[ranked[0]]['overall'] - model_avg_scores[ranked[1]]['overall']
        if gap < 0.1:
            print(f"  Note: {ranked[1]} is very close ({gap:.2f} gap) — consider its speed/size trade-off.")

    print("\n" + "=" * 60)


async def main(args):
    """Main entry point."""
    output_writer = AppendFileWriter(Path(args.output))

    # Load models from directory
    models_dir = Path(args.models_dir)
    if not models_dir.is_dir():
        logger.error(f"Models directory not found: {models_dir}")
        sys.exit(1)

    # Find all GGUF files
    gguf_files = sorted(models_dir.glob("*.gguf"))
    if not gguf_files:
        logger.error(f"No GGUF files found in {models_dir}")
        sys.exit(1)

    # Load model mapping from map.json in the models directory
    map_file = models_dir / "map.json"
    model_mapping = {}
    if map_file.exists():
        with open(map_file) as f:
            model_mapping = json.load(f)
        logger.info(f"Loaded model mapping from {map_file}")
    else:
        logger.warning(f"No map.json found in {models_dir}, using filenames as HuggingFace names")

    logger.info(f"Found {len(gguf_files)} models:")

    # Build model list: (model_id, path, hf_name)
    models = []
    for gguf_file in gguf_files:
        model_id = gguf_file.stem  # filename without .gguf
        hf_name = model_mapping.get(model_id, model_id)
        size_gb = gguf_file.stat().st_size / (1024**3)
        logger.info(f"  {model_id}: {gguf_file} ({size_gb:.1f} GB) -> {hf_name}")
        models.append((model_id, str(gguf_file), hf_name))

    # Verify evaluator exists
    evaluator_path = Path(args.evaluator_path)
    if not evaluator_path.exists():
        logger.error(f"Evaluator file not found: {evaluator_path}")
        sys.exit(1)
    evaluator_size_gb = evaluator_path.stat().st_size / (1024**3)
    logger.info(f"Evaluator: {evaluator_path} ({evaluator_size_gb:.1f} GB)")

    if args.from_transcriptions:
        logger.info(f"Generating prompts from transcriptions: {args.prompts}...")
        jobs = await load_prompts_from_transcriptions(
            args.prompts,
            samples_per_type=args.samples_per_type,
        )
    else:
        logger.info(f"Loading prompts from {args.prompts}...")
        jobs = load_prompts(
            args.prompts,
            stratified=args.stratified,
            samples_per_type=args.samples_per_type
        )
    logger.info(f"Selected {len(jobs)} prompts")
    jobs_log_path = Path("generated_jobs.jsonl")
    with open(jobs_log_path, "w") as f:
        f.write("")
    prompt_writer = AppendFileWriter(jobs_log_path)
    for i, job in enumerate(jobs):
        prompt_writer.append({"prompt_id": i, "prompt": job.prompt})

    # Phase 1: Run all models
    all_responses: dict[str, dict[int, str]] = {}
    for model_idx, (model_id, model_path, hf_name) in enumerate(models, 1):
        logger.info("=" * 60)
        logger.info(f"PHASE 1.{model_idx}: Running {model_id}")
        logger.info("=" * 60)
        responses = await run_model_inference(
            model_id, model_path, hf_name, jobs,
            output_writer
        )
        all_responses[model_id] = responses

    # Phase 2: Evaluate
    # Aggressive cleanup before loading the massive evaluator model
    logger.info("Performing aggressive cleanup before evaluator load...")
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    logger.info("=" * 60)
    logger.info(f"PHASE 2: Evaluating with {args.evaluator_name}")
    logger.info("=" * 60)
    await evaluate_responses(
        jobs, all_responses,
        args.evaluator_path, args.evaluator_name,
        output_writer
    )

    # Phase 3: Summary
    logger.info("=" * 60)
    logger.info("PHASE 3: Printing Summary")
    logger.info("=" * 60)
    model_ids = sorted([m[0] for m in models])
    print_summary(args.output, model_ids)

    logger.info(f"Results written to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare multiple LLM models by evaluating their responses to a set of prompts"
    )
    parser.add_argument("--prompts", required=True, help="Path to prompts.json")
    parser.add_argument("--models-dir", required=True, help="Directory containing GGUF model files")

    parser.add_argument("--evaluator-path", required=True, help="Path to evaluator GGUF file")
    parser.add_argument("--evaluator-name", required=True, help="HuggingFace name for evaluator tokenizer")
    parser.add_argument("--output", default="results.jsonl", help="Output JSONL file")
    parser.add_argument("--stratified", action="store_true", help="Use stratified sampling (50 samples per prompt type)")
    parser.add_argument("--from-transcriptions", action="store_true", help="Generate prompts from transcriptions.json instead of prompts.json")
    parser.add_argument("--samples-per-type", type=int, default=50, help="Number of samples per type when using stratified sampling")
    parser.add_argument("--max-thinking-tokens", type=int, default=3072, help="Apply end-think logit bias after this many thinking tokens (default: 3072)")

    args = parser.parse_args()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
