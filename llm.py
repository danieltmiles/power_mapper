"""
Generic LLM Worker for RabbitMQ

This is a "dumb" worker that simply takes prompts and returns generated text.
All business logic (parsing, retries, etc.) is handled by the producer.

Usage:
    python get_topics_consumer.py get_topics_config.json

Configuration file format (JSON):
{
    "work_queue": "llm/generic",
    "model_path": "/path/to/model",
    "host": "localhost",
    "port": 5672,
    "username": "guest",
    "password": "guest"
}

Message format (JSON):
{
    "job_id": "uuid",
    "reply_to": "queue-name",
    "request_id": "optional-request-id",
    "prompt": "The prompt text to generate from",
    "params": {
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "repetition_penalty": 1.1
    }
}

Response format (JSON):
{
    "job_id": "uuid",
    "request_id": "optional-request-id",
    "status": "success" | "error",
    "generated_text": "...",
    "error": "error message if status is error"
}
"""
import json
import os
import signal
import threading
import time

import aio_pika
import argparse
import torch
import asyncio

from aio_pika.abc import AbstractIncomingMessage
from aiormq import ChannelInvalidStateError, ChannelClosed, AMQPError

import serialization
import wire_formats
from cached_iterator import CachedMessageIterator
from logger import get_logger
from utils import load_config, load_quantized_llm_model, quantized_generate_from_prompt, dial_rabbit_from_config, dial_redis_from_config, find_start_think_token, find_end_think_token
from wire_formats import LLMPromptJob, Metaparams

logger = get_logger("llm")


def watch_parent(ppid: int, interval: float = 5.0):
    """Exit if our parent process disappears (e.g. menubar app crashed).

    Runs in a daemon thread. Polls every `interval` seconds by sending
    signal 0 to the original parent PID — harmless, but raises OSError
    if the process no longer exists.
    """
    while True:
        time.sleep(interval)
        try:
            os.kill(ppid, 0)
        except OSError:
            logger.warning("Parent process %d is gone — exiting", ppid)
            os._exit(0)


# Device detection - prioritize CUDA over MPS
device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
model_type = None
logger.info(f"Detected device: {device}")


def generate_text(prompt: str, model, tokenizer, params: dict) -> str:
    """
    Generate text from a prompt using the loaded LLM.
    
    Args:
        prompt: The prompt text to generate from
        model: The loaded LLM model
        tokenizer: The tokenizer for the model (None for llama-cpp)
        params: Generation parameters (max_tokens, temperature, top_p, top_k, etc.)
    
    Returns:
        str: Generated text
    """
    logger.info(json.dumps(params, indent=4))
    # Extract parameters with defaults
    max_tokens = params.get('max_tokens', 4096)
    temperature = params.get('temperature', 0.7)
    top_p = params.get('top_p', None)
    top_k = params.get('top_k', 40)
    repetition_penalty = params.get('repetition_penalty', 1.1)
    stop = params.get("stop")

    # Use the utility function which handles both MLX and GGUF models
    start = time.time()
    generated_text = quantized_generate_from_prompt(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        model_type=model_type,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        stop=stop,
    )
    end = time.time()
    logger.info(f"Generation completed, {len(generated_text)} characters in {end - start:.2f} seconds")
    return generated_text


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


async def process_message(message: AbstractIncomingMessage, model, tokenizer, config: dict):
    """
    Process an LLM generation job message from RabbitMQ.
    
    Expected message format:
    {
        'job_id': str,
        'reply_to': str (queue name for response),
        'request_id': str (optional - for tracking multiple requests in a job),
        'prompt': str,
        'params': {
            'max_tokens': int,
            'temperature': float,
            'top_p': float,
            'top_k': int,
            'repetition_penalty': float
        }
    }
    """
    try:
        job_desc: LLMPromptJob = serialization.load(message.body.decode(), cls=LLMPromptJob)
    except TypeError as type_error:
        logger.error(str(type_error))
        body = json.loads(message.body.decode())
        job_desc = LLMPromptJob(
            job_id=body.get("job_id"),
            filename=body.get("filename", "") or body.get("transcript_file_name"),
            reply_to=body.get("reply_to"),
            prompt=body.get("prompt"),
            request_id=body.get("request_id"),
        )
        params = body.get("params", {})
        if params:
            params_kwargs = {}
            if max_tokens := params.get("max_tokens"):
                params_kwargs["max_tokens"] = max_tokens
            if temperature := params.get("temperature"):
                params_kwargs["temperature"] = temperature
            if top_p := params.get("top_p"):
                params_kwargs["top_p"] = top_p
            if top_k := params.get("top_k"):
                params_kwargs["top_k"] = top_k
            if repetition_penalty := params.get("repetition_penalty"):
                params_kwargs["repetition_penalty"] = repetition_penalty
            job_desc.meta_params = Metaparams(**params_kwargs)
    
    logger.info(f"Received job {job_desc.job_id}, request {job_desc.request_id}")

    if isinstance(job_desc.prompt, list):
        logger.info(f"Prompt is a conversation with {len(job_desc.prompt)} messages")
        add_gen_prompt = should_add_generation_prompt(tokenizer)
        logger.info(f"Using add_generation_prompt={add_gen_prompt}")
        text_prompt = tokenizer.apply_chat_template(
            job_desc.prompt,
            tokenize=False,
            add_generation_prompt=add_gen_prompt,
        )
        begin_thinking_token = find_start_think_token(tokenizer)
        if begin_thinking_token:
            if job_desc.encourage_thinking:
                logger.info(f"Encouraging thinking: appending {begin_thinking_token!r}")
                text_prompt += begin_thinking_token + "\n"
            else:
                end_thinking_token = find_end_think_token(tokenizer)
                logger.info(f"Suppressing thinking: appending {begin_thinking_token!r}{end_thinking_token!r}")
                text_prompt += begin_thinking_token + (end_thinking_token or "") + "\n"
    else:
        logger.info(f"Prompt length: {len(job_desc.prompt)} chars")
        text_prompt = job_desc.prompt

    # Run text generation in thread pool to prevent blocking
    logger.info(f"<prompt>\n{text_prompt}\n</prompt>")
    loop = asyncio.get_event_loop()
    start = time.time()
    generated_text = await loop.run_in_executor(
        None,
        generate_text,
        text_prompt,
        model,
        tokenizer,
        job_desc.meta_params.asdict(),
    )
    end = time.time()
    logger.info(f"Generated {len(generated_text)} chars in {end - start:.2f} seconds")
    
    # Prepare response
    response = wire_formats.LLMPromptResponse.from_llm_prompt_job(job_desc, generated_text=generated_text)

    # Send response with fresh connection to avoid stale channel issues
    async with await dial_rabbit_from_config(config) as rabbitmq_connection:
        async with await rabbitmq_connection.channel() as channel:
            await channel.declare_queue(job_desc.reply_to, durable=True)
            # Copy headers from the incoming message to the response
            response_headers = {}
            if hasattr(message, 'headers') and message.headers:
                response_headers = dict(message.headers)

            logger.info(f"publishing response to {job_desc.reply_to}")
            await channel.default_exchange.publish(
                aio_pika.Message(
                    body=serialization.dumps(response).encode(),
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                    headers=response_headers if response_headers else None,
                ),
                routing_key=job_desc.reply_to,
            )

    logger.info(f"Job {job_desc.job_id} request {job_desc.request_id} completed and response sent")


async def main(config):
    """
    Main function to start the LLM worker with reconnection logic.
    Handles connection failures and automatically reconnects with exponential backoff.
    """
    logger.info("Initializing generic LLM worker...")

    # Retry configuration
    max_retries = 10
    base_retry_delay = 2  # seconds
    max_retry_delay = 60  # seconds

    # Load model once at startup
    logger.info("Loading LLM model...")
    model_path = config.get('model_path')
    hf_model_name = config.get('hf_model_name')
    global model_type
    model, tokenizer, model_type = load_quantized_llm_model(device, model_path, hf_model_name)
    
    retry_count = 0
    
    while True:
        try:
            # Connect to RabbitMQ
            logger.info(f"Connecting to RabbitMQ at {config['host']}:{config['port']}...")
            connection = await dial_rabbit_from_config(config)
            redis_client = await dial_redis_from_config(config)
            
            # Reset retry count on successful connection
            retry_count = 0
            
            async with connection:
                async with await connection.channel() as channel:
                    work_queue = config['work_queue']
                    await channel.declare_queue(work_queue, durable=True)
                
                logger.info(f"Successfully connected! Listening for LLM jobs on queue: {work_queue}")
                logger.info("Waiting for jobs. To exit press CTRL+C")
                
                # Start consuming messages
                async with CachedMessageIterator(
                        rabbitmq_connection=connection,
                        redis_client=redis_client,
                        queue_name=work_queue,
                        redis_key_prefix="backup:get_topics",
                        config=config,
                ) as queue_iter:
                    async for message in queue_iter:
                        async with queue_iter.processing(message):
                            await process_message(message, model, tokenizer, config)
                            
        except (AMQPError, ChannelInvalidStateError, ChannelClosed, ConnectionError) as conn_error:
            retry_count += 1
            if retry_count > max_retries:
                logger.error(f"Max retries ({max_retries}) exceeded. Giving up.")
                raise

            delay = min(base_retry_delay * (2 ** (retry_count - 1)), max_retry_delay)
            logger.error(f"Connection error: {conn_error}")
            logger.info(f"Reconnection attempt {retry_count}/{max_retries} in {delay} seconds...")
            await asyncio.sleep(delay)

        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}", exc_info=True)

            retry_count += 1
            if retry_count > max_retries:
                logger.error(f"Max retries ({max_retries}) exceeded. Giving up.")
                raise

            delay = min(base_retry_delay * (2 ** (retry_count - 1)), max_retry_delay)
            logger.info(f"Retrying in {delay} seconds...")
            await asyncio.sleep(delay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generic RabbitMQ LLM worker - takes prompts, returns generated text',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Configuration file format (JSON):
{
    "work_queue": "llm/generic",
    "model_path": "/path/to/model",
    "host": "localhost",
    "port": 5672,
    "username": "guest",
    "password": "guest"
}

Message format (JSON):
{
    "job_id": "uuid",
    "reply_to": "queue-name",
    "request_id": "optional-request-id",
    "prompt": "The prompt text",
    "params": {
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40
    }
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

    threading.Thread(
        target=watch_parent,
        args=(os.getppid(),),
        daemon=True,
        name="parent-watcher",
    ).start()

    logger.info(f"Loaded configuration from: {args.config_file}")
    logger.info(f"Work queue: {config['work_queue']}")
    logger.info(f"Model path: {config.get('model_path', 'default')}")
    logger.info(f"RabbitMQ host: {config['host']}:{config['port']}")
    logger.info(f"Username: {config['username']}")

    try:
        asyncio.run(main(config))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
