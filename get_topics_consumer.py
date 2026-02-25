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
import platform
import time

import aio_pika
import argparse
import torch
import asyncio

from aio_pika.abc import AbstractIncomingMessage
from aiormq import ChannelInvalidStateError, ChannelClosed, AMQPError
from pamqp.commands import Basic
from sympy.physics.units import temperature

import serialization
import wire_formats
from utils import load_config, create_ssl_context, load_quantized_llm_model, quantized_generate_from_prompt
from wire_formats import LLMPromptJob, Metaparams

# Device detection - prioritize CUDA over MPS
device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
model_type = None
print(f"Detected device: {device}")


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
    print(json.dumps(params, indent=4))
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
    print(f"Generation completed, {len(generated_text)} characters in {end - start:.2f} seconds")
    return generated_text


async def process_message(message: AbstractIncomingMessage, model, tokenizer):
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
        try:
            job_desc: LLMPromptJob = serialization.load(message.body.decode(), cls=LLMPromptJob)
        except TypeError as type_error:
            print(type_error)
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
        
        print(f"Received job {job_desc.job_id}, request {job_desc.request_id}")
        print(f"Prompt length: {len(job_desc.prompt)} chars")

        # Run text generation in thread pool to prevent blocking
        print(f"<prompt>\n{job_desc.prompt}\n</prompt>")
        loop = asyncio.get_event_loop()
        start = time.time()
        generated_text = await loop.run_in_executor(
            None,
            generate_text,
            job_desc.prompt,
            model,
            tokenizer,
            job_desc.meta_params.asdict(),
        )
        end = time.time()
        print(f"Generated {len(generated_text)} chars in {end - start:.2f} seconds")
        
        # Prepare response
        response = wire_formats.LLMPromptResponse.from_llm_prompt_job(job_desc, generated_text=generated_text)

        # Send response with error handling for invalid state
        try:
            channel = message.channel
            await channel.queue_declare(job_desc.reply_to, durable=True)
            # Copy headers from the incoming message to the response
            response_headers = {}
            if message.headers:
                response_headers = dict(message.headers)

            print(f"publishing response to {job_desc.reply_to}")
            resp = await channel.basic_publish(
                body=serialization.dumps(response).encode(),
                exchange="",
                routing_key=job_desc.reply_to,
                properties=Basic.Properties(
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                    headers=response_headers if response_headers else None,
                ),
            )
            print(f"response from basic publish: {resp}")
        except (ChannelInvalidStateError, ChannelClosed) as channel_error:
            print(f"Channel error while sending response for job {job_desc.job_id}: {channel_error}")
            print(f"Message will be re-queued for retry")
            await message.nack(requeue=True)
            return

        print(f"Job {job_desc.job_id} request {job_desc.request_id} completed and response sent")
        
        # Acknowledge successful processing
        await message.ack()
        
    except Exception as e:
        print(f"Error processing message: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to send error response if possible
        try:
            body = json.loads(message.body.decode())
            job_id = body.get('job_id', 'unknown')
            reply_to = body.get('reply_to')
            request_id = body.get('request_id')
            
            if reply_to:
                error_response = {
                    'job_id': job_id,
                    'request_id': request_id,
                    'status': 'error',
                    'error': str(e),
                }
                
                try:
                    channel = message.channel
                    
                    # Copy headers from the incoming message to the error response as well
                    response_headers = {}
                    if message.headers:
                        response_headers = dict(message.headers)
                    
                    await channel.default_exchange.publish(
                        aio_pika.Message(
                            body=json.dumps(error_response).encode(),
                            headers=response_headers if response_headers else None,
                        ),
                        routing_key=reply_to,
                    )
                except (ChannelInvalidStateError, ChannelClosed):
                    print(f"Could not send error response due to channel error - message will be requeued")
        except Exception as error_e:
            print(f"Error sending error response: {error_e}")
        
        # Nack the message so it gets requeued
        try:
            await message.nack(requeue=True)
        except Exception as nack_error:
            print(f"Error nacking message: {nack_error}")


async def main(config):
    """
    Main function to start the LLM worker with reconnection logic.
    Handles connection failures and automatically reconnects with exponential backoff.
    """
    print("Initializing generic LLM worker...")
    
    # Retry configuration
    max_retries = 10
    base_retry_delay = 2  # seconds
    max_retry_delay = 60  # seconds
    
    # Load model once at startup
    print("Loading LLM model...")
    model_path = config.get('model_path')
    hf_model_name = config.get('hf_model_name')
    global model_type
    model, tokenizer, model_type = load_quantized_llm_model(device, model_path, hf_model_name)
    
    ssl_context = create_ssl_context()
    
    retry_count = 0
    
    while True:
        try:
            # Connect to RabbitMQ
            print(f"Connecting to RabbitMQ at {config['host']}:{config['port']}...")
            
            connection = await aio_pika.connect_robust(
                host=config['host'],
                port=config['port'],
                login=config['username'],
                password=config['password'],
                ssl=True,
                ssl_context=ssl_context,
            )
            
            # Reset retry count on successful connection
            retry_count = 0
            
            async with connection:
                channel = await connection.channel()
                await channel.set_qos(prefetch_count=1)
                
                work_queue = config['work_queue']
                queue = await channel.declare_queue(work_queue, durable=True)
                
                print(f"Successfully connected! Listening for LLM jobs on queue: {work_queue}")
                print("Waiting for jobs. To exit press CTRL+C")
                
                async with queue.iterator() as queue_iter:
                    async for message in queue_iter:
                        try:
                            await process_message(message, model, tokenizer)
                        except (ChannelInvalidStateError, ChannelClosed) as channel_err:
                            print(f"Channel error during message processing: {channel_err}")
                            print("Will attempt to reconnect...")
                            raise
                        except Exception as e:
                            print(f"Unexpected error processing message: {e}")
                            import traceback
                            traceback.print_exc()
                            
        except (AMQPError, ChannelInvalidStateError, ChannelClosed, ConnectionError) as conn_error:
            retry_count += 1
            if retry_count > max_retries:
                print(f"Max retries ({max_retries}) exceeded. Giving up.")
                raise
            
            delay = min(base_retry_delay * (2 ** (retry_count - 1)), max_retry_delay)
            print(f"Connection error: {conn_error}")
            print(f"Reconnection attempt {retry_count}/{max_retries} in {delay} seconds...")
            await asyncio.sleep(delay)
            
        except KeyboardInterrupt:
            print("\nShutting down gracefully...")
            break
        except Exception as e:
            print(f"Unexpected error in main loop: {e}")
            import traceback
            traceback.print_exc()
            
            retry_count += 1
            if retry_count > max_retries:
                print(f"Max retries ({max_retries}) exceeded. Giving up.")
                raise
            
            delay = min(base_retry_delay * (2 ** (retry_count - 1)), max_retry_delay)
            print(f"Retrying in {delay} seconds...")
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

    print(f"Loaded configuration from: {args.config_file}")
    print(f"Work queue: {config['work_queue']}")
    print(f"Model path: {config.get('model_path', 'default')}")
    print(f"RabbitMQ host: {config['host']}:{config['port']}")
    print(f"Username: {config['username']}")

    try:
        asyncio.run(main(config))
    except KeyboardInterrupt:
        print("\nInterrupted by user")
