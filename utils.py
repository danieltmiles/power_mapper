import asyncio
import io
import os
import re
import subprocess
import sys
import json
import ssl
import tempfile
from dataclasses import dataclass

# from difflib import SequenceMatcher
from typing import Optional
from urllib.parse import urlparse
import redis.asyncio as redis

import aio_pika
import torch.cuda
from aio_pika.abc import AbstractRobustConnection
from huggingface_hub.errors import RemoteEntryNotFoundError

from logger import get_logger

logger = get_logger("utils")


#import torch
# import torchaudio
#from pydub import AudioSegment
# from torch import Tensor
# from transformers import AutoTokenizer


@dataclass
class TranscriptionJob:
    job_id: str
    filename: str
    human_readable_filename: str
    status: str
    progress: int
    transcript: Optional[str] = None
    error: Optional[str] = None

# Iterator to create audio segments from diarization timestamps
def diarized_segment_iter(signal, diarization, sample_rate):
    """
    Yields audio segments based on diarization timestamps.

    Args:
        signal: Audio signal tensor
        diarization: DiarizeOutput object with speaker_diarization attribute
        sample_rate: Sample rate of the audio

    Yields:
        dict with 'audio', 'start', 'end', and 'speaker' keys
    """
    for diar_seg in diarization.speaker_diarization:
        start_time = diar_seg[0].start
        end_time = diar_seg[0].end
        speaker = diar_seg[1]

        # Convert time to sample indices
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)

        # Extract audio segment
        audio_segment = signal[start_sample:end_sample]
        squeezed_segment = audio_segment.squeeze()
        if hasattr(squeezed_segment, "numpy"):
            squeezed_segment = squeezed_segment.numpy()

        yield {
            'audio': squeezed_segment,
            'start': start_time,
            'end': end_time,
            'speaker': speaker
        }

# Create speaker assignment for all segments based on temporal overlap
def assign_speaker_to_segment(diarization, segment_start, segment_end):
    """Assign speaker to a segment based on temporal overlap with diarization segments"""
    best_overlap = 0
    best_speaker = "Speaker_Unknown"  # default fallback

    for i, (diar_seg, speaker) in enumerate(diarization.speaker_diarization):
        # Calculate overlap between transcription segment and diarization segment
        overlap_start = max(segment_start, diar_seg.start)
        overlap_end = min(segment_end, diar_seg.end)
        overlap = max(0, overlap_end - overlap_start)

        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = speaker

    return best_speaker


def normalize_audio(audio_file_path: str):
    import torch
    import torchaudio

    file_extension = os.path.splitext(audio_file_path)[1].lower()
    logger.info(f"Loading audio file {audio_file_path}")

    temp_wav_path = None
    try:
        if file_extension != ".wav":
            # Create a temporary WAV file on disk
            temp_wav_fd, temp_wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(temp_wav_fd)  # Close the file descriptor, we just need the path

            # Use ffmpeg to convert directly on disk without loading into memory
            # Convert to 16kHz mono directly since that's what we need for Whisper
            logger.info(f"Converting to temporary WAV file using ffmpeg: {temp_wav_path}")
            num_threads = os.cpu_count() or 1  # Get CPU count, default to 1 if unavailable
            result = subprocess.run(
                [
                    'ffmpeg',
                    '-threads', str(num_threads),  # Use all available CPU threads
                    '-i', audio_file_path,  # Input file
                    '-acodec', 'pcm_s16le',  # PCM 16-bit little-endian codec
                    '-ar', '16000',  # 16kHz sample rate (required for Whisper)
                    '-ac', '1',  # Mono audio
                    '-y',  # Overwrite output file if it exists
                    temp_wav_path  # Output file
                ],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"ffmpeg conversion completed using {num_threads} threads")
            
            # Load from temporary WAV file using torchaudio
            signal, sr = torchaudio.load(temp_wav_path)
        else:
            # Directly load WAV files
            signal, sr = torchaudio.load(audio_file_path)
        
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        signal = signal.squeeze()

        # whisper needs a sample rate of 16000
        if sr != 16000:
            signal = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(signal)
            sr = 16000
        return signal, sr
    finally:
        # Clean up temporary WAV file if it was created
        if temp_wav_path and os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path)
                logger.info(f"Cleaned up temporary WAV file: {temp_wav_path}")
            except Exception as e:
                logger.error(f"Could not remove temporary file {temp_wav_path}: {e}")


def load_config(config_file):
    """Load configuration from JSON file.
    
    Expected JSON format:
    {
        "work_queue": "queue_name",
        "host": "localhost",
        "port": 5672,
        "username": "guest",
        "password": "guest"
    }
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Validate required fields
        required_fields = ['work_queue', 'host', 'port', 'username', 'password']
        missing_fields = [field for field in required_fields if field not in config]
        
        if missing_fields:
            raise ValueError(f"Missing required fields in config file: {', '.join(missing_fields)}")
        
        # Ensure port is an integer
        config['port'] = int(config['port'])
        
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file '{config_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"{e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)


def create_ssl_context(cert_file='server_certificate.pem', verify=True):
    cafile = cert_file if cert_file and os.path.exists(cert_file) else None
    ssl_context = ssl.create_default_context(cafile=cafile)
    if not verify:
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
    return ssl_context

def get_available_video_memory() -> int:
    import torch
    if torch.cuda.is_available():
        import nvidia_smi
        try:
            nvidia_smi.nvmlInit()

            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            logger.info(f"Total video memory: {info.total / 1024 ** 3}GB")
            logger.info(f"Free video memory: {info.free / 1024 ** 3}GB")
            logger.info(f"Used video memory: {info.used / 1024 ** 3}GB")
            return info.free
        finally:
            nvidia_smi.nvmlShutdown()
    elif torch.mps.is_available():
        import psutil
        meminfo = psutil.virtual_memory()
        logger.info(f"Apple Silicon system memory available to GPU: {meminfo.available / 1024 ** 3}GB")
        return meminfo.available
    else:
        logger.warning("No available VRAM detected. Assuming 23GB.")
        return 23 * 1024 ** 3


def estimate_max_context(gguf_path: str, vram_bytes: int = None) -> tuple[int, int]:
    """
    Examine a GGUF file, determine its memory requirements, and return the
    maximum context length (in tokens) that will fit in the given VRAM budget.

    The estimate accounts for:
      - Model weights (approximated by the file size on disk)
      - KV cache in FP16: 2 (K+V) × n_layers × n_kv_heads × head_dim × 2 bytes per token
      - A fixed overhead allowance for activations / framework bookkeeping

    Args:
        gguf_path:  Path to the GGUF model file.
        vram_bytes: Total VRAM budget in bytes (default 24 GB).

    Returns:
        Maximum context length in tokens that fits within the VRAM budget.
    """

    vram_bytes = vram_bytes or get_available_video_memory()

    from gguf import GGUFReader

    reader = GGUFReader(gguf_path)

    # Discover the architecture prefix (e.g. "qwen3", "llama", …)
    arch_field = reader.fields.get("general.architecture")
    arch = bytes(arch_field.parts[-1]).decode() if arch_field else ""

    def _read_int(key: str) -> int:
        field = reader.fields.get(key)
        if field is None:
            raise ValueError(f"GGUF file is missing metadata key: {key}")
        return int(field.parts[-1][0])

    n_layers   = _read_int(f"{arch}.block_count")
    logger.info(f"{n_layers=}")
    n_kv_heads = _read_int(f"{arch}.attention.head_count_kv")
    key_dim    = _read_int(f"{arch}.attention.key_length")
    value_dim  = _read_int(f"{arch}.attention.value_length")

    # Weight memory ≈ file size (quantised weights dominate the file)
    weight_bytes = os.path.getsize(gguf_path)

    # Fixed overhead for activations, scratch buffers, framework state
    overhead_bytes = 512 * 1024**2  # 512 MB

    available = vram_bytes - weight_bytes - overhead_bytes
    if available <= 0:
        raise ValueError(
            f"Model weights ({weight_bytes / 1024**3:.1f} GB) + overhead "
            f"exceed VRAM budget ({vram_bytes / 1024**3:.1f} GB)"
        )

    # KV cache per token (FP16 = 2 bytes per element)
    #   2 (K and V) × n_layers × n_kv_heads × head_dim × 2 bytes
    kv_bytes_per_token = 2 * n_layers * n_kv_heads * (key_dim + value_dim) * 2

    max_ctx = available // kv_bytes_per_token
    # fudge a little
    max_ctx *= 1.1


    # How many layers can we offload to GPU while fitting a target context?
    target_ctx = 10240
    kv_per_token_per_layer = 2 * n_kv_heads * (key_dim + value_dim) * 2
    cost_per_layer = (weight_bytes / n_layers) + (kv_per_token_per_layer * target_ctx)
    max_layers = min(int((vram_bytes - overhead_bytes) // cost_per_layer), n_layers)
    logger.info(
        f"\nGGUF estimate for {os.path.basename(gguf_path)}:\n"
        f"weights={weight_bytes / 1024**3:.1f} GB,\n"
        f"weight bytes per layer={weight_bytes / n_layers / 1024**2:.1f} MB,\n"
        f"kv/token={kv_bytes_per_token} B,\n"
        f"max_context={max_ctx} tokens\n"
        f"max_layers={max_layers} layers\n"
        f"(vram budget={vram_bytes / 1024**3:.0f} GB)"
    )
    return int(max_ctx), int(max_layers)


def load_quantized_llm_model(device: str, model_path: str = None, hf_model_name: str = None):
    """
    Load the LLM model for speaker identification.

    Supports different hardware backends:
    - CUDA (NVIDIA): Uses llama-cpp-python for GPU acceleration
    - MPS (Apple Metal): Uses llama-cpp-python with GGUF models
    - CPU: Not optimized for quantized models

    Args:
        model_path: Path to the model (optional, uses default if not provided)

    Returns:
        tuple: (model, tokenizer) - tokenizer may be None for llama-cpp
    """
    from transformers import AutoTokenizer
    from llama_cpp import Llama
    try:
        logger.info(f"Loading GGUF model for {device} device...")
        model_path = model_path or "./Qwen3-32B-Q4_K_M.gguf"
        hf_model_name = hf_model_name or "Qwen/Qwen3-32B"

        # Temporarily suppress llama.cpp warnings
        os.environ['LLAMA_LOG_DISABLE'] = '1'
        max_ctx, max_layers = estimate_max_context(model_path)
        if max_ctx < 10240:
            max_ctx = 10240
        else:
            max_layers = -1
        max_ctx = min(10240, max_ctx)
        logger.info(f"{max_layers=}, {max_ctx=}")
        model = Llama(
            model_path=model_path,
            n_gpu_layers=max_layers,  # Use all GPU layers
            n_ctx=max_ctx,  # Context window size
            use_mmap=True,   # Map the model file into memory rather than copying it
            use_mlock=False, # Don't pin pages — let the OS page out unused weights
            verbose=False
        )
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

        # Some models (e.g. Gemma) ship chat_template as a separate .jinja file
        # rather than embedding it in tokenizer_config.json. Load it manually if needed.
        if not getattr(tokenizer, 'chat_template', None):
            try:
                from huggingface_hub import hf_hub_download
                jinja_path = hf_hub_download(hf_model_name, "chat_template.jinja")
                with open(jinja_path, "r") as f:
                    tokenizer.chat_template = f.read()
                logger.info(f"Loaded chat_template.jinja for {hf_model_name}")
            except RemoteEntryNotFoundError:
                if "gemma" in hf_model_name.lower():
                    tokenizer.chat_template = (
                        "{% for message in messages %}"
                        "<|turn>{{ message['role'] }}\n{{ message['content'] }}<turn|>\n"
                        "{% endfor %}"
                        "{% if add_generation_prompt %}<|turn>model{% endif %}"
                    )
                    logger.info(f"Applied built-in Gemma chat template for {hf_model_name}")
                else:
                    logger.warning(
                        f"No chat_template found for {hf_model_name}. "
                        f"apply_chat_template() may fail for conversation-style prompts."
                    )
                    tokenizer.chat_template = (
                        "{% for message in messages %}"
                        "{{ message['content'] }}\n"
                        "{% endfor %}"
                    )

        # Re-enable logging after model load
        if 'LLAMA_LOG_DISABLE' in os.environ:
            del os.environ['LLAMA_LOG_DISABLE']

        model_type = "llamacpp"
        logger.info(f"Successfully loaded GGUF model: {model_path}")
        return model, tokenizer, model_type  # llama-cpp handles tokenization internally
    except ImportError:
        logger.error("llama-cpp-python not available. Install llama-cpp-python for CUDA support.")
        raise
    except Exception as e:
        logger.error(f"Error loading GGUF model: {e}")
        raise

# Keywords that indicate a thinking/reasoning token
_THINK_KEYWORDS = frozenset({
    'think', 'thought', 'thinking',
    'reason', 'reasoning',
    'reflect', 'reflection',
    'internal',
    'scratchpad',
    'analysis',
})

# Patterns that indicate a closing/end token
_CLOSING_PATTERNS = ('/', 'end_of_', '_end', 'end_', '/>')


def _collect_special_tokens(tokenizer) -> list[str]:
    """Collect all candidate special/added tokens from any tokenizer."""
    seen: set[str] = set()
    candidates: list[str] = []

    def add(token):
        if token and isinstance(token, str) and token not in seen:
            seen.add(token)
            candidates.append(token)

    for token in getattr(tokenizer, 'additional_special_tokens', []):
        add(token)
    if hasattr(tokenizer, 'added_tokens_encoder'):
        for token in tokenizer.added_tokens_encoder:
            add(token)
    if hasattr(tokenizer, 'special_tokens_map'):
        for value in tokenizer.special_tokens_map.values():
            if isinstance(value, str):
                add(value)
            elif isinstance(value, list):
                for item in value:
                    add(item)

    return candidates


def _contains_think_keyword(token: str) -> bool:
    """True if the token relates to thinking/reasoning."""
    t = token.lower()
    return any(kw in t for kw in _THINK_KEYWORDS)


def _is_closing_token(token: str) -> bool:
    """True if the token marks the END of a thinking block."""
    t = token.lower()
    return any(pattern in t for pattern in _CLOSING_PATTERNS)


def find_start_think_token(tokenizer) -> Optional[str]:
    """
    Find the opening thinking token for any tokenizer.

    Handles formats such as:
      <think>  <|begin_of_thought|>  <Thought>  <|thinking|>  <|im_think|>
    """
    if tokenizer is None:
        return None
    for token in _collect_special_tokens(tokenizer):
        if _contains_think_keyword(token) and not _is_closing_token(token):
            return token
    return None


def find_end_think_token(tokenizer) -> Optional[str]:
    """
    Find the closing thinking token for any tokenizer.

    Handles formats such as:
      </think>  <|end_of_thought|>  </Thought>  <|end_thinking|>
    """
    if tokenizer is None:
        return None
    for token in _collect_special_tokens(tokenizer):
        if _contains_think_keyword(token) and _is_closing_token(token):
            return token
    return None


def find_end_think_token_id(tokenizer) -> Optional[int]:
    """Find the token ID of the closing thinking token."""
    token_str = find_end_think_token(tokenizer)
    if token_str is None or tokenizer is None:
        return None
    # Fast path: direct lookup in added_tokens_encoder
    if hasattr(tokenizer, 'added_tokens_encoder') and token_str in tokenizer.added_tokens_encoder:
        return tokenizer.added_tokens_encoder[token_str]
    # Fallback: encode and take the last ID
    try:
        ids = tokenizer.encode(token_str, add_special_tokens=False)
        if ids:
            return ids[-1]
    except Exception:
        pass
    return None


def quantized_generate_from_prompt(
    prompt: str,
    model,
    tokenizer,
    model_type,
    max_tokens: int = 12288,
    temperature: float = 0.7,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    repetition_penalty: float = 1.1,
    **kwargs,
) -> str:
    """
    Generate text from a prompt using the appropriate backend.

    Handles llama-cpp model types with their respective APIs.

    Args:
        prompt: The input prompt
        model: The loaded model
        tokenizer: The tokenizer (None for llama-cpp)
        max_tokens: Maximum number of tokens to generate (default: 12288, acts as safety limit)

    Returns:
        str: The generated text
    """
    if model_type == "llamacpp":
        # llama-cpp-python streaming generation - stops naturally on EOS token
        try:
            generated_text = ""

            # Stream the response with stream=True
            metakwargs = {}
            if max_tokens is not None:
                metakwargs['max_tokens'] = max_tokens
            if temperature is not None:
                metakwargs['temperature'] = temperature
            if top_p is not None:
                metakwargs['top_p'] = top_p
            if top_k is not None:
                metakwargs['top_k'] = top_k
            if min_p is not None:
                metakwargs['min_p'] = min_p
            if repetition_penalty is not None:
                metakwargs['repeat_penalty'] = repetition_penalty
            for k, v in kwargs.items():
                metakwargs[k] = v

            # Gemma models need explicit stop sequences to avoid recapitulating the prompt
            if 'stop' not in metakwargs and hasattr(model, 'model_path'):
                model_path_lower = (model.model_path or "").lower() if isinstance(model.model_path, str) else ""
                if "gemma" in model_path_lower:
                    metakwargs['stop'] = ["<turn|>", "<|turn>"]
                    logger.info("Added Gemma stop sequences: <turn|>, <|turn>")

            stream = model(
                prompt,
                echo=False,
                stream=True,  # Enable streaming
                **metakwargs,
            )

            # Accumulate tokens from the stream
            for chunk in stream:
                # Each chunk has the structure: {'choices': [{'text': '...', 'finish_reason': ...}]}
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    choice = chunk['choices'][0]
                    token_text = choice.get('text', '')
                    print(token_text, end='', flush=True)
                    generated_text += token_text

                    # Check for finish reason (None, 'stop', 'length')
                    finish_reason = choice.get('finish_reason', None)
                    if finish_reason == 'stop':
                        logger.info("EOS token detected, stopping generation.")
                        break
                    elif finish_reason == 'length':
                        logger.info("Max tokens reached.")
                        break

            return generated_text.strip()
        except Exception as e:
            logger.error(f"GGUF generation error: {e}", exc_info=True)
            return ""

    else:
        logger.error(f"Unknown model type: {model_type}")
        return ""


async def gather_with_concurrency(n, *coros, **kwargs):
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro):
        async with semaphore:
            return await coro
    return await asyncio.gather(*(sem_coro(c) for c in coros), **kwargs)


def get_answer(generated: str, start_delim: str, end_delim: str) -> str:
    try:
        if not generated:
            return ""
        begin_indexes = [i for i in range(len(generated)) if generated.startswith(start_delim, i)]
        end_indexes = [i for i in range(len(generated)) if generated.startswith(end_delim, i)]
        if not begin_indexes:
            begin_indexes = [0]
        if not end_indexes:
            end_indexes = [len(generated)]
        # print(f"{begin_indexes=}")
        # print(f"{end_indexes=}")
        if begin_indexes == end_indexes:
            idx = -1
            while abs(idx) <= len(begin_indexes):
                start_idx = begin_indexes[idx-1]
                end_idx = end_indexes[idx]
                ret = generated[start_idx + len(start_delim):end_idx]
                if stripped := ret.strip():
                    return stripped
                idx -= 1
        if ret := generated[begin_indexes[-1] + len(start_delim):end_indexes[-1]]:
            return ret.strip()
        ret = generated[begin_indexes[-2] + len(start_delim):end_indexes[-1]]
        answer = ret.strip()
        return answer
    except IndexError:
        #print(f"\033[91mIndex error getting answer, bad indexes, returning empty string\033[0m")
        pass
    return ""


def parse_smb_uri(smb_uri: str) -> tuple[Optional[str], Optional[str]]:
    """
    Parse a SMB URI into address and share name components.

    Args:
        smb_uri: SMB URI in format smb://address/sharename

    Returns:
        Tuple of (address, sharename) or (None, None) if invalid
    """
    try:
        parsed = urlparse(smb_uri)
        if parsed.scheme != 'smb':
            return None, None

        address = parsed.hostname or parsed.netloc.split('/')[0]
        # Get the path without leading slash
        sharename = parsed.path.lstrip('/').split('/')[0] if parsed.path else None

        return address, sharename
    except Exception:
        return None, None


def find_smb_mount_macos(address: str, sharename: str) -> Optional[str]:
    """
    Find SMB mount on macOS using the mount command.

    Args:
        address: The server address/hostname
        sharename: The share name

    Returns:
        Local mount path or None if not found
    """
    try:
        # Run mount command to get all mounted filesystems
        result = subprocess.run(['mount'], capture_output=True, text=True, check=True)

        # Parse mount output
        # Format on macOS: //user@address/sharename on /mount/point (smbfs, ...)
        for line in result.stdout.splitlines():
            if 'smbfs' in line.lower() or 'cifs' in line.lower():
                # Match patterns like: //address/sharename on /path or //user@address/sharename on /path
                match = re.search(r'//(?:[^@]+@)?([^/]+)/([^\s]+)\s+on\s+([^\s]+(?:\s+[^\(]+)?)\s*\(', line)
                if match:
                    mount_address = match.group(1)
                    mount_share = match.group(2)
                    mount_point = match.group(3).strip()

                    # Check if this matches our target share
                    if mount_share == sharename and (mount_address == address or
                                                     mount_address.lower() == address.lower()):
                        return mount_point

        return None
    except subprocess.CalledProcessError:
        return None
    except Exception:
        return None


def find_smb_mount_linux(address: str, sharename: str) -> Optional[str]:
    """
    Find SMB mount on Linux using /proc/mounts or mount command.

    Args:
        address: The server address/hostname
        sharename: The share name

    Returns:
        Local mount path or None if not found
    """
    try:
        # First try reading /proc/mounts
        try:
            with open('/proc/mounts', 'r') as f:
                mounts_content = f.read()
        except FileNotFoundError:
            # If /proc/mounts doesn't exist, use mount command
            result = subprocess.run(['mount'], capture_output=True, text=True, check=True)
            mounts_content = result.stdout

        # Parse mount entries
        # Format on Linux: //address/sharename /mount/point cifs ...
        for line in mounts_content.splitlines():
            if 'cifs' in line.lower() or 'smb' in line.lower():
                parts = line.split()
                if len(parts) >= 2:
                    device = parts[0]
                    mount_point = parts[1]

                    # Match patterns like //address/sharename
                    match = re.match(r'//([^/]+)/(.+)', device)
                    if match:
                        mount_address = match.group(1)
                        mount_share = match.group(2)

                        # Check if this matches our target share
                        if mount_share == sharename and (mount_address == address or
                                                         mount_address.lower() == address.lower()):
                            return mount_point

        return None
    except Exception:
        return None


def find_smb_mount(smb_uri: str) -> Optional[str]:
    """
    Find the local mount point for a given SMB URI.

    This function examines the local machine (both Linux and macOS compatible)
    to determine if the given Samba share is mounted anywhere.

    Args:
        smb_uri: SMB URI in format smb://address/sharename

    Returns:
        Local directory path where the share is mounted, or None if not found

    Example:
        >>> find_smb_mount("smb://server.local/shared")
        '/Volumes/shared'

        >>> find_smb_mount("smb://192.168.1.100/data")
        '/mnt/data'
    """
    # Parse the SMB URI
    address, sharename = parse_smb_uri(smb_uri)

    if not address or not sharename:
        return None

    # Determine the operating system
    system = sys.platform.system()

    if system == 'Darwin':  # macOS
        return find_smb_mount_macos(address, sharename)
    elif system == 'Linux':
        return find_smb_mount_linux(address, sharename)
    else:
        # Unsupported OS
        return None


class SimilarityCalculator:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Ensure initialization happens only once
        if not self._initialized:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self._initialized = True

    def semantic_similarity(self, text1, text2):
        embeddings = self.model.encode([text1, text2])
        from scipy.spatial.distance import cosine
        return 1 - cosine(embeddings[0], embeddings[1])

    def simple_similarity(self, text1, text2) -> float:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1.lower().strip(), text2.lower().strip()).ratio()

    def jaccard_similarity(self, text1, text2):
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union) if union else 0

    def text_similarity(self, text1, text2):
        simple_sim = self.simple_similarity(text1, text2)
        jaccard_sim = self.jaccard_similarity(text1, text2)
        semantic_sim = self.semantic_similarity(text1, text2)
        return 0.5 * simple_sim + 0.5 * jaccard_sim + 2 * semantic_sim / 3


VECTOR_DIMENSION = 384  # all-MiniLM-L6-v2 output size (used by SimilarityCalculator / scribe)


def encode_text_to_vector(text: str) -> list[float]:
    return SimilarityCalculator().model.encode(text).tolist()


def encode_texts_to_vectors(texts: list[str]) -> list[list[float]]:
    """Encode multiple texts in a single batched inference pass.

    Prefer this over calling encode_text_to_vector in a loop — the model
    processes the whole list as one matrix operation, which is significantly
    faster than N individual calls on both CPU and GPU.
    """
    return SimilarityCalculator().model.encode(texts).tolist()


class EmbeddingModel:
    """Singleton llama_cpp model loaded in embedding mode.

    Call EmbeddingModel().init(model_path) once at startup before use.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
            cls._instance.dimension = None
        return cls._instance

    def init(self, model_path: str, n_ctx: int = 2048, n_gpu_layers: int = -1) -> None:
        if self._model is not None:
            return
        import llama_cpp
        logger.info(f"Loading embedding model from {model_path}...")
        self._model = llama_cpp.Llama(
            model_path=model_path,
            embedding=True,
            pooling_type=llama_cpp.LLAMA_POOLING_TYPE_LAST,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            use_mmap=True,
            use_mlock=False,
            verbose=False,
        )
        # Derive the output dimension from a live test rather than hardcoding.
        self.dimension = len(self._model.embed("test"))
        logger.info(f"Embedding model loaded (dimension={self.dimension})")

    def encode(self, text: str) -> list[float]:
        return self._model.embed(text)

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        # llama_cpp serialises embedding sequences one at a time internally, so
        # passing a list batches all tokens into a single decode call and blows
        # past n_ctx for any non-trivial batch size.  A plain loop is equivalent
        # in throughput and avoids the context overflow.
        old_stderr = sys.stderr
        try:
            sys.stderr = io.StringIO()
            return [self._model.embed(text) for text in texts]
        finally:
            sys.stderr = old_stderr


async def ensure_solr_vector_field_type(client, schema_url: str, dimension: int = VECTOR_DIMENSION) -> str:
    """Ensure the knn_vector field type exists in the Solr schema.

    Returns the field type name so callers can reference it when defining fields.
    Pass `dimension` explicitly when using a model other than all-MiniLM-L6-v2.
    """
    field_types_response = await client.get(f"{schema_url}/fieldtypes")
    field_types_response.raise_for_status()
    existing_type_names = {ft["name"] for ft in field_types_response.json().get("fieldTypes", [])}
    vector_field_type_name = f"knn_vector_{dimension}"

    if vector_field_type_name not in existing_type_names:
        logger.info(f"Adding field type '{vector_field_type_name}'...")
        type_response = await client.post(
            schema_url,
            json={
                "add-field-type": {
                    "name": vector_field_type_name,
                    "class": "solr.DenseVectorField",
                    "vectorDimension": dimension,
                    "similarityFunction": "cosine",
                }
            },
        )
        type_response.raise_for_status()
        logger.info(f"Field type '{vector_field_type_name}' added")
    else:
        logger.info(f"Field type '{vector_field_type_name}' already exists")

    return vector_field_type_name


def load_hf_token(token_file="hf_token.txt"):
    """Load HuggingFace token from a file.

    Args:
        token_file: Path to the token file (default: hf_token.txt)

    Returns:
        str: The token string

    Raises:
        FileNotFoundError: If the token file doesn't exist
        ValueError: If the token file is empty
    """
    token_path = os.path.join(os.path.dirname(__file__), token_file)
    if not os.path.exists(token_path):
        raise FileNotFoundError(f"Token file not found: {token_path}")

    with open(token_path, 'r') as f:
        token = f.read().strip()

    if not token:
        raise ValueError(f"Token file is empty: {token_path}")

    return token


# Redis backup utilities for message resilience
async def connect_redis(redis_config: dict):
    """Connect to Redis for message backup.
    
    Args:
        redis_config: Dictionary with Redis connection parameters
                     (host, port, ssl, ssl_ca_certs, ssl_certfile, ssl_keyfile)
    
    Returns:
        redis.Redis client or None if no config provided
    """
    import redis.asyncio as redis
    
    if not redis_config:
        logger.info("No Redis configuration found. Message backup disabled.")
        return None

    redis_client = await redis.Redis(
        host=redis_config['host'],
        port=redis_config['port'],
        ssl=redis_config.get('ssl', False),
        ssl_ca_certs=redis_config.get('ssl_ca_certs'),
        ssl_certfile=redis_config.get('ssl_certfile'),
        ssl_keyfile=redis_config.get('ssl_keyfile'),
    )
    logger.info(f"Connected to Redis at {redis_config['host']}:{redis_config['port']}")
    return redis_client


async def backup_message_to_redis(
    redis_client,
    semaphore: asyncio.Semaphore,
    key_prefix: str,
    identifier: str,
    message_body: bytes
):
    """Backup message to Redis before acknowledging.
    
    Args:
        redis_client: Redis client instance
        semaphore: Asyncio semaphore for rate limiting
        key_prefix: Prefix for Redis key (e.g., "backup:diarizations")
        identifier: Unique identifier for this message
        message_body: Raw message body bytes
    """
    if redis_client is None:
        return

    key = f"{key_prefix}:{identifier}"
    async with semaphore:
        await redis_client.set(key, message_body)


async def remove_message_from_redis(
    redis_client,
    semaphore: asyncio.Semaphore,
    key_prefix: str,
    identifier: str
):
    """Remove backed up message from Redis after processing.
    
    Args:
        redis_client: Redis client instance
        semaphore: Asyncio semaphore for rate limiting
        key_prefix: Prefix for Redis key
        identifier: Unique identifier for this message
    """
    if redis_client is None:
        return
    
    key = f"{key_prefix}:{identifier}"
    async with semaphore:
        await redis_client.delete(key)


async def cleanup_redis_backups_by_pattern(
    redis_client,
    semaphore: asyncio.Semaphore,
    pattern: str
):
    """Remove all backup messages matching a pattern.
    
    Args:
        redis_client: Redis client instance
        semaphore: Asyncio semaphore for rate limiting
        pattern: Redis key pattern (e.g., "backup:filename:*")
    """
    if redis_client is None:
        return
    
    cursor = 0
    while True:
        async with semaphore:
            cursor, keys = await redis_client.scan(cursor, match=pattern, count=100)
            if keys:
                await redis_client.delete(*keys)
            if cursor == 0:
                break
    logger.info(f"Cleaned up Redis backups matching pattern: {pattern}")


async def recover_messages_from_redis(
    redis_client,
    semaphore: asyncio.Semaphore,
    pattern: str,
    deserialize_func,
    process_func
):
    """Recover messages from Redis after a crash or restart.
    
    Args:
        redis_client: Redis client instance
        semaphore: Asyncio semaphore for rate limiting
        pattern: Redis key pattern to match (e.g., "backup:*")
        deserialize_func: Function to deserialize message body
        process_func: Function to process each recovered message
                     Should accept (key, deserialized_message)
    
    Returns:
        Number of messages recovered
    """
    if redis_client is None:
        logger.info("No Redis connection - skipping recovery")
        return 0

    logger.info("Checking Redis for backed up messages...")
    cursor = 0
    recovered_count = 0
    
    # Collect all backup keys
    all_keys = []
    while True:
        async with semaphore:
            cursor, keys = await redis_client.scan(cursor, match=pattern, count=100)
            all_keys.extend(keys)
            if cursor == 0:
                break
    
    if not all_keys:
        logger.info("No backed up messages found in Redis")
        return 0

    logger.info(f"Found {len(all_keys)} backed up messages in Redis, recovering...")
    
    # Process each backed up message
    for key in all_keys:
        try:
            # Get the message body from Redis
            async with semaphore:
                message_body = await redis_client.get(key)
            
            if message_body is None:
                continue
            
            # Deserialize and process
            deserialized = deserialize_func(message_body.decode())
            await process_func(key, deserialized)
            
            recovered_count += 1
            
        except Exception as e:
            logger.error(f"Error recovering message from key {key}: {e}")
            continue

    logger.info(f"Successfully recovered {recovered_count} messages from Redis")
    return recovered_count

async def publish_event(config: dict, event_message: str):
    """Publish a human-readable event message to the 'events' queue for pipeline observability.
    
    This is fire-and-forget instrumentation - failures to publish events
    will never break the main processing pipeline.
    
    Args:
        config: RabbitMQ connection config dict (must have host, port, username, password)
        event_message: Human-readable description of the event
    """
    import datetime
    try:
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        text = f"[{timestamp}] {event_message}"
        logger.info(f"EVENT: {text}")
        async with await dial_rabbit_from_config(config) as connection:
            async with await connection.channel() as channel:
                await channel.declare_queue("events", durable=True)
                await channel.default_exchange.publish(
                    aio_pika.Message(
                        body=text.encode("utf-8"),
                        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                    ),
                    routing_key="events",
                )
    except Exception as e:
        logger.error(f"Failed to publish event '{event_message}': {e}")


async def dial_rabbit_from_config(config: dict) -> AbstractRobustConnection:
    logger.info(f"Creating new RabbitMQ connection to {config['host']}:{config['port']}...")
    rabbit_connection = await aio_pika.connect_robust(
        host=config['host'],
        port=config['port'],
        login=config['username'],
        password=config['password'],
        ssl=True,
        ssl_context=create_ssl_context(config.get('ssl_cert_file', 'server_certificate.pem')),
        heartbeat=60,
    )
    logger.info("RabbitMQ connection established successfully")
    
    return rabbit_connection

async def dial_redis_from_config(redis_config: dict) -> redis.Redis:
    if "redis" in redis_config:
        redis_config = redis_config["redis"]
    return await redis.Redis(
        host=redis_config['host'],
        port=redis_config['port'],
        ssl=redis_config.get('ssl', False),
        ssl_ca_certs=redis_config.get('ssl_ca_certs'),
        ssl_certfile=redis_config.get('ssl_certfile'),
        ssl_keyfile=redis_config.get('ssl_keyfile'),
    )
