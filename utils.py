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
    print(f"Loading audio file {audio_file_path}")
    
    temp_wav_path = None
    try:
        if file_extension != ".wav":
            # Create a temporary WAV file on disk
            temp_wav_fd, temp_wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(temp_wav_fd)  # Close the file descriptor, we just need the path
            
            # Use ffmpeg to convert directly on disk without loading into memory
            # Convert to 16kHz mono directly since that's what we need for Whisper
            print(f"Converting to temporary WAV file using ffmpeg: {temp_wav_path}")
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
            print(f"ffmpeg conversion completed using {num_threads} threads")
            
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
                print(f"Cleaned up temporary WAV file: {temp_wav_path}")
            except Exception as e:
                print(f"Warning: Could not remove temporary file {temp_wav_path}: {e}")


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
        print(f"Error: Configuration file '{config_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)


def create_ssl_context(cert_file='server_certificate.pem', verify=True):
    """Create SSL context for RabbitMQ connections.
    
    Args:
        cert_file: Path to the certificate file
        verify: Whether to verify certificates (set to False for self-signed)
    
    Returns:
        ssl.SSLContext configured for RabbitMQ
    """
    ssl_context = ssl.create_default_context(cafile=cert_file)
    if not verify:
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
    return ssl_context


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
        print(f"Loading GGUF model for {device} device...")
        model_path = model_path or "./Qwen3-32B-Q4_K_M.gguf"
        hf_model_name = hf_model_name or "Qwen/Qwen3-32B"

        # Temporarily suppress llama.cpp warnings
        os.environ['LLAMA_LOG_DISABLE'] = '1'
        model = Llama(
            model_path=model_path,
            n_gpu_layers=-1,  # Use all GPU layers
            n_ctx=10240,  # Context window size
            verbose=False
        )
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

        # Re-enable logging after model load
        if 'LLAMA_LOG_DISABLE' in os.environ:
            del os.environ['LLAMA_LOG_DISABLE']

        model_type = "llamacpp"
        print(f"Successfully loaded GGUF model: {model_path}")
        return model, tokenizer, model_type  # llama-cpp handles tokenization internally
    except ImportError:
        print("llama-cpp-python not available. Install llama-cpp-python for CUDA support.")
        raise
    except Exception as e:
        print(f"Error loading GGUF model: {e}")
        raise

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
                    print(token_text, end="", flush=True)
                    generated_text += token_text

                    # Check for finish reason (None, 'stop', 'length')
                    finish_reason = choice.get('finish_reason', None)
                    if finish_reason == 'stop':
                        print("EOS token detected, stopping generation.")
                        break
                    elif finish_reason == 'length':
                        print("Max tokens reached.")
                        break

            return generated_text.strip()
        except Exception as e:
            print(f"GGUF generation error: {e}")
            import traceback
            traceback.print_exc()
            return ""

    else:
        print(f"Unknown model type: {model_type}")
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