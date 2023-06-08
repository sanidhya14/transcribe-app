from typing import BinaryIO, List, NamedTuple, Optional, Union, Iterable
import numpy as np
from faster_whisper.transcribe import Segment, TranscriptionInfo
from faster_whisper.vad import VadOptions

#See Args here: https://github.com/guillaumekln/faster-whisper/blob/d4222da952fde2aa4064aad820e207d0c7a9de75/faster_whisper/transcribe.py#LL90C11-L90C29
class FasterWhisperModelOptions(NamedTuple):
    model_size: str = "small"
    device: str = "cpu"
    device_index: Union[int, List[int]] = 0
    compute_type: str = "int8"
    cpu_threads: int = 0
    num_workers: int = 1
    download_root: Optional[str] = None
    # TODO: Update to fetch model only which is locally stored.
    # Custom impl will be done to fetch from self managed remote & not HuggingFace
    local_files_only: bool = False

# See Args from: https://github.com/guillaumekln/faster-whisper/blob/d4222da952fde2aa4064aad820e207d0c7a9de75/faster_whisper/transcribe.py#L187
class TranscriptionInferenceOptions(NamedTuple):
    audio: Union[str, BinaryIO, np.ndarray]
    language: str = "en"
    task: str = "transcribe"
    best_of: int = 5
    beam_size: int = 5
    patience: float = 1
    length_penalty: float = 1
    suppress_tokens: Optional[List[int]] = [-1]
    initial_prompt: str = None
    condition_on_previous_text: bool = True
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    word_timestamps: bool = True
    prepend_punctuations: str = "\"\'“¿([{-"
    append_punctuations: str = "\"\'.。,，!！?？:：”)]}、"
    vad_filter: bool = False
    vad_parameters: dict = None

class TransciptionOutputOptions(NamedTuple):
    output_file_name: str
    output_dir: str
    output_format: str = "all"
    highlight_words: bool = False
    max_line_width: int = None
    max_line_count: int = None

class TranscriptionRequest(NamedTuple):
    model_options: FasterWhisperModelOptions
    inference_options: TranscriptionInferenceOptions
    output_options: TransciptionOutputOptions

class TranscriptionResult(NamedTuple):
    segments: Iterable[Segment]
    info: TranscriptionInfo