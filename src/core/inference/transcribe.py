from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment, TranscriptionInfo
from typing import BinaryIO, List, NamedTuple, Optional, Union, Iterable
import numpy as np

# See Args from: https://github.com/guillaumekln/faster-whisper/blob/d4222da952fde2aa4064aad820e207d0c7a9de75/faster_whisper/transcribe.py#L187
class TranscriptionResult(NamedTuple):
    segments: Iterable[Segment]
    info: TranscriptionInfo

def transcribe_with_faster_whisper(
        audio: Union[str, BinaryIO, np.ndarray],
        language: str = "en",
        task: str = "transcribe",
        best_of: int = 5,
        beam_size: int = 5,
        patience: float = 1,
        length_penalty: float = 1,
        suppress_tokens: Optional[List[int]] = [-1],
        initial_prompt: str = None,
        condition_on_previous_text: bool = True,
        compression_ratio_threshold: float = 2.4,
        log_prob_threshold: float = -1.0,
        no_speech_threshold: float = 0.6,
        word_timestamps: bool = True,
        prepend_punctuations: str = "\"\'“¿([{-",
        append_punctuations: str = "\"\'.。,，!！?？:：”)]}、",
    ) -> TranscriptionResult:

    model = get_model_instance()
    segments, info = model.transcribe(
        audio,
        language=language,
        task=task,
        best_of=best_of,
        beam_size=beam_size,
        patience=patience,
        length_penalty=length_penalty,
        suppress_tokens=suppress_tokens,
        initial_prompt=initial_prompt,
        condition_on_previous_text=condition_on_previous_text,
        compression_ratio_threshold=compression_ratio_threshold,
        log_prob_threshold=log_prob_threshold,
        no_speech_threshold=no_speech_threshold,
        word_timestamps=word_timestamps,
        prepend_punctuations=prepend_punctuations,
        append_punctuations=append_punctuations,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=2000),
    )
    # The transcription will actually run here
    segments = list(segments)
    return TranscriptionResult(segments=segments, info=info)

#See Args here: https://github.com/guillaumekln/faster-whisper/blob/d4222da952fde2aa4064aad820e207d0c7a9de75/faster_whisper/transcribe.py#LL90C11-L90C29
def get_model_instance(
        model_size: str = "small",
        device: str = "cpu",
        device_index: Union[int, List[int]] = 0,
        compute_type: str = "int8",
        cpu_threads: int = 0,
        num_workers: int = 1,
        download_root: Optional[str] = None,
        # TODO: Update to fetch model only which is locally stored.
        # Custom impl will be done to fetch from self managed remote & not HuggingFace
        local_files_only: bool = False,
):

    return WhisperModel(
        model_size_or_path=model_size,
        device=device,
        device_index=device_index,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        num_workers=num_workers,
        download_root=download_root,
        local_files_only=local_files_only,
    )

# # TODO: Remove this native whisper implementation after stable release.
# def transcribe_with_whisper():
#     # Properties picked up from: https://github.com/openai/whisper/blob/main/whisper/transcribe.py
#     audio_file = "transcribe-app/src/core/resources/bolt.m4a"
#     model_options = {
#         "name": "tiny",
#         "download_root": None,
#         "device": "cuda" if torch.cuda.is_available() else "cpu",
#         "in_memory": False
#     }
#     primary_transcribe_options = {
#         "temperature": 0,
#         # Add logic for this
#         "temperature_increment_on_fallback": 0.2
#     }
#     transcribe_options = {
#         "language": "en",
#         "verbose": True,
#         "task": "transcribe",
#         "language": "en",
#         "best_of": 5,
#         "beam_size": 5,
#         "patience": None,
#         "length_penalty": None,
#         "suppress_tokens": "-1",
#         "initial_prompt": None,
#         "condition_on_previous_text": True,
#         "fp16": True,
#         "compression_ratio_threshold": 2.4,
#         "logprob_threshold": -1.0,
#         "no_speech_threshold": 0.6,
#         "word_timestamps": True,
#         "prepend_punctuations": "\"\'“¿([{-",
#         "append_punctuations": "\"\'.。,，!！?？:：”)]}、"
#     }
#     writer_options = {
#         "output_dir": "transcribe-app/src/core/resources",
#         "output_format": "all"
#     }
#     writer_preferences = {
#         "highlight_words": False,
#         "max_line_width": None,
#         "max_line_count": None
#     }

#     # Transcription execution
#     writer = get_writer(**writer_options)
#     model = whisper.load_model(**model_options)
#     result = model.transcribe(audio_file, **transcribe_options)

#     #Output Results
#     writer(result, 'generated-text', writer_preferences)
