import torch
from .utils import (
    get_writer
)
import whisper

def transcribe_with_whisper():
    # Properties picked up from: https://github.com/openai/whisper/blob/main/whisper/transcribe.py
    audio_file = "transcribe-app/src/core/resources/bolt.m4a"
    model_options = {
        "name": "tiny",
        "download_root": None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "in_memory": False
    }
    primary_transcribe_options = {
        "temperature": 0,
        # Add logic for this
        "temperature_increment_on_fallback": 0.2
    }
    transcribe_options = {
        "language": "en",
        "verbose": True,
        "task": "transcribe",
        "language": "en",
        "best_of": 5,
        "beam_size": 5,
        "patience": None,
        "length_penalty": None,
        "suppress_tokens": "-1",
        "initial_prompt": None,
        "condition_on_previous_text": True,
        "fp16": True,
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "word_timestamps": True,
        "prepend_punctuations": "\"\'“¿([{-",
        "append_punctuations": "\"\'.。,，!！?？:：”)]}、"
    }
    writer_options = {
        "output_dir": "transcribe-app/src/core/resources",
        "output_format": "all"
    }
    writer_preferences = {
        "highlight_words": False,
        "max_line_width": None,
        "max_line_count": None
    }

    # Transcription execution
    writer = get_writer(**writer_options)
    model = whisper.load_model(**model_options)
    result = model.transcribe(audio_file, **transcribe_options)

    #Output Results
    writer(result, 'generated-text', writer_preferences)
