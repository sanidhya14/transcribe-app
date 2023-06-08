from .inference.transcribe import get_model_instance, transcribe_with_faster_whisper, TranscriptionResult
from .utils.writer_utils import get_writer, log_result
from .models.transcription import TransciptionOutputOptions, TranscriptionRequest, FasterWhisperModelOptions, TranscriptionInferenceOptions

def main(request: TranscriptionRequest):
    model = get_model_instance(
        request.model_options
    )
    result = transcribe_with_faster_whisper(
        model,
        request.inference_options,
    )
    process_transcription_result(
        result,
        request.output_options,
    )

def process_transcription_result(
        result: TranscriptionResult,
        output_options: TransciptionOutputOptions,
    ):
    writer = get_writer(
        output_dir=output_options.output_dir,
        output_format=output_options.output_format,
    )

    # Result Debugging
    log_result(result, True)

    writer(
        result,
        output_options.output_file_name,
        output_options._asdict(),
    )

if __name__ == "__main__":
    main(TranscriptionRequest(
        model_options=FasterWhisperModelOptions(
            model_size="small",
            device="cpu",
        ),
        inference_options=TranscriptionInferenceOptions(
            audio="bolt.m4a",
        ),
        output_options=TransciptionOutputOptions(
            output_file_name="result-v2",
            output_dir="results",
        ),
    ))