from .inference.transcribe import transcribe_with_faster_whisper
from .inference.transcribe import TranscriptionResult
from .utils.writer_utils import get_writer, log_result

def main():
    result = transcribe_with_faster_whisper(
        "transcribe-app/src/core/resources/bolt.m4a",
    )
    process_result(
        result=result,
        output_dir="results",
    )

def process_result(
        result: TranscriptionResult,
        output_dir: str = None,
        output_format: str = "all",
        highlight_words: bool = False,
        max_line_width: int = None,
        max_line_count: int = None,
    ):
    writer = get_writer(
        output_dir=output_dir,
        output_format=output_format,
    )

    # Result Debugging
    log_result(result, True)

    writer(
        result,
        "output-txt", {
            "highlight_words": highlight_words,
            "max_line_count": max_line_count,
            "max_line_width": max_line_width
        }
    )

if __name__ == "__main__":
    main()