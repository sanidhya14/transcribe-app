from faster_whisper import WhisperModel
from ..models.transcription import TranscriptionInferenceOptions, TranscriptionResult, FasterWhisperModelOptions
from faster_whisper.vad import VadOptions, get_speech_timestamps, collect_chunks
from faster_whisper.transcribe import decode_audio, FeatureExtractor
from typing import BinaryIO, List, Union
import numpy as np
import threading
import multiprocessing
from multiprocessing import sharedctypes
import concurrent
from concurrent.futures import ThreadPoolExecutor

transcription_results: List[TranscriptionResult] = []

def transcribe(
        model: WhisperModel,
        inference_options: TranscriptionInferenceOptions 
):
    sampling_rate = FeatureExtractor().sampling_rate

    raw_audio = 0
    if not isinstance(inference_options.audio, np.ndarray):
        raw_audio = decode_audio(inference_options.audio, sampling_rate=sampling_rate)

    duration = raw_audio.shape[0] / sampling_rate
    print("Processing audio duration: " + str(duration))
    audio_batches = []
    # TODO: add a max batch size limit, default to that if limit exceeded
    audio_batches = chunk_audio_into_batches(
        audio=raw_audio,
        batch_size=inference_options.batch_size,
        vad_parameters=inference_options.vad_parameters,
    )

    print("Audio batched into: " + str(len(audio_batches)))

        # Create a ThreadPoolExecutor
    executor = ThreadPoolExecutor()

    tasks = []
    for audio in audio_batches:
        tasks.append(executor.submit(transcribe_with_faster_whisper(
            model=model,
            audio=audio,
            inference_options=inference_options,
        )))
    # Submit tasks for asynchronous execution

    # Wait for all tasks to complete
    concurrent.futures.wait(tasks)
    
    # threads: List[threading.Thread] = []
    # for audio in audio_batches:
    #     thread = threading.Thread(target=transcribe_with_faster_whisper(
    #         model=model,
    #         audio=audio,
    #         inference_options=inference_options,
    #     ))
    #     thread.start()
    #     threads.append(thread)

    # for thread in threads:
    #     thread.join()

    # print(transcription_results)




    # # Create a multiprocessing pool with the number of processes you desire
    # num_processes = 1  # You can set the desired number of processes here
    # pool = multiprocessing.Pool(processes=num_processes)

    # # Use the multiprocessing pool to execute the function with different audio values
    # results = []
    # for audio in audio_batches:
    #     results.append(pool.apply_async(transcribe_with_faster_whisper, (model_options, audio, inference_options)))

    # # Get the results from the pool
    # for result in results:
    #     transcribed_output = result.get()
    #     print(transcribed_output)
    #     # Process the transcribed output as desired

    # Close the pool
    # pool.close()
    # pool.join()

def chunk_audio_into_batches(
        audio: Union[str, BinaryIO, np.ndarray],
        batch_size: int,
        vad_parameters: TranscriptionInferenceOptions
) -> List[Union[str, BinaryIO, np.ndarray]]:
    if vad_parameters is None:
        vad_parameters = VadOptions()
    elif isinstance(vad_parameters, dict):
        vad_parameters = VadOptions(**vad_parameters)
    speech_chunks = get_speech_timestamps(audio, vad_parameters)
    speech_chunks = np.array(speech_chunks)
    print("array size" + str(speech_chunks))
    speech_chunks_aggregated_into_batches = np.array_split(speech_chunks, batch_size)
    print(speech_chunks_aggregated_into_batches)
    
    audio_chunks = []
    for aggregated_chunk in speech_chunks_aggregated_into_batches:
        audio_chunks.append(collect_chunks(audio, aggregated_chunk.tolist()))
    
    return audio_chunks
        # self.logger.info(
        #     "VAD filter removed %s of audio",
        #     format_timestamp(duration - (audio.shape[0] / sampling_rate)),
        # )

        # if self.logger.isEnabledFor(logging.DEBUG):
        #     self.logger.debug(
        #         "VAD filter kept the following audio segments: %s",
        #         ", ".join(
        #             "[%s -> %s]"
        #             % (
        #                 format_timestamp(chunk["start"] / sampling_rate),
        #                 format_timestamp(chunk["end"] / sampling_rate),
        #             )
        #             for chunk in speech_chunks
        #         ),
        #     )
    

def transcribe_with_faster_whisper(
        model: WhisperModel,
        audio: Union[str, BinaryIO, np.ndarray],
        inference_options: TranscriptionInferenceOptions
    ) -> TranscriptionResult:
    segments, info = model.transcribe(
        audio,
        language=inference_options.language,
        task=inference_options.task,
        best_of=inference_options.best_of,
        beam_size=inference_options.beam_size,
        patience=inference_options.patience,
        length_penalty=inference_options.length_penalty,
        suppress_tokens=inference_options.suppress_tokens,
        initial_prompt=inference_options.initial_prompt,
        condition_on_previous_text=inference_options.condition_on_previous_text,
        compression_ratio_threshold=inference_options.compression_ratio_threshold,
        log_prob_threshold=inference_options.log_prob_threshold,
        no_speech_threshold=inference_options.no_speech_threshold,
        word_timestamps=inference_options.word_timestamps,
        prepend_punctuations=inference_options.prepend_punctuations,
        append_punctuations=inference_options.append_punctuations,
        vad_filter=inference_options.vad_filter,
        vad_parameters=inference_options.vad_parameters,
    )
    # The transcription will actually run here
    segments = list(segments)
    #transcription_results.append(TranscriptionResult(segments=segments, info=info))
    #return TranscriptionResult(segments=segments, info=info)
    print(TranscriptionResult(segments=segments, info=info))

def get_model_instance(
        model_options: FasterWhisperModelOptions
    ) -> WhisperModel:
    
    return WhisperModel(
        model_size_or_path=model_options.model_size,
        device=model_options.device,
        device_index=model_options.device_index,
        compute_type=model_options.compute_type,
        cpu_threads=model_options.cpu_threads,
        num_workers=model_options.num_workers,
        download_root=model_options.download_root,
        local_files_only=model_options.local_files_only,
    )
