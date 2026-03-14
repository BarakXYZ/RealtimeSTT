from __future__ import annotations

import os
import time
from typing import Iterable

import faster_whisper
import numpy as np
import soundfile as sf
from faster_whisper import BatchedInferencePipeline

from ..interfaces import ASRBackendConfig, TranscriptMetadata, TranscriptResult, TranscriptSegment
from ..model_resolver import resolve_model_identifier


def _segment_time_to_ms(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value) * 1000)
    except (TypeError, ValueError):
        return None


class FasterWhisperBackend:
    def __init__(self, config: ASRBackendConfig):
        self.config = config
        self.model_path = resolve_model_identifier(config.model_id, config.download_root)
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model

        model = faster_whisper.WhisperModel(
            model_size_or_path=self.model_path,
            device=self.config.device,
            compute_type=self.config.compute_type,
            device_index=self.config.gpu_device_index,
            download_root=self.config.download_root,
        )
        if self.config.batch_size > 0:
            model = BatchedInferencePipeline(model=model)

        self._model = model
        return model

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        if not self.config.normalize_audio or audio.size == 0:
            return audio

        peak = np.max(np.abs(audio))
        if peak <= 0:
            return audio
        return (audio / peak) * 0.95

    def _build_result(self, segments: Iterable[object], info: object, elapsed_s: float) -> TranscriptResult:
        normalized_segments = [
            TranscriptSegment(
                text=getattr(segment, "text", ""),
                t0_ms=_segment_time_to_ms(getattr(segment, "start", None)),
                t1_ms=_segment_time_to_ms(getattr(segment, "end", None)),
            )
            for segment in segments
        ]
        text = " ".join(segment.text for segment in normalized_segments).strip()
        metadata = TranscriptMetadata(
            language=getattr(info, "language", None) if getattr(info, "language_probability", 0) > 0 else None,
            language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
            backend_name="faster-whisper",
            model_id=self.config.model_id,
            timings={"transcription_s": elapsed_s},
        )
        return TranscriptResult(text=text, segments=normalized_segments, metadata=metadata)

    def warmup(self) -> TranscriptResult:
        current_dir = os.path.dirname(os.path.realpath(__file__))
        warmup_audio_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "warmup_audio.wav")
        warmup_audio_data, _ = sf.read(warmup_audio_path, dtype="float32")
        return self.transcribe(warmup_audio_data, language="en", use_prompt=False)

    def transcribe(self, audio: np.ndarray, language: str | None = None, use_prompt: bool = True) -> TranscriptResult:
        model = self._load_model()
        normalized_audio = self._normalize_audio(audio)
        prompt = self.config.initial_prompt if use_prompt else None
        kwargs = {
            "language": language if language else None,
            "beam_size": self.config.beam_size,
            "initial_prompt": prompt,
            "suppress_tokens": self.config.suppress_tokens,
            "vad_filter": self.config.faster_whisper_vad_filter,
        }
        if self.config.batch_size > 0:
            kwargs["batch_size"] = self.config.batch_size

        start_t = time.time()
        segments, info = model.transcribe(normalized_audio, **kwargs)
        elapsed = time.time() - start_t
        return self._build_result(segments, info, elapsed)
