from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import soundfile as sf

from ...whisper_cpp_native import WhisperCppModel, require_whisper_cpp_native
from ..interfaces import ASRBackendConfig, TranscriptMetadata, TranscriptResult, TranscriptSegment
from ..model_resolver import resolve_coreml_encoder_path, resolve_model_identifier, running_on_apple_silicon


def _use_gpu_for_acceleration(acceleration: str) -> bool:
    if acceleration == "cpu":
        return False
    if acceleration == "auto":
        return True
    return acceleration in {"metal", "coreml", "cuda", "vulkan", "openvino"}


def _normalize_segments(segments: Iterable[object]) -> list[TranscriptSegment]:
    return [
        TranscriptSegment(
            text=getattr(segment, "text", ""),
            t0_ms=int(getattr(segment, "t0_ms", 0) or 0),
            t1_ms=int(getattr(segment, "t1_ms", 0) or 0),
        )
        for segment in segments
    ]


class WhisperCppBackend:
    def __init__(self, config: ASRBackendConfig):
        require_whisper_cpp_native()
        self.config = config
        model_identifier = config.whisper_cpp_model_path or config.model_id
        self.model_path = resolve_model_identifier(model_identifier, config.download_root, backend="whisper.cpp")
        expected_coreml_path = resolve_coreml_encoder_path(self.model_path, None)
        if config.whisper_cpp_coreml_encoder_path and config.whisper_cpp_coreml_encoder_path != expected_coreml_path:
            raise ValueError(
                "whisper.cpp Core ML currently relies on the official co-located "
                "'*-encoder.mlmodelc' layout next to the model file."
            )
        self.coreml_encoder_path = expected_coreml_path
        self._model = None
        self._state = None

    def _resolved_acceleration(self) -> str:
        if self.config.whisper_cpp_acceleration != "auto":
            return self.config.whisper_cpp_acceleration
        if running_on_apple_silicon():
            if self.coreml_encoder_path:
                return "coreml"
            return "metal"
        return "cpu"

    def _load_state(self):
        if self._state is not None:
            return self._state

        acceleration = self._resolved_acceleration()
        use_gpu = _use_gpu_for_acceleration(acceleration)
        self._model = WhisperCppModel(
            model_path=self.model_path,
            use_gpu=use_gpu,
            gpu_device=int(self.config.gpu_device_index if isinstance(self.config.gpu_device_index, int) else 0),
            flash_attn=False,
            openvino_encoder_path=self.config.whisper_cpp_openvino_encoder_path or "",
            openvino_device=self.config.whisper_cpp_openvino_device,
            openvino_cache_dir=self.config.whisper_cpp_openvino_cache_dir or "",
        )
        self._state = self._model.create_state()
        return self._state

    def warmup(self) -> TranscriptResult:
        current_dir = os.path.dirname(os.path.realpath(__file__))
        warmup_audio_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "warmup_audio.wav")
        warmup_audio_data, _ = sf.read(warmup_audio_path, dtype="float32")
        return self.transcribe(warmup_audio_data, language="en", use_prompt=False)

    def transcribe(self, audio: np.ndarray, language: str | None = None, use_prompt: bool = True) -> TranscriptResult:
        state = self._load_state()
        prompt = self.config.initial_prompt if use_prompt and self.config.initial_prompt else ""
        whisper_result = state.transcribe(
            audio=audio.astype(np.float32, copy=False),
            language=language or self.config.language or "",
            detect_language=not bool(language or self.config.language),
            initial_prompt=prompt,
            beam_size=self.config.beam_size,
            no_context=self.config.whisper_cpp_no_context,
            single_segment=self.config.whisper_cpp_single_segment,
            no_timestamps=False,
            token_timestamps=False,
            n_threads=self.config.whisper_cpp_threads or 0,
        )
        segments = _normalize_segments(whisper_result.segments)
        metadata = TranscriptMetadata(
            language=whisper_result.language or None,
            language_probability=float(getattr(whisper_result, "language_probability", 0.0) or 0.0),
            backend_name="whisper.cpp",
            model_id=self.model_path,
            timings={},
        )
        return TranscriptResult(
            text=whisper_result.text.strip(),
            segments=segments,
            metadata=metadata,
        )
