from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

import numpy as np


@dataclass(frozen=True)
class TranscriptSegment:
    text: str
    t0_ms: Optional[int] = None
    t1_ms: Optional[int] = None


@dataclass(frozen=True)
class TranscriptMetadata:
    language: Optional[str]
    language_probability: float
    backend_name: str
    model_id: str
    timings: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TranscriptResult:
    text: str
    segments: List[TranscriptSegment]
    metadata: TranscriptMetadata


@dataclass(frozen=True)
class ASRBackendConfig:
    model_id: str
    backend: str = "faster-whisper"
    download_root: Optional[str] = None
    language: str = ""
    device: str = "cpu"
    compute_type: str = "default"
    gpu_device_index: int | list[int] = 0
    beam_size: int = 5
    initial_prompt: Optional[str] = None
    suppress_tokens: Optional[list[int]] = None
    batch_size: int = 0
    faster_whisper_vad_filter: bool = False
    normalize_audio: bool = False
    whisper_cpp_threads: Optional[int] = None
    whisper_cpp_acceleration: str = "auto"
    whisper_cpp_model_path: Optional[str] = None
    whisper_cpp_coreml_encoder_path: Optional[str] = None
    whisper_cpp_auto_generate_coreml: bool = True
    whisper_cpp_openvino_encoder_path: Optional[str] = None
    whisper_cpp_openvino_device: str = "CPU"
    whisper_cpp_openvino_cache_dir: Optional[str] = None
    whisper_cpp_no_context: bool = True
    whisper_cpp_single_segment: bool = False
    whisper_cpp_stream_length_ms: int = 5000
    whisper_cpp_stream_step_ms: int = 700
    whisper_cpp_stream_keep_ms: int = 200


class ASRBackend(Protocol):
    def warmup(self) -> TranscriptResult:
        ...

    def transcribe(self, audio: np.ndarray, language: Optional[str] = None, use_prompt: bool = True) -> TranscriptResult:
        ...

    def abort(self) -> None:
        ...
