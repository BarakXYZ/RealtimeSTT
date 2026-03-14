from .backends.faster_whisper_backend import FasterWhisperBackend
from .backends.whisper_cpp_backend import WhisperCppBackend
from .interfaces import ASRBackend, ASRBackendConfig


def create_asr_backend(config: ASRBackendConfig) -> ASRBackend:
    if config.backend == "faster-whisper":
        return FasterWhisperBackend(config)
    if config.backend == "whisper.cpp":
        return WhisperCppBackend(config)

    raise ValueError(f"Unsupported ASR backend: {config.backend}")
