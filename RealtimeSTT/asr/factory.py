from .backends.faster_whisper_backend import FasterWhisperBackend
from .interfaces import ASRBackend, ASRBackendConfig


def create_asr_backend(config: ASRBackendConfig) -> ASRBackend:
    if config.backend == "faster-whisper":
        return FasterWhisperBackend(config)

    raise ValueError(f"Unsupported ASR backend: {config.backend}")

