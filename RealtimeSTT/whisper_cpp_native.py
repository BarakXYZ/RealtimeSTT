from __future__ import annotations


try:
    from ._whisper_cpp_native import WhisperCppModel, WhisperCppState, WhisperSegment, WhisperTranscription
except ImportError as exc:  # pragma: no cover - exercised in runtime envs without native build
    _NATIVE_IMPORT_ERROR = exc
    WhisperCppModel = None
    WhisperCppState = None
    WhisperSegment = None
    WhisperTranscription = None
else:
    _NATIVE_IMPORT_ERROR = None


def require_whisper_cpp_native() -> None:
    if _NATIVE_IMPORT_ERROR is not None:
        raise RuntimeError(
            "whisper.cpp native extension is not available. "
            "Build the package with the native extension enabled before using backend='whisper.cpp'."
        ) from _NATIVE_IMPORT_ERROR

