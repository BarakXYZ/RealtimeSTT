"""Small benchmark harness for recorder-level ASR latency comparisons."""

from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path

import numpy as np


def _load_audio(path: Path) -> np.ndarray:
    import soundfile as sf

    audio, sample_rate = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        raise ValueError(f"Expected 16kHz audio, got {sample_rate}Hz")
    return audio.astype(np.float32)


def benchmark_once(audio: np.ndarray, backend: str, model: str, realtime_model: str) -> dict:
    from RealtimeSTT import AudioToTextRecorder

    kwargs = {
        "use_microphone": False,
        "model": model,
        "realtime_model_type": realtime_model,
        "spinner": False,
        "enable_realtime_transcription": True,
    }
    if "backend" in inspect.signature(AudioToTextRecorder.__init__).parameters:
        kwargs["backend"] = backend

    recorder = AudioToTextRecorder(**kwargs)
    try:
        start = time.perf_counter()
        recorder.feed_audio((audio * 32767.0).astype(np.int16).tobytes())
        final = recorder.text()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {
            "backend": backend,
            "model": model,
            "realtime_model": realtime_model,
            "elapsed_ms": round(elapsed_ms, 2),
            "text": final,
        }
    finally:
        recorder.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark RealtimeSTT ASR backends")
    parser.add_argument("audio", type=Path, help="16kHz mono float-compatible wav file")
    parser.add_argument("--backend", default="faster-whisper")
    parser.add_argument("--model", default="tiny")
    parser.add_argument("--realtime-model", default="tiny")
    args = parser.parse_args()

    result = benchmark_once(_load_audio(args.audio), args.backend, args.model, args.realtime_model)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
