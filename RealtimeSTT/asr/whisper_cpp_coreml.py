from __future__ import annotations

import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger(__name__)
COREML_CONVERSION_MODULE = "RealtimeSTT.asr.resources.whisper_cpp.convert_whisper_to_coreml"

WHISPER_CPP_COREML_SUPPORTED_MODELS = {
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "small.en-tdrz",
    "medium",
    "medium.en",
    "large-v1",
    "large-v2",
    "large-v3",
    "large-v3-turbo",
}
COREML_FAILURE_RETRY_SECONDS = 6 * 60 * 60
COREML_LOCK_STALE_SECONDS = 2 * 60 * 60
COREML_GENERATION_TIMEOUT_SECONDS = 60 * 60
COREML_REQUIRED_MODULES = ("torch", "coremltools", "whisper", "ane_transformers")


@dataclass(frozen=True)
class WhisperCppCoreMLRequest:
    model_name: str
    model_path: str
    target_path: str


_scheduled_jobs: set[str] = set()
_schedule_lock = threading.Lock()


def _running_on_apple_silicon() -> bool:
    return os.uname().sysname == "Darwin" and os.uname().machine == "arm64"


def _derive_model_name(model_identifier: str | None, model_path: str) -> str | None:
    candidates = []
    if model_identifier:
        candidates.append(Path(model_identifier).name)
        candidates.append(model_identifier)

    model_filename = Path(model_path).name
    candidates.append(model_filename)
    if model_filename.startswith("ggml-") and model_filename.endswith(".bin"):
        candidates.append(model_filename[len("ggml-"):-len(".bin")])
    if model_filename.startswith("gguf-") and model_filename.endswith(".gguf"):
        candidates.append(model_filename[len("gguf-"):-len(".gguf")])
    if model_filename.endswith(".gguf"):
        candidates.append(model_filename[:-len(".gguf")])

    for candidate in candidates:
        normalized = candidate.strip()
        if normalized in WHISPER_CPP_COREML_SUPPORTED_MODELS:
            return normalized
    return None


def _lock_path(target_path: Path) -> Path:
    return target_path.parent / f".{target_path.name}.lock"


def _failure_marker_path(target_path: Path) -> Path:
    return target_path.parent / f".{target_path.name}.failed.json"


def _read_failure_marker(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _recent_failure(path: Path) -> bool:
    marker = _read_failure_marker(path)
    if not marker:
        return False
    timestamp = float(marker.get("timestamp", 0))
    return (time.time() - timestamp) < COREML_FAILURE_RETRY_SECONDS


def _write_failure_marker(path: Path, error_message: str) -> None:
    payload = {
        "timestamp": time.time(),
        "error": error_message,
    }
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        logger.debug("Failed to write whisper.cpp Core ML failure marker: %s", path, exc_info=True)


def _clear_failure_marker(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.debug("Failed to remove whisper.cpp Core ML failure marker: %s", path, exc_info=True)


def _acquire_lock(lock_path: Path) -> bool:
    try:
        lock_path.mkdir()
        return True
    except FileExistsError:
        try:
            age_seconds = time.time() - lock_path.stat().st_mtime
        except OSError:
            return False
        if age_seconds > COREML_LOCK_STALE_SECONDS:
            logger.warning("Removing stale whisper.cpp Core ML generation lock: %s", lock_path)
            shutil.rmtree(lock_path, ignore_errors=True)
            try:
                lock_path.mkdir()
                return True
            except FileExistsError:
                return False
        return False


def _release_lock(lock_path: Path) -> None:
    shutil.rmtree(lock_path, ignore_errors=True)


def _missing_runtime_dependencies() -> list[str]:
    return [name for name in COREML_REQUIRED_MODULES if importlib.util.find_spec(name) is None]


def _run_subprocess(command: list[str], cwd: Path) -> None:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        check=True,
        text=True,
        capture_output=True,
        timeout=COREML_GENERATION_TIMEOUT_SECONDS,
    )
    if completed.stdout:
        logger.info(completed.stdout.strip())


def _generate_coreml_encoder_sync(request: WhisperCppCoreMLRequest) -> None:
    target_path = Path(request.target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if importlib.util.find_spec(COREML_CONVERSION_MODULE) is None:
        raise FileNotFoundError(
            "whisper.cpp Core ML conversion module is missing: "
            f"{COREML_CONVERSION_MODULE}"
        )
    if shutil.which("xcrun") is None:
        raise FileNotFoundError("xcrun is required for whisper.cpp Core ML generation")

    missing_modules = _missing_runtime_dependencies()
    if missing_modules:
        raise RuntimeError(
            "Missing whisper.cpp Core ML runtime dependencies: "
            + ", ".join(missing_modules)
        )

    with tempfile.TemporaryDirectory(prefix=f".coreml-{request.model_name}-", dir=target_path.parent) as tmp_dir:
        tmp_root = Path(tmp_dir)
        models_dir = tmp_root / "models"
        compiled_dir = tmp_root / "compiled"
        models_dir.mkdir(parents=True, exist_ok=True)
        compiled_dir.mkdir(parents=True, exist_ok=True)

        _run_subprocess(
            [
                sys.executable,
                "-m",
                COREML_CONVERSION_MODULE,
                "--model",
                request.model_name,
                "--encoder-only",
                "True",
                "--optimize-ane",
                "True",
            ],
            cwd=tmp_root,
        )

        mlpackage_path = models_dir / f"coreml-encoder-{request.model_name}.mlpackage"
        if not mlpackage_path.exists():
            raise FileNotFoundError(f"Expected Core ML package was not generated: {mlpackage_path}")

        _run_subprocess(
            [
                "xcrun",
                "coremlc",
                "compile",
                str(mlpackage_path),
                str(compiled_dir),
            ],
            cwd=tmp_root,
        )

        compiled_bundle = compiled_dir / f"coreml-encoder-{request.model_name}.mlmodelc"
        if not compiled_bundle.exists():
            raise FileNotFoundError(f"Expected compiled Core ML bundle is missing: {compiled_bundle}")

        temp_target = target_path.parent / f".{target_path.name}.tmp-{uuid.uuid4().hex}"
        shutil.rmtree(temp_target, ignore_errors=True)
        shutil.move(str(compiled_bundle), str(temp_target))
        if target_path.exists():
            shutil.rmtree(target_path, ignore_errors=True)
        os.replace(temp_target, target_path)


def schedule_whisper_cpp_coreml_generation(
    model_identifier: str | None,
    model_path: str,
    target_path: str,
) -> bool:
    if not _running_on_apple_silicon():
        return False

    model_name = _derive_model_name(model_identifier, model_path)
    if not model_name:
        return False

    request = WhisperCppCoreMLRequest(
        model_name=model_name,
        model_path=model_path,
        target_path=target_path,
    )
    target = Path(target_path)
    if target.exists():
        return False

    failure_marker = _failure_marker_path(target)
    if _recent_failure(failure_marker):
        logger.info(
            "Skipping whisper.cpp Core ML generation for '%s' due to a recent failure marker. "
            "Remove '%s' or wait to retry.",
            model_name,
            failure_marker,
        )
        return False

    with _schedule_lock:
        if target_path in _scheduled_jobs:
            return False
        _scheduled_jobs.add(target_path)

    def _worker() -> None:
        lock_path = _lock_path(target)
        if not _acquire_lock(lock_path):
            with _schedule_lock:
                _scheduled_jobs.discard(target_path)
            return

        try:
            logger.info(
                "Starting background whisper.cpp Core ML generation for '%s' at '%s'.",
                request.model_name,
                request.target_path,
            )
            _generate_coreml_encoder_sync(request)
            _clear_failure_marker(failure_marker)
            logger.info(
                "Completed whisper.cpp Core ML generation for '%s'.",
                request.model_name,
            )
        except Exception as exc:
            logger.warning(
                "whisper.cpp Core ML generation failed for '%s': %s",
                request.model_name,
                exc,
            )
            _write_failure_marker(failure_marker, str(exc))
        finally:
            _release_lock(lock_path)
            with _schedule_lock:
                _scheduled_jobs.discard(target_path)

    thread = threading.Thread(
        target=_worker,
        name=f"whisper-cpp-coreml-{request.model_name}",
        daemon=True,
    )
    thread.start()
    return True
