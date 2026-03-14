from __future__ import annotations

import hashlib
import os
import shutil
import time
import urllib.request
import uuid
from pathlib import Path


WHISPER_CPP_MODEL_BASE_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
WHISPER_CPP_KNOWN_MODELS = {
    "tiny",
    "tiny.en",
    "tiny-q5_1",
    "tiny.en-q5_1",
    "tiny-q8_0",
    "base",
    "base.en",
    "base-q5_1",
    "base.en-q5_1",
    "base-q8_0",
    "small",
    "small.en",
    "small.en-tdrz",
    "small-q5_1",
    "small.en-q5_1",
    "small-q8_0",
    "medium",
    "medium.en",
    "medium-q5_0",
    "medium.en-q5_0",
    "medium-q8_0",
    "large-v1",
    "large-v2",
    "large-v2-q5_0",
    "large-v2-q8_0",
    "large-v3",
    "large-v3-q5_0",
    "large-v3-turbo",
    "large-v3-turbo-q5_0",
    "large-v3-turbo-q8_0",
}
WHISPER_CPP_KNOWN_HASHES = {
    "tiny": "bd577a113a864445d4c299885e0cb97d4ba92b5f",
    "tiny.en": "c78c86eb1a8faa21b369bcd33207cc90d64ae9df",
    "base": "465707469ff3a37a2b9b8d8f89f2f99de7299dac",
    "base.en": "137c40403d78fd54d454da0f9bd998f78703390c",
    "small": "55356645c2b361a969dfd0ef2c5a50d530afd8d5",
    "small.en": "db8a495a91d927739e50b3fc1cc4c6b8f6c2d022",
    "small.en-tdrz": "b6c6e7e89af1a35c08e6de56b66ca6a02a2fdfa1",
    "medium": "fd9727b6e1217c2f614f9b698455c4ffd82463b4",
    "medium.en": "8c30f0e44ce9560643ebd10bbe50cd20eafd3723",
    "large-v1": "b1caaf735c4cc1429223d5a74f0f4d0b9b59a299",
    "large-v2": "0f4c8e34f21cf1a914c59d8b3ce882345ad349d6",
    "large-v2-q5_0": "00e39f2196344e901b3a2bd5814807a769bd1630",
    "large-v3": "ad82bf6a9043ceed055076d0fd39f5f186ff8062",
    "large-v3-q5_0": "e6e2ed78495d403bef4b7cff42ef4aaadcfea8de",
    "large-v3-turbo": "4af2b29d7ec73d781377bfd1758ca957a807e941",
    "large-v3-turbo-q5_0": "e050f7970618a659205450ad97eb95a18d69c9ee",
}
WHISPER_CPP_MODEL_LOCK_STALE_SECONDS = 60 * 60
WHISPER_CPP_MODEL_LOCK_TIMEOUT_SECONDS = 10 * 60
WHISPER_CPP_MODEL_LOCK_POLL_SECONDS = 0.25


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _default_model_cache(download_root: str | None) -> Path:
    if download_root:
        return Path(download_root)
    return Path.home() / ".cache" / "RealtimeSTT" / "whisper.cpp"


def _model_lock_path(model_path: Path) -> Path:
    return model_path.parent / f".{model_path.name}.lock"


def _acquire_model_lock(lock_path: Path) -> None:
    deadline = time.time() + WHISPER_CPP_MODEL_LOCK_TIMEOUT_SECONDS
    while True:
        try:
            lock_path.mkdir()
            return
        except FileExistsError:
            try:
                age_seconds = time.time() - lock_path.stat().st_mtime
            except OSError:
                age_seconds = 0
            if age_seconds > WHISPER_CPP_MODEL_LOCK_STALE_SECONDS:
                shutil.rmtree(lock_path, ignore_errors=True)
                continue
            if time.time() >= deadline:
                raise TimeoutError(f"Timed out waiting for whisper.cpp model lock: {lock_path}")
            time.sleep(WHISPER_CPP_MODEL_LOCK_POLL_SECONDS)


def _release_model_lock(lock_path: Path) -> None:
    shutil.rmtree(lock_path, ignore_errors=True)


def resolve_model_identifier(model_id: str, download_root: str | None = None, backend: str = "faster-whisper") -> str:
    if not model_id:
        raise ValueError("model_id must not be empty")

    direct_path = Path(model_id).expanduser()
    if direct_path.exists():
        return str(direct_path)

    if backend != "whisper.cpp":
        if download_root:
            candidate = Path(download_root) / model_id
            if candidate.exists():
                return str(candidate)
        return model_id

    if model_id not in WHISPER_CPP_KNOWN_MODELS:
        return model_id

    model_cache = _default_model_cache(download_root)
    model_cache.mkdir(parents=True, exist_ok=True)
    model_path = model_cache / f"ggml-{model_id}.bin"
    expected_sha = WHISPER_CPP_KNOWN_HASHES.get(model_id)
    lock_path = _model_lock_path(model_path)
    _acquire_model_lock(lock_path)

    temp_path: Path | None = None
    try:
        if model_path.exists():
            if expected_sha and _sha1(model_path) != expected_sha:
                model_path.unlink(missing_ok=True)
            else:
                return str(model_path)

        url = f"{WHISPER_CPP_MODEL_BASE_URL}/ggml-{model_id}.bin"
        temp_path = model_cache / f".{model_path.name}.tmp-{uuid.uuid4().hex}"
        urllib.request.urlretrieve(url, temp_path)

        if expected_sha and _sha1(temp_path) != expected_sha:
            temp_path.unlink(missing_ok=True)
            raise ValueError(f"Checksum verification failed for whisper.cpp model '{model_id}'")

        os.replace(temp_path, model_path)
        temp_path = None
        return str(model_path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        _release_model_lock(lock_path)


def resolve_coreml_encoder_path(
    model_path: str,
    explicit_path: str | None = None,
    model_identifier: str | None = None,
    auto_generate: bool = False,
) -> str | None:
    if explicit_path:
        explicit = Path(explicit_path).expanduser()
        if explicit.exists():
            return str(explicit)
        raise FileNotFoundError(f"Explicit whisper.cpp Core ML encoder path does not exist: {explicit}")

    candidate = Path(model_path).with_suffix("")
    candidate = candidate.parent / f"{candidate.name}-encoder.mlmodelc"
    if candidate.exists():
        return str(candidate)

    if auto_generate:
        from .whisper_cpp_coreml import schedule_whisper_cpp_coreml_generation

        schedule_whisper_cpp_coreml_generation(
            model_identifier=model_identifier,
            model_path=model_path,
            target_path=str(candidate),
        )

    return None


def running_on_apple_silicon() -> bool:
    return os.uname().sysname == "Darwin" and os.uname().machine == "arm64"
