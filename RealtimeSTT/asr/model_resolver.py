from __future__ import annotations

from pathlib import Path


def resolve_model_identifier(model_id: str, download_root: str | None = None) -> str:
    if not model_id:
        raise ValueError("model_id must not be empty")

    if download_root:
        candidate = Path(download_root) / model_id
        if candidate.exists():
            return str(candidate)

    return model_id

