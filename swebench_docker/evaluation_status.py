"""Atomic checkpoints for individual Docker evaluation settings."""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def evaluation_log_path(log_dir, task_id, model, setting):
    return Path(log_dir) / f"{task_id}.{model}.{setting}.eval.log"


def evaluation_status_path(log_dir, task_id, model, setting):
    return Path(log_dir) / f"{task_id}.{model}.{setting}.eval.status.json"


def _log_fingerprint(path):
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def write_evaluation_status(log_dir, task_id, model, setting, status, **details):
    """Write a setting checkpoint without exposing a partially written JSON file.

    A completed checkpoint requires the corresponding evaluation log and records
    its fingerprint. Test assertion failures are valid completed evaluations;
    container or setup failures must use ``failed`` instead.
    """
    if status not in {"running", "completed", "failed", "interrupted"}:
        raise ValueError(f"Unknown evaluation status: {status}")
    path = evaluation_status_path(log_dir, task_id, model, setting)
    log_path = evaluation_log_path(log_dir, task_id, model, setting)
    metadata = {
        **details,
        "version": 1,
        "id": task_id,
        "model": model,
        "setting": setting,
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        metadata["log_fingerprint"] = _log_fingerprint(log_path)
    except FileNotFoundError:
        if status == "completed":
            raise
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as output:
            temporary_path = output.name
            json.dump(metadata, output, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return metadata


def evaluation_is_complete(log_dir, task_id, model, setting):
    """Return whether a completed checkpoint still matches this setting's log."""
    path = evaluation_status_path(log_dir, task_id, model, setting)
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if (
            metadata.get("version") != 1
            or metadata.get("id") != task_id
            or metadata.get("model") != model
            or metadata.get("setting") != setting
            or metadata.get("status") != "completed"
        ):
            return False
        fingerprint = _log_fingerprint(
            evaluation_log_path(log_dir, task_id, model, setting)
        )
        return fingerprint["size"] > 0 and metadata.get("log_fingerprint") == fingerprint
    except (OSError, ValueError, TypeError, AttributeError):
        return False
