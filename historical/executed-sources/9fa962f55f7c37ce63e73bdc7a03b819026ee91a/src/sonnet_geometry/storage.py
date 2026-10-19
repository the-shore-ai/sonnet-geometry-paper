"""Immutable local runs and create-only, hash-validated GCS persistence."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any
import uuid


class IntegrityError(RuntimeError):
    """A run is incomplete, altered, or inconsistent with its checkpoint."""


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in {".", ".."}:
        raise ValueError(f"Unsafe artifact or run name: {name!r}")
    return name


def write_exclusive(path: Path, data: bytes) -> None:
    """Atomically publish a complete file without ever replacing an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)  # Atomic create-only publication on the same filesystem.
    finally:
        temporary.unlink(missing_ok=True)


class RunStore:
    """One immutable sonnet run. A failed run receives no completion checkpoint."""

    def __init__(self, output_dir: str | Path, run_id: str | None = None):
        self.run_id = _safe_name(run_id or uuid.uuid4().hex)
        self.path = Path(output_dir) / self.run_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.artifacts: dict[str, dict[str, Any]] = {}

    def write_bytes(self, name: str, data: bytes) -> Path:
        name = _safe_name(name)
        if name in {"manifest.json", "checkpoint.json"}:
            raise ValueError("Manifest and checkpoint names are reserved")
        path = self.path / name
        write_exclusive(path, data)
        self.artifacts[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        return path

    def write_json(self, name: str, value: Any) -> Path:
        return self.write_bytes(name, canonical_json(value))

    def write_tensors(self, tensors: dict[str, Any], metadata: dict[str, str] | None = None) -> Path:
        from safetensors.torch import save

        # safetensors preserves tensor dtype and all bits; no conversion/quantization.
        return self.write_bytes("activations.safetensors", save(tensors, metadata=metadata))

    def complete(self, manifest: dict[str, Any]) -> dict[str, Any]:
        final = {**manifest, "run_id": self.run_id, "artifacts": self.artifacts}
        write_exclusive(self.path / "manifest.json", canonical_json(final))
        checkpoint = {
            "schema_version": 1,
            "status": "complete",
            "run_id": self.run_id,
            "manifest_sha256": sha256_file(self.path / "manifest.json"),
        }
        write_exclusive(self.path / "checkpoint.json", canonical_json(checkpoint))
        verify_run(self.path)
        return final


def verify_run(run_dir: str | Path) -> dict[str, Any]:
    """Resume is permitted only after every artifact and manifest hash validates."""
    path = Path(run_dir)
    try:
        checkpoint = json.loads((path / "checkpoint.json").read_text())
        if checkpoint.get("status") != "complete":
            raise IntegrityError("Checkpoint is not complete")
        if sha256_file(path / "manifest.json") != checkpoint["manifest_sha256"]:
            raise IntegrityError("Manifest checksum mismatch")
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest["run_id"] != checkpoint["run_id"]:
            raise IntegrityError("Run identity mismatch")
        if not manifest.get("artifacts"):
            raise IntegrityError("Run has no artifacts")
        for name, expected in manifest["artifacts"].items():
            artifact = path / _safe_name(name)
            if artifact.is_symlink() or artifact.stat().st_size != expected["bytes"]:
                raise IntegrityError(f"Artifact size or type mismatch: {name}")
            if sha256_file(artifact) != expected["sha256"]:
                raise IntegrityError(f"Artifact checksum mismatch: {name}")
        return manifest
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise IntegrityError(f"Incomplete or invalid run: {path}") from exc


def _remote_matches(blob: Any, expected_sha256: str, expected_size: int) -> bool:
    """Validate actual remote bytes, pinned to the observed object generation."""
    blob.reload()
    if int(blob.size) != expected_size:
        return False
    digest = hashlib.sha256()
    with tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024) as handle:
        blob.download_to_file(handle, if_generation_match=int(blob.generation), checksum="auto")
        handle.seek(0)
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == expected_sha256


def upload_run_gcs(
    run_dir: str | Path,
    bucket_name: str,
    prefix: str,
    *,
    client: Any = None,
) -> dict[str, Any]:
    """Upload/resume a validated run, never overwrite, publish checkpoint last.

    Caller controls the bucket and model/experiment/sonnet/run prefix. Existing
    objects are accepted only if generation-pinned downloaded bytes match SHA256.
    This function neither creates buckets nor changes access controls.
    """
    from google.api_core.exceptions import PreconditionFailed
    from google.cloud import storage

    path = Path(run_dir)
    manifest = verify_run(path)
    prefix = prefix.strip("/")
    if not prefix or any(p in {".", ".."} for p in prefix.split("/")):
        raise ValueError("A nonempty, safe GCS prefix is required")
    bucket = (client or storage.Client()).bucket(bucket_name)
    files = sorted(manifest["artifacts"]) + ["manifest.json", "checkpoint.json"]
    uploaded = []
    for name in files:
        local = path / name
        digest = sha256_file(local)
        size = local.stat().st_size
        blob = bucket.blob(f"{prefix}/{name}")
        blob.metadata = {"sha256": digest, "run_id": manifest["run_id"]}
        try:
            blob.upload_from_filename(str(local), if_generation_match=0, checksum="auto")
        except PreconditionFailed:
            if not _remote_matches(blob, digest, size):
                raise IntegrityError(f"Existing GCS object differs: gs://{bucket_name}/{blob.name}") from None
        uploaded.append({"name": blob.name, "generation": str(blob.generation), "sha256": digest, "bytes": size})
    return {"bucket": bucket_name, "prefix": prefix, "run_id": manifest["run_id"], "objects": uploaded}
