"""
Tracks which files have already been ingested, keyed by content hash
(not just filename) so:
  - re-uploading the exact same file is skipped instantly
  - a renamed-but-identical file is also skipped
  - a file with the same name but different/edited content IS reprocessed
"""
import hashlib
import json
import os
from datetime import datetime

REGISTRY_PATH = "data/processed_files.json"


def file_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()


def load_registry(path: str = REGISTRY_PATH) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_registry(registry: dict, path: str = REGISTRY_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)


def is_processed(file_hash_value: str, registry: dict) -> bool:
    return any(entry["hash"] == file_hash_value for entry in registry.values())


def register_file(filename: str, file_hash_value: str, chunk_count: int, registry: dict) -> dict:
    registry[filename] = {
        "hash": file_hash_value,
        "chunk_count": chunk_count,
        "processed_at": datetime.now().isoformat(timespec="seconds"),
    }
    return registry


def remove_file(filename: str, registry: dict) -> dict:
    registry.pop(filename, None)
    return registry