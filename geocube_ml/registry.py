from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import copy
import json

from .provenance import file_sha256


REGISTRY_FILENAME = ".geocube_ml_layer_registry.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def registry_path(cube_path: str | Path) -> Path:
    return Path(cube_path) / REGISTRY_FILENAME


@dataclass
class LayerBuildSpec:
    source_path: str
    source_variable: str | None
    layer_name: str
    cube_name: str
    grid_name: str
    region: str
    crs: str
    resolution_degrees: float
    extent: list[float]
    resampling: str
    source_nodata: float | None
    missing_value: float
    geocube_ml_version: str = "0.1.0"

    def normalized(self) -> dict:
        d = asdict(self)
        d["source_path"] = str(Path(self.source_path).resolve())
        return d

    def checksum_payload(self) -> dict:
        d = self.normalized()
        d["source_sha256"] = file_sha256(self.source_path)
        return d


def load_registry(cube_path: str | Path) -> dict:
    path = registry_path(cube_path)

    if not path.exists():
        return {
            "software": "geocube-ml",
            "registry_version": "0.1.0",
            "created_at_utc": now_utc(),
            "last_updated_utc": now_utc(),
            "layers": {},
        }

    return json.loads(path.read_text())


def save_registry(cube_path: str | Path, registry: dict) -> None:
    path = registry_path(cube_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    registry["last_updated_utc"] = now_utc()
    path.write_text(json.dumps(registry, indent=2))


def latest_layer_record(cube_path: str | Path, layer_name: str) -> dict | None:
    registry = load_registry(cube_path)
    layer = registry.get("layers", {}).get(layer_name)

    if not layer or layer.get("deleted"):
        return None

    history = layer.get("history", [])
    if not history:
        return None

    return history[-1]


def determine_ingest_action(
    cube_path: str | Path,
    spec: LayerBuildSpec,
    update_mode: str = "checksum",
) -> dict:
    """
    update_mode:
      - missing: only ingest if layer does not exist
      - skip: skip if layer exists, regardless of changes
      - checksum: ingest if checksum or processing spec changed
      - overwrite: always ingest
    """
    if update_mode not in {"missing", "skip", "checksum", "overwrite"}:
        raise ValueError(
            "update_mode must be one of: missing, skip, checksum, overwrite"
        )

    current_payload = spec.checksum_payload()
    previous = latest_layer_record(cube_path, spec.layer_name)

    if previous is None:
        return {
            "action": "ingest",
            "reason": "new_layer",
            "current_payload": current_payload,
            "previous_payload": None,
        }

    if update_mode in {"missing", "skip"}:
        return {
            "action": "skip",
            "reason": f"update_mode_{update_mode}",
            "current_payload": current_payload,
            "previous_payload": previous.get("build_spec"),
        }

    if update_mode == "overwrite":
        return {
            "action": "ingest",
            "reason": "update_mode_overwrite",
            "current_payload": current_payload,
            "previous_payload": previous.get("build_spec"),
        }

    previous_payload = previous.get("build_spec", {})

    changed_keys = sorted(
        key
        for key, value in current_payload.items()
        if previous_payload.get(key) != value
    )

    if changed_keys:
        return {
            "action": "ingest",
            "reason": "changed",
            "changed_keys": changed_keys,
            "current_payload": current_payload,
            "previous_payload": previous_payload,
        }

    return {
        "action": "skip",
        "reason": "unchanged",
        "changed_keys": [],
        "current_payload": current_payload,
        "previous_payload": previous_payload,
    }


def append_layer_registry_record(
    cube_path: str | Path,
    layer_name: str,
    build_spec_payload: dict,
    provenance: dict,
    statistics: dict | None = None,
    validation: dict | None = None,
) -> dict:
    registry = load_registry(cube_path)
    layers = registry.setdefault("layers", {})

    layer_entry = layers.setdefault(
        layer_name,
        {
            "layer_name": layer_name,
            "current_version": 0,
            "history": [],
        },
    )

    version = int(layer_entry.get("current_version", 0)) + 1

    record = {
        "version": version,
        "ingested_at_utc": now_utc(),
        "build_spec": build_spec_payload,
        "provenance": provenance,
        "statistics": statistics or {},
        "validation": validation or {},
    }

    layer_entry["current_version"] = version
    layer_entry["deleted"] = False
    layer_entry["history"].append(record)

    save_registry(cube_path, registry)
    return record


def mark_layer_deleted(cube_path: str | Path, layer_name: str) -> dict:
    registry = load_registry(cube_path)
    layers = registry.setdefault("layers", {})

    layer_entry = layers.setdefault(
        layer_name,
        {
            "layer_name": layer_name,
            "current_version": 0,
            "history": [],
        },
    )

    version = int(layer_entry.get("current_version", 0)) + 1
    record = {
        "version": version,
        "action": "deleted",
        "deleted_at_utc": now_utc(),
    }

    layer_entry["current_version"] = version
    layer_entry["deleted"] = True
    layer_entry["deleted_at_utc"] = record["deleted_at_utc"]
    layer_entry["history"].append(record)

    save_registry(cube_path, registry)
    return record


def rename_layer_registry(
    cube_path: str | Path,
    old_name: str,
    new_name: str,
) -> dict:
    registry = load_registry(cube_path)
    layers = registry.setdefault("layers", {})

    existing_new = layers.get(new_name)
    if existing_new and not existing_new.get("deleted"):
        raise ValueError(f"Layer already exists in registry: {new_name}")

    old_entry = layers.get(old_name)
    if not old_entry:
        save_registry(cube_path, registry)
        return {}

    new_entry = copy.deepcopy(old_entry)
    old_version = int(old_entry.get("current_version", 0)) + 1
    renamed_at = now_utc()

    old_event = {
        "version": old_version,
        "action": "renamed_to",
        "new_layer_name": new_name,
        "renamed_at_utc": renamed_at,
    }
    old_entry["current_version"] = old_version
    old_entry["deleted"] = True
    old_entry["renamed_to"] = new_name
    old_entry["history"].append(old_event)

    new_version = int(new_entry.get("current_version", 0)) + 1
    new_event = {
        "version": new_version,
        "action": "renamed_from",
        "old_layer_name": old_name,
        "renamed_at_utc": renamed_at,
    }
    new_entry["layer_name"] = new_name
    new_entry["current_version"] = new_version
    new_entry["deleted"] = False
    new_entry.pop("renamed_to", None)
    new_entry["history"].append(new_event)
    layers[new_name] = new_entry

    save_registry(cube_path, registry)
    return new_event
