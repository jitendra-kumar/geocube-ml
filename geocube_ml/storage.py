from __future__ import annotations

from pathlib import Path
import shutil


LAYER_GROUP_ROOT = "layers"


def validate_layer_name(layer_name: str) -> str:
    if not layer_name:
        raise ValueError("Layer name cannot be empty.")

    path = Path(layer_name)
    if path.name != layer_name or layer_name in {".", ".."}:
        raise ValueError(
            "Layer names must be relative names without path separators."
        )

    return layer_name


def layer_group_name(layer_name: str) -> str:
    layer_name = validate_layer_name(layer_name)
    return f"{LAYER_GROUP_ROOT}/{layer_name}"


def layer_group_root(cube_path: str | Path) -> Path:
    return Path(cube_path) / LAYER_GROUP_ROOT


def layer_group_path(cube_path: str | Path, layer_name: str) -> Path:
    layer_name = validate_layer_name(layer_name)
    return layer_group_root(cube_path) / layer_name


def list_layer_group_names(cube_path: str | Path) -> list[str]:
    root = layer_group_root(cube_path)
    if not root.exists():
        return []

    return sorted(path.name for path in root.iterdir() if path.is_dir())


def layer_group_exists(cube_path: str | Path, layer_name: str) -> bool:
    return layer_group_path(cube_path, layer_name).exists()


def remove_layer_group(cube_path: str | Path, layer_name: str) -> None:
    path = layer_group_path(cube_path, layer_name)
    if path.exists():
        shutil.rmtree(path)
