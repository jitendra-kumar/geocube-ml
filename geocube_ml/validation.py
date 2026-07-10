from __future__ import annotations

import numpy as np
import zarr

from .storage import layer_group_exists, layer_group_path


def validate_layer_zarr(
    cube_path: str,
    layer_name: str,
    grid,
    missing_value: float,
) -> dict:
    if layer_group_exists(cube_path, layer_name):
        zg = zarr.open_group(str(layer_group_path(cube_path, layer_name)), mode="r")
    else:
        zg = zarr.open_group(str(cube_path), mode="r")

    arr = zg[layer_name]

    errors = []

    expected_shape = (grid.height, grid.width)

    if tuple(arr.shape) != expected_shape:
        errors.append(f"Shape mismatch: expected {expected_shape}, found {arr.shape}")

    if str(arr.dtype) != "float32":
        errors.append(f"Dtype mismatch: expected float32, found {arr.dtype}")

    # Light check: inspect chunk corners instead of scanning entire cube.
    sample_windows = [
        (slice(0, min(32, grid.height)), slice(0, min(32, grid.width))),
        (slice(max(0, grid.height - 32), grid.height), slice(max(0, grid.width - 32), grid.width)),
    ]

    nan_found = False
    for ys, xs in sample_windows:
        block = arr[ys, xs]
        if np.isnan(block).any():
            nan_found = True
            break

    if nan_found:
        errors.append("NaN values found in sampled blocks; expected valid values or missing_value flag.")

    return {
        "layer_name": layer_name,
        "expected_shape": expected_shape,
        "actual_shape": tuple(arr.shape),
        "dtype": str(arr.dtype),
        "missing_value": missing_value,
        "errors": errors,
        "ok": len(errors) == 0,
    }
