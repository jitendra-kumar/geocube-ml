from dataclasses import dataclass
import json
import xarray as xr
import zarr

from .storage import (
    layer_group_exists,
    layer_group_name,
    list_layer_group_names,
    remove_layer_group,
    validate_layer_name,
)


@dataclass
class LayerInfo:
    name: str
    description: str | None
    dims: tuple
    dtype: str
    region: str
    cube_name: str
    grid_name: str
    resolution_degrees: str | float
    crs: str
    provenance: dict | None


def _open_root_cube(
    cube_path: str,
    chunks: dict | str | None = "auto",
) -> xr.Dataset | None:
    try:
        return xr.open_zarr(cube_path, chunks=chunks)
    except Exception:
        return None


def _single_data_var(ds: xr.Dataset, layer_name: str) -> str:
    if layer_name in ds.data_vars:
        return layer_name

    data_vars = list(ds.data_vars)
    if len(data_vars) == 1:
        return data_vars[0]

    raise KeyError(f"Layer not found: {layer_name}")


def _open_layer_dataset(
    cube_path: str,
    layer_name: str,
    chunks: dict | str | None = "auto",
) -> xr.Dataset:
    validate_layer_name(layer_name)

    if layer_group_exists(cube_path, layer_name):
        ds = xr.open_zarr(
            cube_path,
            group=layer_group_name(layer_name),
            chunks=chunks,
        )
        var_name = _single_data_var(ds, layer_name)
        if var_name != layer_name:
            ds = ds.rename({var_name: layer_name})
        return ds[[layer_name]]

    root_ds = _open_root_cube(cube_path, chunks=chunks)
    if root_ds is not None and layer_name in root_ds.data_vars:
        return root_ds[[layer_name]]

    raise KeyError(f"Layer not found: {layer_name}")


def open_cube(cube_path: str, chunks: dict | str | None = "auto") -> xr.Dataset:
    return load_layers(cube_path, chunks=chunks)


def get_layer_provenance(cube_path: str, layer_name: str) -> dict:
    ds = _open_layer_dataset(cube_path, layer_name)
    raw = ds[layer_name].attrs.get("provenance")
    if not raw:
        return {}

    return json.loads(raw)


def list_layers(cube_path: str) -> list[LayerInfo]:
    layers = []

    seen = set()
    for name in list_layer_group_names(cube_path):
        ds = _open_layer_dataset(cube_path, name)
        da = ds[name]
        raw_prov = da.attrs.get("provenance")
        provenance = json.loads(raw_prov) if raw_prov else None

        layers.append(
            LayerInfo(
                name=name,
                description=da.attrs.get("description"),
                dims=tuple(da.dims),
                dtype=str(da.dtype),
                region=da.attrs.get("region", "unspecified"),
                cube_name=da.attrs.get("cube_name", "unknown"),
                grid_name=da.attrs.get("grid_name", "unknown"),
                resolution_degrees=da.attrs.get("resolution_degrees", "unknown"),
                crs=da.attrs.get("crs", "unknown"),
                provenance=provenance,
            )
        )
        seen.add(name)

    if seen:
        return layers

    root_ds = _open_root_cube(cube_path)
    if root_ds is None:
        return layers

    for name, da in root_ds.data_vars.items():
        if name in seen:
            continue

        raw_prov = da.attrs.get("provenance")
        provenance = json.loads(raw_prov) if raw_prov else None

        layers.append(
            LayerInfo(
                name=name,
                description=da.attrs.get("description"),
                dims=tuple(da.dims),
                dtype=str(da.dtype),
                region=da.attrs.get("region", "unspecified"),
                cube_name=da.attrs.get("cube_name", "unknown"),
                grid_name=da.attrs.get("grid_name", "unknown"),
                resolution_degrees=da.attrs.get("resolution_degrees", "unknown"),
                crs=da.attrs.get("crs", "unknown"),
                provenance=provenance,
            )
        )

    return layers


def load_layers(
    cube_path: str,
    layers: list[str] | None = None,
    region: str | None = None,
    chunks: dict | str | None = "auto",
) -> xr.Dataset:
    available = [layer.name for layer in list_layers(cube_path)]
    selected = layers or available

    missing = [layer for layer in selected if layer not in available]
    if missing:
        raise KeyError(f"Missing layers in cube: {missing}")

    datasets = []
    for layer in selected:
        ds = _open_layer_dataset(cube_path, layer, chunks=chunks)
        da = ds[layer]

        if region and da.attrs.get("region") != region:
            continue

        datasets.append(ds)

    if not datasets:
        return xr.Dataset()

    return xr.merge(datasets)


def delete_layer_data(cube_path: str, layer_name: str) -> None:
    validate_layer_name(layer_name)

    if layer_group_exists(cube_path, layer_name):
        remove_layer_group(cube_path, layer_name)
        return

    try:
        zg = zarr.open_group(str(cube_path), mode="r+")
        if layer_name in zg:
            del zg[layer_name]
            return
    except Exception:
        pass

    raise KeyError(f"Layer not found: {layer_name}")


def rename_layer_data(cube_path: str, old_name: str, new_name: str) -> None:
    validate_layer_name(old_name)
    validate_layer_name(new_name)

    if old_name == new_name:
        raise ValueError("New layer name must differ from old layer name.")

    available = {layer.name for layer in list_layers(cube_path)}
    if new_name in available:
        raise ValueError(f"Layer already exists: {new_name}")
    if old_name not in available:
        raise KeyError(f"Layer not found: {old_name}")

    ds = _open_layer_dataset(cube_path, old_name)
    ds = ds.rename({old_name: new_name})

    attrs = dict(ds[new_name].attrs)
    raw_prov = attrs.get("provenance")
    if raw_prov:
        provenance = json.loads(raw_prov)
        provenance["layer_name"] = new_name
        attrs["provenance"] = json.dumps(provenance, indent=2)
    ds[new_name].attrs.update(attrs)

    ds.to_zarr(
        cube_path,
        group=layer_group_name(new_name),
        mode="a",
    )

    delete_layer_data(cube_path, old_name)
