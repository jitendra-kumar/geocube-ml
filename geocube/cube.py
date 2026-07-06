from dataclasses import dataclass
import xarray as xr


@dataclass
class LayerInfo:
    name: str
    dims: tuple
    dtype: str
    region: str
    grid_name: str
    resolution_degrees: str | float
    crs: str


def open_cube(cube_path: str, chunks: dict | str | None = "auto") -> xr.Dataset:
    return xr.open_zarr(cube_path, chunks=chunks)


def list_layers(cube_path: str) -> list[LayerInfo]:
    ds = open_cube(cube_path)

    layers = []
    for name, da in ds.data_vars.items():
        layers.append(
            LayerInfo(
                name=name,
                dims=tuple(da.dims),
                dtype=str(da.dtype),
                region=da.attrs.get("region", "unspecified"),
                grid_name=da.attrs.get("grid_name", "unknown"),
                resolution_degrees=da.attrs.get("resolution_degrees", "unknown"),
                crs=da.attrs.get("crs", "unknown"),
            )
        )

    return layers


def load_layers(
    cube_path: str,
    layers: list[str] | None = None,
    region: str | None = None,
    chunks: dict | str | None = "auto",
) -> xr.Dataset:
    ds = open_cube(cube_path, chunks=chunks)

    if region:
        matching = [
            name for name, da in ds.data_vars.items()
            if da.attrs.get("region") == region
        ]
        ds = ds[matching]

    if layers:
        missing = [layer for layer in layers if layer not in ds.data_vars]
        if missing:
            raise KeyError(f"Missing layers in cube: {missing}")
        ds = ds[layers]

    return ds
