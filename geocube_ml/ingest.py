from pathlib import Path
import json
import numpy as np
import xarray as xr
import dask.array as dsa
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
import zarr
import math
from tqdm import tqdm
from .manifest import update_manifest
from .validation import validate_layer_zarr
from .provenance import build_provenance
from .catalog import upsert_stac_item
from .storage import (
    layer_group_name,
    layer_group_path,
    remove_layer_group,
)
from .registry import (
    LayerBuildSpec,
    append_layer_registry_record,
    determine_ingest_action,
)

RESAMPLING = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "average": Resampling.average,
    "mode": Resampling.mode,
}

class RunningStats:
    def __init__(self, missing_value: float):
        self.missing_value = missing_value
        self.valid_count = 0
        self.missing_count = 0
        self.total_count = 0
        self.min = None
        self.max = None
        self.sum = 0.0
        self.sum_sq = 0.0

    def update(self, block):
        mask = block != self.missing_value
        valid = block[mask]

        self.total_count += block.size
        self.valid_count += valid.size
        self.missing_count += block.size - valid.size

        if valid.size == 0:
            return

        bmin = float(valid.min())
        bmax = float(valid.max())

        self.min = bmin if self.min is None else min(self.min, bmin)
        self.max = bmax if self.max is None else max(self.max, bmax)

        self.sum += float(valid.sum(dtype="float64"))
        self.sum_sq += float((valid.astype("float64") ** 2).sum())

    def finalize(self) -> dict:
        if self.valid_count == 0:
            return {
                "valid_count": 0,
                "missing_count": self.missing_count,
                "total_count": self.total_count,
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
            }

        mean = self.sum / self.valid_count
        variance = max((self.sum_sq / self.valid_count) - mean**2, 0.0)

        return {
            "valid_count": int(self.valid_count),
            "missing_count": int(self.missing_count),
            "total_count": int(self.total_count),
            "min": self.min,
            "max": self.max,
            "mean": mean,
            "std": variance**0.5,
        }

def rasterio_source_path(source_path: str, variable: str | None = None) -> str:
    """
    Return a GDAL/rasterio-readable source path.

    GeoTIFF:
        file.tif

    NetCDF:
        NETCDF:"file.nc":variable
    """
    lower = source_path.lower()

    if lower.endswith((".tif", ".tiff")):
        return source_path

    if lower.endswith((".nc", ".nc4", ".netcdf")):
        if variable is None:
            raise ValueError("NetCDF ingest requires variable=...")
        return f'NETCDF:"{source_path}":{variable}'

    return source_path


def initialize_layer_zarr(
    cube_path: str,
    layer_name: str,
    grid,
    attrs: dict,
    missing_value: float = -9999.0,
    overwrite: bool = True,
):
    """
    Create xarray/Zarr metadata and an empty Dask-backed variable.

    This does not materialize the full array in memory.
    """
    cube_path = Path(cube_path)
    cube_path.mkdir(parents=True, exist_ok=True)
    group_path = layer_group_path(cube_path, layer_name)

    if group_path.exists():
        if not overwrite:
            raise ValueError(f"Layer already exists: {layer_name}")
        remove_layer_group(cube_path, layer_name)

    y = grid.y_coords()
    x = grid.x_coords()

    data = dsa.full(
        shape=(grid.height, grid.width),
        fill_value=missing_value,
        dtype="float32",
        chunks=grid.chunks,
    )

    da = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={"y": y, "x": x},
        name=layer_name,
        attrs=attrs,
    )

    ds = da.to_dataset()
    ds.attrs.update(
        {
            "grid_name": grid.name,
            "crs": grid.crs,
            "resolution_degrees": grid.resolution,
            "extent": [grid.xmin, grid.ymin, grid.xmax, grid.ymax],
        }
    )

    encoding = {
        layer_name: {
            "chunks": grid.chunks,
            "dtype": "float32",
            "_FillValue": missing_value,
        }
    }

    # Writes metadata and creates the variable lazily/chunked.
    delayed = ds.to_zarr(
        cube_path,
        group=layer_group_name(layer_name),
        mode="a",
        encoding=encoding,
        compute=False,
    )

    # Actually initialize the array with missing values chunkwise.
    # This is still Dask-chunked, not one giant allocation.
    delayed.compute()


def ingest_layer(
    source_path: str,
    cube_path: str,
    cube_name: str,
    grid,
    layer_name: str,
    description: str | None = None,
    variable: str | None = None,
    region: str | None = None,
    resampling: str = "bilinear",
    nodata: float | None = None,
    missing_value: float = -9999.0,
    overwrite: bool = True,
    stac_dir: str | None = None,
    update_mode: str = "checksum",
    dry_run: bool = False,
):
    """
    Memory-efficient ingest.

    Reprojects source data into the target cube grid one output block at a time.
    Peak memory is roughly one target chunk plus GDAL's internal source window.
    """
    region = region or grid.name
    src_path = rasterio_source_path(source_path, variable=variable)

    with rasterio.open(src_path) as src:
        source_nodata = nodata if nodata is not None else src.nodata

        build_spec = LayerBuildSpec(
            source_path=source_path,
            source_variable=variable,
            layer_name=layer_name,
            cube_name=cube_name,
            grid_name=grid.name,
            region=region,
            crs=grid.crs,
            resolution_degrees=grid.resolution,
            extent=[grid.xmin, grid.ymin, grid.xmax, grid.ymax],
            resampling=resampling,
            source_nodata=source_nodata,
            missing_value=missing_value,
        )
        
        plan = determine_ingest_action(
            cube_path=cube_path,
            spec=build_spec,
            update_mode=update_mode,
        )
        
        if plan["action"] == "skip":
            return {
                "layer": layer_name,
                "status": "skipped",
                "reason": plan["reason"],
                "changed_keys": plan.get("changed_keys", []),
            }
        
        if dry_run:
            return {
                "layer": layer_name,
                "status": "would_ingest",
                "reason": plan["reason"],
                "changed_keys": plan.get("changed_keys", []),
            }

        provenance = build_provenance(
            source_path=source_path,
            layer_name=layer_name,
            cube_name=cube_name,
            grid=grid,
            region=region,
            resampling=resampling,
            source_variable=variable,
            description=description,
            source_nodata=source_nodata,
            missing_value=missing_value,
        )

        attrs = {
            "source_path": str(source_path),
            "description": description,
            "region": region,
            "cube_name": cube_name,
            "grid_name": grid.name,
            "resolution_degrees": grid.resolution,
            "crs": grid.crs,
            "missing_value": missing_value,
            "extent": [grid.xmin, grid.ymin, grid.xmax, grid.ymax],
            "resampling": resampling,
            "provenance": provenance.to_json(),
        }

        initialize_layer_zarr(
            cube_path=cube_path,
            layer_name=layer_name,
            grid=grid,
            attrs=attrs,
            missing_value=missing_value,
            overwrite=overwrite,
        )

        zg = zarr.open_group(str(layer_group_path(cube_path, layer_name)), mode="a")
        zarr_arr = zg[layer_name]

        vrt_kwargs = {
            "crs": grid.crs,
            "transform": grid.transform,
            "width": grid.width,
            "height": grid.height,
            "nodata": missing_value,
            "resampling": RESAMPLING[resampling],
        }

        if nodata is not None:
            vrt_kwargs["src_nodata"] = nodata

        stats = RunningStats(missing_value)

        nrows = math.ceil(grid.height / grid.chunks[0])
        ncols = math.ceil(grid.width / grid.chunks[1])
        total_windows = nrows * ncols
        
        with WarpedVRT(src, **vrt_kwargs) as vrt:
            for window in tqdm(
                grid.iter_windows(),
                total=total_windows,
                desc=f"Ingesting {layer_name}",
                unit="block",
            ):
                row0 = int(window.row_off)
                row1 = row0 + int(window.height)
                col0 = int(window.col_off)
                col1 = col0 + int(window.width)
        
                block = vrt.read(
                    1,
                    window=window,
                    out_shape=(int(window.height), int(window.width)),
                    masked=True,
                )
        
                block = np.asarray(block.filled(missing_value), dtype="float32")
                block[~np.isfinite(block)] = missing_value
        
                stats.update(block)
        
                zarr_arr[row0:row1, col0:col1] = block

    layer_stats = stats.finalize()

    # Reopen variable attrs through xarray and attach stats.
    ds = xr.open_zarr(
        cube_path,
        group=layer_group_name(layer_name),
        chunks={},
    )
    attrs = dict(ds[layer_name].attrs)
    attrs["statistics"] = json.dumps(layer_stats)
    
    zg = zarr.open_group(str(layer_group_path(cube_path, layer_name)), mode="a")
    zg[layer_name].attrs.update(attrs)
    
    validation = validate_layer_zarr(
        cube_path=cube_path,
        layer_name=layer_name,
        grid=grid,
        missing_value=missing_value,
    )
    
    if not validation["ok"]:
        raise RuntimeError(f"Layer validation failed: {validation}")
    
    update_manifest(
        cube_path=cube_path,
        layer_name=layer_name,
        layer_attrs=attrs,
        stats=layer_stats,
    )

    provenance_dict = provenance.to_dict()
    provenance_dict["statistics"] = layer_stats
    provenance_dict["validation"] = validation

    if stac_dir:
        upsert_stac_item(
            stac_dir=stac_dir,
            cube_path=cube_path,
            cube_name=cube_name,
            layer_name=layer_name,
            grid=grid,
            source_path=source_path,
            region=region,
            description=description,
            provenance=provenance_dict,
            zarr_group=layer_group_name(layer_name),
        )

    append_layer_registry_record(
        cube_path=cube_path,
        layer_name=layer_name,
        build_spec_payload=plan["current_payload"],
        provenance=provenance_dict,
        statistics=layer_stats,
        validation=validation,
    )
    
    return {
        "layer": layer_name,
        "status": "ingested",
        "reason": plan["reason"],
        "changed_keys": plan.get("changed_keys", []),
        "statistics": layer_stats,
        "validation": validation,
    }
