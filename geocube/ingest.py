import numpy as np
from pathlib import Path
import xarray as xr
import rioxarray  # noqa
from rasterio.enums import Resampling
from .grid import CubeGrid
from .catalog import upsert_stac_item


RESAMPLING = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "average": Resampling.average,
    "mode": Resampling.mode,
}


def _standardize_missing(da: xr.DataArray, missing_value: float) -> xr.DataArray:
    da = da.where(np.isfinite(da), missing_value)
    da = da.fillna(missing_value)
    da = da.astype("float32")
    da = da.rio.write_nodata(missing_value)
    return da


def align_to_grid(
    da: xr.DataArray,
    grid: CubeGrid,
    resampling: str = "bilinear",
    missing_value: float = -9999.0,
) -> xr.DataArray:
    """
    Reproject/resample source data to the exact target grid.

    Guarantees:
    - CRS matches grid.crs
    - x/y coordinates match grid.template()
    - shape matches target region
    - no data outside the region is retained
    - gaps inside the region are filled with missing_value
    """
    template = grid.template()

    aligned = da.rio.reproject_match(
        template,
        resampling=RESAMPLING[resampling],
        nodata=missing_value,
    )

    aligned = aligned.reindex_like(template)
    aligned = _standardize_missing(aligned, missing_value)

    return aligned

def open_layer(path: str, variable: str | None = None) -> xr.DataArray:
    path = str(path)

    if path.lower().endswith((".tif", ".tiff")):
        da = rioxarray.open_rasterio(path, masked=True, chunks=True)
        if "band" in da.dims and da.sizes["band"] == 1:
            da = da.squeeze("band", drop=True)
        return da

    ds = xr.open_dataset(path, chunks={})
    if variable is None:
        data_vars = list(ds.data_vars)
        if len(data_vars) != 1:
            raise ValueError(f"NetCDF has multiple variables: {data_vars}. Pass variable=...")
        variable = data_vars[0]

    da = ds[variable]

    if not da.rio.crs:
        crs = ds.attrs.get("crs") or ds.attrs.get("spatial_ref")
        if not crs:
            raise ValueError("NetCDF variable has no CRS. Assign CRS before ingest.")
        da = da.rio.write_crs(crs)

    da.rio.set_spatial_dims(x_dim="lon" if "lon" in da.dims else "x",
                            y_dim="lat" if "lat" in da.dims else "y",
                            inplace=True)
    return da

def ingest_layer(
    source_path: str,
    cube_path: str,
    grid: CubeGrid,
    layer_name: str,
    variable: str | None = None,
    region: str | None = None,
    resampling: str = "bilinear",
    nodata: float | None = None,
    missing_value: float = -9999.0,
    overwrite: bool = True,
    stac_dir: str | None = None,
):
    da = open_layer(source_path, variable=variable)

    if nodata is not None:
        da = da.rio.write_nodata(nodata)

    da = align_to_grid(
        da,
        grid=grid,
        resampling=resampling,
        missing_value=missing_value,
    )

    da.name = layer_name
    da.attrs.update(
        {
            "source_path": str(source_path),
            "region": region or grid.name,
            "grid_name": grid.name,
            "resolution_degrees": grid.resolution,
            "crs": grid.crs,
            "missing_value": missing_value,
            "extent": [grid.xmin, grid.ymin, grid.xmax, grid.ymax],
        }
    )

    ds = da.to_dataset()

    mode = "a" if Path(cube_path).exists() else "w"

    if not overwrite and mode == "a":
        existing = xr.open_zarr(cube_path)
        if layer_name in existing:
            raise ValueError(f"{layer_name} already exists in {cube_path}")

    encoding = {
        layer_name: {
            "chunks": (min(1024, da.sizes["y"]), min(1024, da.sizes["x"])),
            "dtype": "float32",
            "_FillValue": missing_value,
        }
    }

    ds.to_zarr(cube_path, mode=mode, encoding=encoding)

    if stac_dir:
        upsert_stac_item(
            stac_dir=stac_dir,
            cube_path=cube_path,
            layer_name=layer_name,
            grid=grid,
            source_path=source_path,
            region=region or grid.name,
        )

    return da


