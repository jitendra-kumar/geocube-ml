import geopandas as gpd
import xarray as xr


def extract_points(
    cube_path: str,
    points: gpd.GeoDataFrame,
    layers: list[str] | None = None,
    method: str = "nearest",
):
    ds = xr.open_zarr(cube_path, chunks={})

    if layers:
        ds = ds[layers]

    pts = points.to_crs("EPSG:4326")
    xs = xr.DataArray(pts.geometry.x.values, dims="sample")
    ys = xr.DataArray(pts.geometry.y.values, dims="sample")

    sampled = ds.sel(x=xs, y=ys, method=method).to_dataframe().reset_index()
    sampled = sampled.drop(columns=["x", "y"], errors="ignore")

    return pts.reset_index(drop=True).join(sampled)
