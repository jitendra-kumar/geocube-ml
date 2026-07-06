from pathlib import Path
import json
import typer
import geopandas as gpd

from .grid import CubeGrid
from .ingest import ingest_layer
from .extract import extract_points
from .cube import list_layers, load_layers

app = typer.Typer(help="Build and query analysis-ready ancillary Zarr cubes.")


def load_grid(grid_path: str) -> CubeGrid:
    with open(grid_path, "r") as f:
        cfg = json.load(f)

    return CubeGrid(**cfg)


@app.command()
def ingest(
    source: str = typer.Argument(..., help="Input GeoTIFF or NetCDF path."),
    cube: str = typer.Option(..., help="Output Zarr cube path."),
    layer: str = typer.Option(..., help="Layer name to write into cube."),
    grid: str = typer.Option(..., help="Grid JSON config path."),
    variable: str | None = typer.Option(None, help="NetCDF variable name."),
    region: str | None = typer.Option(None, help="Region label, e.g. global, CONUS, Amazon."),
    resampling: str = typer.Option("bilinear", help="nearest, bilinear, cubic, average, mode."),
    nodata: float | None = typer.Option(None, help="Optional nodata value."),
    stac_dir: str | None = typer.Option(None, help="Optional STAC catalog directory."),
    overwrite: bool = typer.Option(True, help="Overwrite existing layer."),
):
    grid_obj = load_grid(grid)

    ingest_layer(
        source_path=source,
        cube_path=cube,
        grid=grid_obj,
        layer_name=layer,
        variable=variable,
        region=region,
        resampling=resampling,
        nodata=nodata,
        overwrite=overwrite,
        stac_dir=stac_dir,
    )

    typer.echo(f"Ingested {layer} into {cube}")


@app.command()
def extract(
    cube: str = typer.Argument(..., help="Input Zarr cube path."),
    points: str = typer.Argument(..., help="Point file readable by GeoPandas."),
    out: str = typer.Option(..., help="Output table path: .parquet, .csv, or .gpkg."),
    layers: list[str] | None = typer.Option(None, help="Layer names to extract."),
    method: str = typer.Option("nearest", help="xarray selection method."),
):
    gdf = gpd.read_file(points)

    table = extract_points(
        cube_path=cube,
        points=gdf,
        layers=layers,
        method=method,
    )

    suffix = Path(out).suffix.lower()

    if suffix == ".parquet":
        table.to_parquet(out)
    elif suffix == ".csv":
        table.drop(columns="geometry", errors="ignore").to_csv(out, index=False)
    elif suffix in [".gpkg", ".geojson", ".shp"]:
        table.to_file(out)
    else:
        raise typer.BadParameter("Output must be .parquet, .csv, .gpkg, .geojson, or .shp")

    typer.echo(f"Wrote extracted table to {out}")


@app.command()
def describe(
    cube: str = typer.Argument(..., help="Input Zarr cube path."),
):
    import xarray as xr

    ds = xr.open_zarr(cube, chunks={})

    typer.echo(f"Cube: {cube}")
    typer.echo(f"Dimensions: {dict(ds.sizes)}")
    typer.echo("Layers:")

    for name in ds.data_vars:
        da = ds[name]
        region = da.attrs.get("region", "unspecified")
        resolution = da.attrs.get("resolution_degrees", "unknown")
        typer.echo(f"  - {name} | region={region} | resolution={resolution}")


@app.command()
def init_grid(
    out: str = typer.Argument(..., help="Output grid JSON path."),
    name: str = typer.Option(..., help="Grid name."),
    resolution: float = typer.Option(..., help="Grid resolution in degrees."),
    xmin: float = typer.Option(-180.0),
    ymin: float = typer.Option(-90.0),
    xmax: float = typer.Option(180.0),
    ymax: float = typer.Option(90.0),
    crs: str = typer.Option("EPSG:4326"),
):
    cfg = {
        "name": name,
        "resolution": resolution,
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
        "crs": crs,
    }

    Path(out).parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        json.dump(cfg, f, indent=2)

    typer.echo(f"Wrote grid config to {out}")


@app.command("layers")
def layers_cmd(
    cube: str = typer.Argument(..., help="Input Zarr cube path."),
    region: str | None = typer.Option(None, help="Optional region filter."),
):
    infos = list_layers(cube)

    if region:
        infos = [info for info in infos if info.region == region]

    for info in infos:
        typer.echo(
            f"{info.name} | region={info.region} | "
            f"grid={info.grid_name} | res={info.resolution_degrees} | "
            f"crs={info.crs} | dtype={info.dtype}"
        )


@app.command("subset")
def subset_cmd(
    cube: str = typer.Argument(..., help="Input Zarr cube path."),
    out: str = typer.Argument(..., help="Output Zarr path."),
    layers: list[str] = typer.Option(..., help="Layer to include. Repeatable."),
    region: str | None = typer.Option(None, help="Optional region filter."),
):
    ds = load_layers(cube, layers=layers, region=region)
    ds.to_zarr(out, mode="w")
    typer.echo(f"Wrote subset cube to {out}")

if __name__ == "__main__":
    app()
