import json
import typer

from .grid import CubeGrid
from .collection import CubeCollection
from .cube import get_layer_provenance
from .manifest import load_manifest, validate_cube_manifest
from .registry import load_registry

app = typer.Typer(help="Build and query analysis-ready ancillary Zarr cubes.")


@app.command("collection-init")
def collection_init(
    root: str = typer.Argument(..., help="Collection root directory."),
):
    collection = CubeCollection(root)
    collection.save()
    typer.echo(f"Initialized GeoCube collection at {root}")


@app.command("collection-add-cube")
def collection_add_cube(
    root: str = typer.Argument(...),
    name: str = typer.Option(...),
    region: str = typer.Option(...),
    resolution_label: str = typer.Option(...),
    resolution: float = typer.Option(...),
    xmin: float = typer.Option(...),
    ymin: float = typer.Option(...),
    xmax: float = typer.Option(...),
    ymax: float = typer.Option(...),
    chunks_y: int = typer.Option(1024),
    chunks_x: int = typer.Option(1024),
    crs: str = typer.Option("EPSG:4326"),
    description: str | None = typer.Option(None),
):
    collection = CubeCollection(root)

    grid = CubeGrid(
        name=name,
        resolution=resolution,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        crs=crs,
        chunks=(chunks_y, chunks_x),
    )

    collection.add_cube(
        name=name,
        grid=grid,
        region=region,
        resolution_label=resolution_label,
        description=description,
    )

    typer.echo(f"Added cube {name} to collection {root}")


@app.command("collection-ingest")
def collection_ingest(
    root: str = typer.Argument(...),
    cube_name: str = typer.Option(...),
    source: str = typer.Argument(...),
    layer: str = typer.Option(...),
    variable: str | None = typer.Option(None),
    resampling: str = typer.Option("bilinear"),
    nodata: float | None = typer.Option(None),
    missing_value: float = typer.Option(-9999.0),
    overwrite: bool = typer.Option(True),
    update_mode: str = typer.Option( "checksum", help="missing, skip, checksum, or overwrite.",),
    dry_run: bool = typer.Option(False),
):
    collection = CubeCollection(root)

    result = collection.ingest(
        cube_name=cube_name,
        source_path=source,
        layer_name=layer,
        variable=variable,
        resampling=resampling,
        nodata=nodata,
        missing_value=missing_value,
        overwrite=overwrite,
        update_mode=update_mode,
        dry_run=dry_run,
    )
    
    typer.echo(json.dumps(result, indent=2))

    typer.echo(f"Ingested {layer} into cube {cube_name}")

@app.command("collection-ingest-dir")
def collection_ingest_dir(
    root: str = typer.Argument(...),
    source_dir: str = typer.Argument(...),
    cube_name: str = typer.Option(...),
    pattern: str = typer.Option("*"),
    variable: str | None = typer.Option(None),
    resampling: str = typer.Option("bilinear"),
    nodata: float | None = typer.Option(None),
    missing_value: float = typer.Option(-9999.0),
    overwrite: bool = typer.Option(True),
    continue_on_error: bool = typer.Option(True),
    update_mode: str = typer.Option( "checksum", help="missing, skip, checksum, or overwrite.",),
    dry_run: bool = typer.Option(False),
):
    collection = CubeCollection(root)

    results = collection.ingest_dir(
        cube_name=cube_name,
        source_dir=source_dir,
        pattern=pattern,
        variable=variable,
        resampling=resampling,
        nodata=nodata,
        missing_value=missing_value,
        overwrite=overwrite,
        continue_on_error=continue_on_error,
        update_mode=update_mode,
        dry_run=dry_run,
    )

    completed = sum(
        1
        for r in results
        if r["status"] in {"ingested", "skipped", "would_ingest"}
    )
    failed = sum(1 for r in results if r["status"] == "failed")

    typer.echo(f"Batch ingest complete: {completed} completed, {failed} failed")

    for result in results:
        if result["status"] == "failed":
            typer.echo(f"FAILED: {result['source']} :: {result['error']}")


@app.command("collection-update-layer")
def collection_update_layer(
    root: str = typer.Argument(...),
    source: str = typer.Argument(...),
    cube_name: str = typer.Option(...),
    layer: str = typer.Option(...),
    variable: str | None = typer.Option(None),
    resampling: str = typer.Option("bilinear"),
    nodata: float | None = typer.Option(None),
    missing_value: float = typer.Option(-9999.0),
    dry_run: bool = typer.Option(False),
):
    collection = CubeCollection(root)
    result = collection.update_layer(
        cube_name=cube_name,
        source_path=source,
        layer_name=layer,
        variable=variable,
        resampling=resampling,
        nodata=nodata,
        missing_value=missing_value,
        dry_run=dry_run,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("collection-overwrite-layer")
def collection_overwrite_layer(
    root: str = typer.Argument(...),
    source: str = typer.Argument(...),
    cube_name: str = typer.Option(...),
    layer: str = typer.Option(...),
    variable: str | None = typer.Option(None),
    resampling: str = typer.Option("bilinear"),
    nodata: float | None = typer.Option(None),
    missing_value: float = typer.Option(-9999.0),
    dry_run: bool = typer.Option(False),
):
    collection = CubeCollection(root)
    result = collection.overwrite_layer(
        cube_name=cube_name,
        source_path=source,
        layer_name=layer,
        variable=variable,
        resampling=resampling,
        nodata=nodata,
        missing_value=missing_value,
        dry_run=dry_run,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("collection-delete-layer")
def collection_delete_layer(
    root: str = typer.Argument(...),
    cube_name: str = typer.Option(...),
    layer: str = typer.Option(...),
):
    collection = CubeCollection(root)
    result = collection.delete_layer(cube_name=cube_name, layer_name=layer)
    typer.echo(json.dumps(result, indent=2))


@app.command("collection-rename-layer")
def collection_rename_layer(
    root: str = typer.Argument(...),
    cube_name: str = typer.Option(...),
    layer: str = typer.Option(...),
    new_layer: str = typer.Option(...),
):
    collection = CubeCollection(root)
    result = collection.rename_layer(
        cube_name=cube_name,
        old_name=layer,
        new_name=new_layer,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("collection-layers")
def collection_layers(
    root: str = typer.Argument(...),
    cube_name: str | None = typer.Option(None),
):
    collection = CubeCollection(root)
    layers = collection.layers(cube_name)

    if cube_name:
        for layer in layers:
            typer.echo(
                f"{layer.name} | cube={layer.cube_name} | region={layer.region} | "
                f"grid={layer.grid_name} | res={layer.resolution_degrees}"
            )
        return

    for cube, cube_layers in layers.items():
        typer.echo(f"\n{cube}")
        for layer in cube_layers:
            typer.echo(f"  - {layer.name} | region={layer.region} | res={layer.resolution_degrees}")


@app.command("provenance")
def provenance_cmd(
    cube: str = typer.Argument(...),
    layer: str = typer.Argument(...),
):
    prov = get_layer_provenance(cube, layer)
    typer.echo(json.dumps(prov, indent=2))

@app.command("manifest")
def manifest_cmd(
    cube: str = typer.Argument(...),
):
    manifest = load_manifest(cube)
    typer.echo(json.dumps(manifest, indent=2))

@app.command("validate-manifest")
def validate_manifest_cmd(
    cube: str = typer.Argument(...),
):
    result = validate_cube_manifest(cube)
    typer.echo(json.dumps(result, indent=2))

    if not result["ok"]:
        raise typer.Exit(code=1)

@app.command("registry")
def registry_cmd(
    cube: str = typer.Argument(...),
):
    registry = load_registry(cube)
    typer.echo(json.dumps(registry, indent=2))

@app.command("layer-history")
def layer_history_cmd(
    cube: str = typer.Argument(...),
    layer: str = typer.Argument(...),
):
    registry = load_registry(cube)
    entry = registry.get("layers", {}).get(layer)

    if not entry:
        raise typer.BadParameter(f"No registry entry found for layer: {layer}")

    typer.echo(json.dumps(entry, indent=2))
