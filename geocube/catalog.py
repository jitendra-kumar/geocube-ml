from pathlib import Path
from datetime import datetime, timezone
import pystac
from shapely.geometry import box, mapping


def upsert_stac_item(
    stac_dir: str,
    cube_path: str,
    layer_name: str,
    grid,
    source_path: str,
    region: str | None = None,
):
    stac_dir = Path(stac_dir)
    stac_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = stac_dir / "catalog.json"

    if catalog_path.exists():
        catalog = pystac.Catalog.from_file(str(catalog_path))
    else:
        catalog = pystac.Catalog(
            id="ancillary-cube-catalog",
            description="Lightweight STAC catalog for analysis-ready ancillary Zarr cubes",
        )

    geom = box(grid.xmin, grid.ymin, grid.xmax, grid.ymax)
    bbox = [grid.xmin, grid.ymin, grid.xmax, grid.ymax]

    item = pystac.Item(
        id=f"{grid.name}-{layer_name}",
        geometry=mapping(geom),
        bbox=bbox,
        datetime=datetime.now(timezone.utc),
        properties={
            "layer_name": layer_name,
            "region": region or "unspecified",
            "grid_name": grid.name,
            "resolution_degrees": grid.resolution,
            "crs": grid.crs,
            "source_path": str(source_path),
        },
    )

    item.add_asset(
        "zarr",
        pystac.Asset(
            href=str(Path(cube_path).resolve()),
            media_type="application/vnd+zarr",
            roles=["data"],
            title=f"{layer_name} in {grid.name}",
        ),
    )

    item.add_asset(
        "source",
        pystac.Asset(
            href=str(Path(source_path).resolve()),
            roles=["source"],
            title="Original source file",
        ),
    )

    existing_ids = {child.id for child in catalog.get_items()}
    if item.id not in existing_ids:
        catalog.add_item(item)

    catalog.normalize_hrefs(str(stac_dir))
    catalog.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
