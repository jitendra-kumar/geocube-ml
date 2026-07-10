from pathlib import Path
from datetime import datetime, timezone
import pystac
from shapely.geometry import box, mapping

from .storage import layer_group_name


def upsert_stac_item(
    stac_dir: str,
    cube_path: str,
    cube_name: str,
    layer_name: str,
    grid,
    source_path: str,
    region: str | None = None,
    provenance: dict | None = None,
    zarr_group: str | None = None,
):
    stac_dir = Path(stac_dir)
    stac_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = stac_dir / "catalog.json"

    if catalog_path.exists():
        catalog = pystac.Catalog.from_file(str(catalog_path))
    else:
        catalog = pystac.Catalog(
            id="geocube-ml-catalog",
            description="Lightweight STAC catalog for analysis-ready ancillary Zarr cubes",
        )

    geom = box(grid.xmin, grid.ymin, grid.xmax, grid.ymax)
    bbox = [grid.xmin, grid.ymin, grid.xmax, grid.ymax]

    item = pystac.Item(
        id=f"{cube_name}-{layer_name}",
        geometry=mapping(geom),
        bbox=bbox,
        datetime=datetime.now(timezone.utc),
        properties={
            "layer_name": layer_name,
            "cube_name": cube_name,
            "region": region or "unspecified",
            "grid_name": grid.name,
            "resolution_degrees": grid.resolution,
            "crs": grid.crs,
            "source_path": str(source_path),
            "zarr_group": zarr_group or layer_group_name(layer_name),
            "provenance": provenance or {},
        },
    )

    item.add_asset(
        "zarr",
        pystac.Asset(
            href=str(Path(cube_path).resolve()),
            media_type="application/vnd+zarr",
            roles=["data"],
            title=f"{layer_name} in {cube_name}",
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
    if item.id in existing_ids:
        catalog.remove_item(item.id)

    catalog.add_item(item)

    catalog.normalize_hrefs(str(stac_dir))
    catalog.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)


def delete_stac_item(
    stac_dir: str,
    cube_name: str,
    layer_name: str,
) -> bool:
    catalog_path = Path(stac_dir) / "catalog.json"
    if not catalog_path.exists():
        return False

    catalog = pystac.Catalog.from_file(str(catalog_path))
    item_id = f"{cube_name}-{layer_name}"
    existing_ids = {child.id for child in catalog.get_items()}

    if item_id not in existing_ids:
        return False

    catalog.remove_item(item_id)
    catalog.normalize_hrefs(str(catalog_path.parent))
    catalog.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
    return True
