from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import platform


def file_sha256(path: str, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block_size):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class LayerProvenance:
    source_path: str
    source_sha256: str
    source_variable: str | None
    layer_name: str
    description: str | None
    cube_name: str
    grid_name: str
    region: str
    crs: str
    resolution_degrees: float
    extent: list[float]
    resampling: str
    source_nodata: float | None
    missing_value: float
    ingested_at_utc: str
    software: str = "geocube-ml"
    software_version: str = "0.1.0"
    python_version: str = platform.python_version()

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def build_provenance(
    source_path: str,
    layer_name: str,
    cube_name: str,
    grid,
    region: str,
    resampling: str,
    source_variable: str | None = None,
    description: str | None = None,
    source_nodata: float | None = None,
    missing_value: float = -9999.0,
) -> LayerProvenance:
    return LayerProvenance(
        source_path=str(Path(source_path).resolve()),
        source_sha256=file_sha256(source_path),
        source_variable=source_variable,
        layer_name=layer_name,
        description=description,
        cube_name=cube_name,
        grid_name=grid.name,
        region=region,
        crs=grid.crs,
        resolution_degrees=grid.resolution,
        extent=[grid.xmin, grid.ymin, grid.xmax, grid.ymax],
        resampling=resampling,
        source_nodata=source_nodata,
        missing_value=missing_value,
        ingested_at_utc=datetime.now(timezone.utc).isoformat(),
    )
