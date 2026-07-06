# GeoCube

> A lightweight framework for building **analysis-ready ancillary data cubes** for ecological, environmental, and geospatial machine learning.

GeoCube converts collections of GeoTIFF and NetCDF layers into standardized, reusable **Zarr data cubes** that can be queried with **xarray** and processed lazily with **Dask**.

It is designed for workflows where many gridded ancillary datasets are repeatedly reprojected, resampled, clipped, stacked, and extracted at ecological point observation locations.

Instead of preprocessing the same rasters for every modeling project, GeoCube lets you process them once, store them in an analysis-ready format, and selectively load only the layers needed for Random Forests, neural networks, XGBoost, spatial prediction, or exploratory analysis.

---

## Important naming note

There is already a Python package named `geocube` on PyPI. That package is maintained by Corteva and focuses on converting vector data into rasterized xarray objects.

This project uses the name **GeoCube** as a project/repository name for an analysis-ready ancillary raster cube workflow. If publishing to PyPI, consider using a distinct package name such as:

* `geocube-ardc`
* `geocube-ml`
* `eco-geocube`
* `ancillary-geocube`

---

## Why GeoCube?

Ecological upscaling and spatial prediction workflows often require many predictor layers:

* elevation
* slope
* aspect
* soil properties
* land cover
* tree cover
* NDVI
* precipitation
* temperature
* geology
* hydrology
* remotely sensed indices

A typical workflow repeatedly performs the same preprocessing:

1. Read source rasters
2. Reproject to a common CRS
3. Resample to a common resolution
4. Clip to a study region
5. Align all layers to the same grid
6. Mask missing values
7. Extract predictor values at observation points
8. Build machine learning tables

GeoCube separates the expensive preprocessing step from model development.

Raw rasters are ingested once into a reusable cube. Modeling workflows then query, load, and extract layers directly from the cube.

---

## Core design

A GeoCube project is organized as a **CubeCollection**.

A collection can contain many region- and resolution-specific cubes.

```text
my_geocube_collection/

  collection.json

  grids/
    global_1km.json
    amazon_1km.json
    amazon_250m.json

  cubes/
    global_1km.zarr
    amazon_1km.zarr
    amazon_250m.zarr

  catalog/
    catalog.json
```

Each individual cube has:

* one CRS
* one extent
* one resolution
* one grid
* many predictor layers

This keeps every cube internally consistent.

---

## CubeCollection abstraction

The `CubeCollection` is the top-level registry for managing many cubes.

It records:

* cube name
* cube path
* grid path
* region label
* resolution label
* description

Example cubes:

```text
global_5km
global_1km
north_america_1km
amazon_1km
amazon_250m
europe_100m
```

This allows the same project to support multiple analysis scales without mixing incompatible grids in one Zarr store.

---

## Data model

Each cube is stored as an `xarray.Dataset` in Zarr format.

Example:

```text
amazon_1km.zarr

Dimensions:
  y
  x

Variables:
  elevation
  slope
  soil_ph
  soil_clay
  annual_precip
  mean_temperature
  landcover
  ndvi
```

All variables share the same:

* `x` coordinates
* `y` coordinates
* CRS
* resolution
* extent
* array shape

This makes the cube immediately suitable for stacking predictors.

---

## Coordinate reference system

GeoCube defaults to WGS84 latitude/longitude:

```text
EPSG:4326
```

Note that EPSG:4326 is the standard WGS84 lat/lon CRS.

---

## Region-specific alignment

Every ingested layer is forced onto the target cube grid.

This means:

* data outside the target region is excluded
* missing data inside the target region is filled with a configured missing value
* output shape always matches the cube grid
* output coordinates always match the cube grid
* resolution always matches the cube grid
* CRS always matches the cube grid

This ensures that a region-specific cube truly represents only the defined region and resolution.

---

## Provenance tracking

GeoCube tracks provenance for every layer.

Each ingested layer stores provenance metadata in the Zarr variable attributes and in the STAC catalog.

Provenance includes:

* source path
* source file SHA256 checksum
* source NetCDF variable, if applicable
* layer name
* cube name
* grid name
* region
* CRS
* extent
* resolution
* resampling method
* source nodata value
* output missing value
* ingest timestamp
* software version
* Python version

This makes each layer reproducible and auditable.

Example provenance:

```json
{
  "source_path": "/data/raw/soil_ph.tif",
  "source_sha256": "abc123...",
  "source_variable": null,
  "layer_name": "soil_ph",
  "cube_name": "amazon_1km",
  "grid_name": "amazon_1km",
  "region": "amazon",
  "crs": "EPSG:4326",
  "resolution_degrees": 0.0083333333,
  "extent": [-80, -25, -45, 10],
  "resampling": "bilinear",
  "source_nodata": -9999,
  "missing_value": -9999,
  "ingested_at_utc": "2026-07-06T12:00:00+00:00",
  "software": "geocube",
  "software_version": "0.1.0",
  "python_version": "3.12.4"
}
```

---

## STAC catalog

GeoCube maintains a lightweight STAC catalog alongside the cubes.

Each layer is represented as a STAC Item with assets for:

* the Zarr cube
* the original source file

The STAC properties include:

* layer name
* cube name
* region
* grid name
* resolution
* CRS
* source path
* provenance metadata

This makes the cube discoverable without opening every Zarr store.

---

## Installation

For development:

```bash
git clone https://github.com/your-org/geocube
cd geocube
pip install -e .
```

Example dependencies:

```toml
[project]
name = "geocube-ardc"
version = "0.1.0"
dependencies = [
  "xarray",
  "rioxarray",
  "rasterio",
  "dask",
  "zarr",
  "netcdf4",
  "pystac",
  "shapely",
  "geopandas",
  "pandas",
  "numpy",
  "typer"
]

[project.scripts]
geocube = "geocube.cli:app"
```

If you publish the package, avoid using `name = "geocube"` unless you intentionally coordinate around the existing PyPI package name.

---

## Quick start

Initialize a collection:

```bash
geocube collection-init my_collection
```

Add a cube:

```bash
geocube collection-add-cube my_collection \
  --name amazon_1km \
  --region amazon \
  --resolution-label 1km \
  --resolution 0.0083333333 \
  --xmin -80 \
  --ymin -25 \
  --xmax -45 \
  --ymax 10 \
  --crs EPSG:4326 \
  --description "Amazon basin 1 km ancillary predictor cube"
```

Ingest a GeoTIFF:

```bash
geocube collection-ingest my_collection soil_ph.tif \
  --cube-name amazon_1km \
  --layer soil_ph \
  --resampling bilinear \
  --missing-value -9999
```

Ingest a NetCDF variable:

```bash
geocube collection-ingest my_collection climate.nc \
  --cube-name amazon_1km \
  --layer annual_precip \
  --variable precip \
  --resampling bilinear \
  --missing-value -9999
```

List layers:

```bash
geocube collection-layers my_collection --cube-name amazon_1km
```

Inspect provenance:

```bash
geocube provenance my_collection/cubes/amazon_1km.zarr soil_ph
```

---

## Python API

Create a collection:

```python
from geocube.collection import CubeCollection
from geocube.grid import CubeGrid

collection = CubeCollection("my_collection")
```

Add a cube:

```python
grid = CubeGrid(
    name="amazon_1km",
    resolution=0.0083333333,
    xmin=-80,
    ymin=-25,
    xmax=-45,
    ymax=10,
    crs="EPSG:4326",
)

collection.add_cube(
    name="amazon_1km",
    grid=grid,
    region="amazon",
    resolution_label="1km",
    description="Amazon basin 1 km ancillary predictor cube",
)
```

Ingest layers:

```python
collection.ingest(
    cube_name="amazon_1km",
    source_path="soil_ph.tif",
    layer_name="soil_ph",
    resampling="bilinear",
    missing_value=-9999,
)

collection.ingest(
    cube_name="amazon_1km",
    source_path="landcover.nc",
    layer_name="landcover",
    variable="lc_class",
    resampling="nearest",
    missing_value=-9999,
)
```

List available layers:

```python
layers = collection.layers("amazon_1km")

for layer in layers:
    print(layer.name, layer.region, layer.resolution_degrees)
```

Load selected layers:

```python
ds = collection.load(
    cube_name="amazon_1km",
    layers=[
        "soil_ph",
        "landcover",
        "annual_precip",
    ],
)

print(ds)
```

Access one layer:

```python
soil = ds["soil_ph"]
```

Trigger computation with Dask:

```python
mean_soil_ph = soil.where(soil != -9999).mean().compute()
```

---

## Query provenance in Python

```python
from geocube.cube import get_layer_provenance

prov = get_layer_provenance(
    "my_collection/cubes/amazon_1km.zarr",
    "soil_ph",
)

print(prov["source_path"])
print(prov["source_sha256"])
print(prov["resampling"])
```

---

## Extract point observations

```python
import geopandas as gpd
from geocube.extract import extract_points

points = gpd.read_file("observations.gpkg")

training = extract_points(
    cube_path="my_collection/cubes/amazon_1km.zarr",
    points=points,
    layers=[
        "soil_ph",
        "landcover",
        "annual_precip",
    ],
)

training.to_parquet("training_data.parquet")
```

The resulting table can be used directly with:

* scikit-learn
* XGBoost
* LightGBM
* PyTorch
* TensorFlow
* statsmodels

---

## Recommended resampling choices

Use `nearest` for categorical data:

* land cover
* geology class
* soil class
* biome
* ecoregion

Use `bilinear` for continuous data:

* elevation
* temperature
* precipitation
* soil pH
* NDVI
* slope

Use `average` when aggregating from finer to coarser grids:

* tree cover percentage
* population density
* fractional vegetation cover

---

## Missing values

GeoCube uses a configurable missing value, defaulting to:

```text
-9999
```

During ingest:

* NaN values are replaced with the missing value
* source nodata is respected when available
* gaps inside the target region are filled with the missing value
* data outside the target region is discarded

For analysis:

```python
da = ds["soil_ph"]
valid = da.where(da != -9999)
```

---

## CLI reference

Initialize a collection:

```bash
geocube collection-init COLLECTION_ROOT
```

Add a cube:

```bash
geocube collection-add-cube COLLECTION_ROOT \
  --name CUBE_NAME \
  --region REGION \
  --resolution-label LABEL \
  --resolution RESOLUTION \
  --xmin XMIN \
  --ymin YMIN \
  --xmax XMAX \
  --ymax YMAX
```

Ingest a source raster:

```bash
geocube collection-ingest COLLECTION_ROOT SOURCE \
  --cube-name CUBE_NAME \
  --layer LAYER_NAME
```

List layers:

```bash
geocube collection-layers COLLECTION_ROOT
```

List layers in one cube:

```bash
geocube collection-layers COLLECTION_ROOT --cube-name CUBE_NAME
```

Show provenance:

```bash
geocube provenance CUBE_PATH LAYER_NAME
```

---

## Example workflow

```bash
geocube collection-init ecological_predictors

geocube collection-add-cube ecological_predictors \
  --name amazon_1km \
  --region amazon \
  --resolution-label 1km \
  --resolution 0.0083333333 \
  --xmin -80 \
  --ymin -25 \
  --xmax -45 \
  --ymax 10

geocube collection-ingest ecological_predictors elevation.tif \
  --cube-name amazon_1km \
  --layer elevation \
  --resampling bilinear

geocube collection-ingest ecological_predictors landcover.tif \
  --cube-name amazon_1km \
  --layer landcover \
  --resampling nearest

geocube collection-layers ecological_predictors --cube-name amazon_1km
```

Then in Python:

```python
from geocube.collection import CubeCollection

collection = CubeCollection("ecological_predictors")

ds = collection.load(
    "amazon_1km",
    layers=["elevation", "landcover"],
)

print(ds)
```

---

## Design philosophy

GeoCube is not intended to replace GIS software.

It is a lightweight bridge between raw geospatial rasters and machine learning workflows.

The goal is simple:

> Process once. Reuse everywhere. Load only what you need.

---

## Roadmap

Planned features:

* parallel batch ingest
* STAC search API
* layer versioning
* time dimension support
* temporal predictor cubes
* cloud object storage support
* COG export
* raster statistics summaries
* ML feature set definitions
* data validation reports
* cube comparison tools
* provenance diffing
* Dask cluster examples
* map preview utilities
