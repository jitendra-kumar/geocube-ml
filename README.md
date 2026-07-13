# GeoCube-ML

GeoCube-ML is a lightweight Python package for building analysis-ready
geospatial predictor cubes for machine learning workflows.

It ingests raster layers such as GeoTIFFs and NetCDF variables, aligns them to a
common grid, stores them as Zarr, tracks provenance and layer history, and
creates a small STAC catalog for discovery.

The package is intended for ecological, environmental, Arctic, remote sensing,
and geospatial ML workflows where many predictor layers must be repeatedly
reprojected, resampled, stacked, queried, and sampled at observation locations.

## Purpose

GeoCube-ML helps separate expensive geospatial preprocessing from model
development.

Instead of reprojecting and clipping every raster for every modeling experiment,
you can process each layer once into a reusable cube collection. Downstream
workflows can then load only the layers they need with `xarray`, use lazy
computation through Dask, and extract values for training tables.

Typical predictor layers include:

- elevation, slope, aspect
- soil pH, clay, sand, carbon, moisture
- vegetation indices
- climate normals or summaries
- land cover and other categorical maps
- hydrology, geology, permafrost, disturbance, or ecological region layers

## Key Features

- Collection-oriented organization for many region/resolution-specific cubes.
- Grid definitions with CRS, extent, resolution, and chunk size.
- Blockwise ingest from GeoTIFF and NetCDF using Rasterio `WarpedVRT`.
- Per-layer Zarr group storage under `cube.zarr/layers/<layer_name>`.
- Layer provenance with source path, SHA256 checksum, description, processing parameters, and software metadata.
- Manifest sidecar for fast layer summaries.
- Registry sidecar for version history and incremental update decisions.
- STAC catalog items for each layer.
- Layer lifecycle operations: update, overwrite, delete, and rename.
- Point extraction utilities for building ML feature tables.
- Standard-library `unittest` test suite with a small Alaska pH raster fixture.

## Collection Layout

A GeoCube-ML collection is a directory containing one or more cube definitions.

```text
my_geocube_collection/
  collection.json

  grids/
    arctic_30sec.json

  cubes/
    arctic_30sec.zarr/
      layers/
        soil_ph/
        elevation/
        annual_precip/
      .geocube_ml_manifest.json
      .geocube_ml_layer_registry.json

  catalog/
    catalog.json
```

Each cube is internally consistent:

- one CRS
- one extent
- one resolution
- one target grid
- many predictor layers

Each layer is stored in its own Zarr group:

```text
cubes/arctic_30sec.zarr/layers/soil_ph
```

This keeps layer updates, deletion, and renaming simpler than storing all
variables in a single root-level Zarr dataset. Read functions retain fallback
support for older root-variable cube stores.

## Installation

For development:

```bash
git clone <repo-url> geocube-ml
cd geocube-ml
python -m pip install -e .
```

The package depends on geospatial Python libraries:

- `xarray`
- `rioxarray`
- `rasterio`
- `dask`
- `zarr`
- `netcdf4`
- `pystac`
- `shapely`
- `geopandas`
- `pandas`
- `numpy`
- `typer[all]`
- `tqdm`

For GDAL/Rasterio-heavy environments, a `conda-forge` environment is usually the
least painful path.

## Command Line Usage

GeoCube-ML installs two equivalent console commands:

```bash
geocube-ml --help
geocube --help
```

If the package has not been installed as an editable package yet, you can run
against a local checkout by setting `PYTHONPATH` and invoking the Typer app:

```bash
PYTHONPATH=/path/to/geocube-ml \
python -c 'from geocube_ml.cli import app; app()' --help
```

### Initialize A Collection

```bash
geocube-ml collection-init /path/to/my_collection
```

This creates the collection directory structure and writes `collection.json`.

### Add A Cube

Example: Arctic cube north of 60N at 30 arc-second resolution.

```bash
geocube-ml collection-add-cube /path/to/arctic_geocubes \
  --name arctic_30sec \
  --region arctic \
  --resolution-label 30sec \
  --resolution 0.0083333333 \
  --xmin -180 \
  --xmax 180 \
  --ymin 60 \
  --ymax 90 \
  --chunks-y 515 \
  --chunks-x 512 \
  --crs EPSG:4326 \
  --description "Collection of datasets for Arctic north of 60N at 30 sec resolution."
```

### Ingest One GeoTIFF Layer

```bash
geocube-ml collection-ingest /path/to/arctic_geocubes \
  /path/to/ph_0-100cm_mean.tif \
  --cube-name arctic_30sec \
  --layer soil_ph \
  --description "Mean soil pH from 0 to 100 cm depth." \
  --resampling bilinear \
  --missing-value -9999
```

Recommended resampling:

- `nearest` for categorical maps such as land cover or geology class.
- `bilinear` for continuous layers such as pH, elevation, temperature, or NDVI.
- `average` for aggregating fine-resolution continuous/fractional layers.
- `mode` for aggregating categorical layers when appropriate.

### Ingest A NetCDF Variable

```bash
geocube-ml collection-ingest /path/to/my_collection \
  /path/to/climate.nc \
  --cube-name arctic_30sec \
  --layer annual_precip \
  --variable precip \
  --description "Annual precipitation summary." \
  --resampling bilinear \
  --missing-value -9999
```

### Batch Ingest A Directory

```bash
geocube-ml collection-ingest-dir /path/to/my_collection \
  /path/to/raw_predictors \
  --cube-name arctic_30sec \
  --pattern "*.tif" \
  --description "Batch-ingested predictor layer." \
  --resampling bilinear \
  --missing-value -9999
```

Batch ingest returns one result per source file and can continue after failures.
When supplied, `--description` is applied to every layer matched by the batch.

### Update A Layer

`collection-update-layer` uses checksum-aware update behavior. If the source
checksum and processing specification are unchanged, the layer is skipped.

```bash
geocube-ml collection-update-layer /path/to/my_collection \
  /path/to/ph_0-100cm_mean.tif \
  --cube-name arctic_30sec \
  --layer soil_ph \
  --description "Mean soil pH from 0 to 100 cm depth." \
  --resampling bilinear \
  --missing-value -9999
```

### Force Overwrite A Layer

```bash
geocube-ml collection-overwrite-layer /path/to/my_collection \
  /path/to/ph_0-100cm_mean.tif \
  --cube-name arctic_30sec \
  --layer soil_ph \
  --description "Mean soil pH from 0 to 100 cm depth." \
  --resampling bilinear \
  --missing-value -9999
```

### Delete A Layer

```bash
geocube-ml collection-delete-layer /path/to/my_collection \
  --cube-name arctic_30sec \
  --layer soil_ph
```

Deletion removes the layer Zarr group, removes the layer from the manifest,
marks it deleted in the registry, and removes the STAC item.

### Rename A Layer

```bash
geocube-ml collection-rename-layer /path/to/my_collection \
  --cube-name arctic_30sec \
  --layer soil_ph \
  --new-layer soil_ph_0_100cm
```

Rename creates a new layer group, removes the old layer data, updates manifest
metadata, writes registry rename events, and replaces the STAC item.

### List Layers

```bash
geocube-ml collection-layers /path/to/my_collection --cube-name arctic_30sec
```

### Inspect Provenance

```bash
geocube-ml provenance \
  /path/to/my_collection/cubes/arctic_30sec.zarr \
  soil_ph
```

### Inspect Manifest

```bash
geocube-ml manifest /path/to/my_collection/cubes/arctic_30sec.zarr
```

### Validate Manifest

```bash
geocube-ml validate-manifest /path/to/my_collection/cubes/arctic_30sec.zarr
```

### Inspect Registry

```bash
geocube-ml registry /path/to/my_collection/cubes/arctic_30sec.zarr
```

### Inspect Layer History

```bash
geocube-ml layer-history \
  /path/to/my_collection/cubes/arctic_30sec.zarr \
  soil_ph
```

## Python API

Create a collection and add a cube:

```python
from geocube_ml.collection import CubeCollection
from geocube_ml.grid import CubeGrid

collection = CubeCollection("/path/to/arctic_geocubes")

grid = CubeGrid(
    name="arctic_30sec",
    resolution=0.0083333333,
    xmin=-180,
    xmax=180,
    ymin=60,
    ymax=90,
    crs="EPSG:4326",
    chunks=(515, 512),
)

collection.add_cube(
    name="arctic_30sec",
    grid=grid,
    region="arctic",
    resolution_label="30sec",
    description="Arctic predictors north of 60N at 30 sec resolution.",
)
```

Ingest a layer:

```python
collection.ingest(
    cube_name="arctic_30sec",
    source_path="/path/to/ph_0-100cm_mean.tif",
    layer_name="soil_ph",
    description="Mean soil pH from 0 to 100 cm depth.",
    resampling="bilinear",
    missing_value=-9999,
)
```

Load selected layers:

```python
ds = collection.load(
    cube_name="arctic_30sec",
    layers=["soil_ph", "elevation"],
)

soil_ph = ds["soil_ph"]
```

Use lifecycle methods:

```python
collection.update_layer(
    cube_name="arctic_30sec",
    source_path="/path/to/ph_0-100cm_mean.tif",
    layer_name="soil_ph",
    description="Mean soil pH from 0 to 100 cm depth.",
)

collection.overwrite_layer(
    cube_name="arctic_30sec",
    source_path="/path/to/ph_0-100cm_mean.tif",
    layer_name="soil_ph",
    description="Mean soil pH from 0 to 100 cm depth.",
)

collection.rename_layer(
    cube_name="arctic_30sec",
    old_name="soil_ph",
    new_name="soil_ph_0_100cm",
)

collection.delete_layer(
    cube_name="arctic_30sec",
    layer_name="soil_ph_0_100cm",
)
```

## Point Extraction

GeoCube-ML can sample cube layers at point observation locations:

```python
import geopandas as gpd
from geocube_ml.extract import extract_points

points = gpd.read_file("observations.gpkg")

training = extract_points(
    cube_path="/path/to/arctic_geocubes/cubes/arctic_30sec.zarr",
    points=points,
    layers=["soil_ph", "elevation", "annual_precip"],
)

training.to_parquet("training_data.parquet")
```

The resulting table can be used with scikit-learn, XGBoost, LightGBM, PyTorch,
TensorFlow, or other ML tooling.

## Metadata And Reproducibility

GeoCube-ML writes three forms of metadata alongside the Zarr data.

### Provenance

Each layer records:

- source path
- source SHA256 checksum
- source NetCDF variable, if applicable
- layer name
- layer description, if supplied
- cube name
- grid name
- region
- CRS, extent, and resolution
- resampling method
- source nodata and output missing value
- ingest timestamp
- software and Python version
- computed statistics and validation result

### Manifest

`cube.zarr/.geocube_ml_manifest.json` summarizes the current cube contents. It
is intended for quick inspection without opening every Zarr group.

### Registry

`cube.zarr/.geocube_ml_layer_registry.json` records layer history. It is used for
checksum-aware incremental updates and records delete/rename events.

### STAC Catalog

The collection-level `catalog/catalog.json` contains one STAC Item per layer.
Each item includes:

- layer name
- layer description, if supplied
- cube name
- region
- grid name
- CRS and resolution
- source path
- Zarr asset
- source asset
- provenance payload
- Zarr group path, such as `layers/soil_ph`

## Tests

The repository includes a standard-library `unittest` suite. Run it with:

```bash
python -m unittest discover -s tests -v
```

The current tests cover:

- small Alaska pH GeoTIFF fixture metadata
- storage path helpers and layer-name validation
- STAC upsert and delete behavior
- registry incremental update, delete, and rename behavior
- collection batch result handling
- collection layer delete side effects across data, manifest, registry, and STAC

Test fixture:

```text
tests/data/ph_0-100cm_mean_alaska_test.tif
```

The fixture was extracted from a SoilGrids pH raster over a small Alaska /
Brooks Range area:

```text
xmin=-151.0
ymin=67.5
xmax=-150.5
ymax=68.0
```

It is 60 x 60 pixels, `EPSG:4326`, `float32`, and uses `-9999` as nodata.

## Development Notes

- Keep source code and data collections separate. The package repo should hold
  code, docs, tests, and small fixtures only. Large Zarr cubes should live in a
  separate working directory.
- `pytest` is not required for the current test suite; `unittest` is enough.
- Some Python 3.13 / Zarr 3 environments may hang on `zarr.open_group(...)` or
  `xarray.Dataset.to_zarr(...)`. If that occurs, test with a stable
  `conda-forge` geospatial stack, preferably Python 3.12, before large ingests.
- The current validation is intentionally lightweight. Broader integration tests
  around full ingest, chunk structure, CRS consistency, and point extraction are
  good next steps.

## Roadmap

Useful next improvements:

- full ingest integration tests using the small Alaska fixture
- stronger CRS and coordinate validation
- dtype support for categorical/integer rasters
- explicit feature-set definitions for ML workflows
- parallel batch ingest
- cloud/object-storage support
- STAC search utilities
- COG export and preview utilities
