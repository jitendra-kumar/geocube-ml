# Test Data

`ph_0-100cm_mean_alaska_test.tif` is a small GeoTIFF fixture extracted from:

`/chrysaor/remotesensing/jbk/data/soils/soilgrids/export/ph_0-100cm_mean.tif`

Extraction bounds:

```text
xmin=-151.0
ymin=67.5
xmax=-150.5
ymax=68.0
```

The source CRS is `EPSG:4326`. The fixture is 60 x 60 pixels and uses
`-9999` as its nodata value. The region is a small Brooks Range / Alaska patch
chosen to keep tests fast while preserving realistic Arctic pH values.
