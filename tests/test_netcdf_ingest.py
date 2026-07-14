from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import xarray as xr

from geocube_ml.collection import CubeCollection
from geocube_ml.ingest import (
    list_netcdf_variables,
    rasterio_source_path,
    validate_source_file,
)


def write_test_netcdf(path: Path) -> None:
    lat = np.array([1.5, 0.5], dtype="float32")
    lon = np.array([0.5, 1.5], dtype="float32")
    ds = xr.Dataset(
        {
            "soil_ph": (
                ("lat", "lon"),
                np.array([[6.0, 6.5], [5.5, 5.8]], dtype="float32"),
            ),
            "soil_soc": (
                ("lat", "lon"),
                np.array([[40.0, 42.0], [35.0, 37.0]], dtype="float32"),
            ),
        },
        coords={"lat": lat, "lon": lon},
    )
    ds.to_netcdf(path)


class NetCDFIngestTest(unittest.TestCase):
    def test_missing_source_file_has_clear_error(self):
        with TemporaryDirectory() as tmp:
            missing_path = Path(tmp) / "missing.nc"

            with self.assertRaisesRegex(FileNotFoundError, "Source file does not exist"):
                validate_source_file(missing_path)

            with self.assertRaisesRegex(FileNotFoundError, "Source file does not exist"):
                rasterio_source_path(str(missing_path), variable="ALT")

    def test_list_netcdf_variables_and_require_variable_name(self):
        with TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "soil.nc"
            write_test_netcdf(source_path)

            self.assertEqual(
                list_netcdf_variables(source_path),
                ["soil_ph", "soil_soc"],
            )

            with self.assertRaisesRegex(ValueError, "Available variables"):
                rasterio_source_path(str(source_path))

    def test_ingest_selected_netcdf_variables_dispatches_layer_ingests(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_path = tmp_path / "soil.nc"
            write_test_netcdf(source_path)

            collection = CubeCollection(tmp_path / "collection")
            calls = []

            def fake_ingest(**kwargs):
                calls.append(kwargs)
                return {
                    "layer": kwargs["layer_name"],
                    "status": "ingested",
                    "reason": "test",
                    "changed_keys": ["source_variable"],
                }

            collection.ingest = fake_ingest
            results = collection.ingest_netcdf(
                cube_name="test_cube",
                source_path=str(source_path),
                variables=["soil_ph", "soil_soc"],
                layer_names=["soil_ph_layer", "soil_soc_layer"],
                description="Synthetic soil variables from NetCDF.",
                resampling="bilinear",
                missing_value=-9999.0,
            )

            self.assertEqual(
                [
                    (result["source_variable"], result["layer"], result["status"])
                    for result in results
                ],
                [
                    ("soil_ph", "soil_ph_layer", "ingested"),
                    ("soil_soc", "soil_soc_layer", "ingested"),
                ],
            )

            self.assertEqual(
                [(call["variable"], call["layer_name"]) for call in calls],
                [("soil_ph", "soil_ph_layer"), ("soil_soc", "soil_soc_layer")],
            )
            self.assertEqual(
                calls[0]["description"],
                "Synthetic soil variables from NetCDF.",
            )
            self.assertEqual(
                calls[0]["source_path"],
                str(source_path),
            )

            with self.assertRaisesRegex(ValueError, "Missing NetCDF variables"):
                collection.ingest_netcdf(
                    cube_name="test_cube",
                    source_path=str(source_path),
                    variables=["missing_variable"],
                )


if __name__ == "__main__":
    unittest.main()
