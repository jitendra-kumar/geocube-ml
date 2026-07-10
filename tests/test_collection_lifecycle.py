from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from geocube_ml.catalog import upsert_stac_item
from geocube_ml.collection import CubeCollection
from geocube_ml.grid import CubeGrid
from geocube_ml.manifest import load_manifest, update_manifest
from geocube_ml.registry import append_layer_registry_record, load_registry


class CollectionLifecycleTest(unittest.TestCase):
    def test_ingest_dir_captures_ingest_result(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_dir = tmp_path / "sources"
            source_dir.mkdir()
            (source_dir / "soil_ph.tif").touch()
            (source_dir / "ignored.txt").touch()

            collection = CubeCollection(tmp_path / "collection")

            def fake_ingest(**kwargs):
                return {
                    "layer": kwargs["layer_name"],
                    "status": "ingested",
                    "reason": "test",
                    "changed_keys": ["source_sha256"],
                }

            collection.ingest = fake_ingest

            results = collection.ingest_dir(
                cube_name="arctic_30sec",
                source_dir=str(source_dir),
                pattern="*",
            )

            self.assertEqual(
                results,
                [
                    {
                        "source": str(source_dir / "soil_ph.tif"),
                        "layer": "soil_ph",
                        "status": "ingested",
                        "reason": "test",
                        "changed_keys": ["source_sha256"],
                        "error": None,
                    }
                ],
            )

    def test_delete_layer_removes_data_and_updates_sidecars(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            collection = CubeCollection(tmp_path / "collection")
            grid = CubeGrid("arctic_30sec", 1.0, -151.0, 67.5, -150.5, 68.0)
            record = collection.add_cube(
                name="arctic_30sec",
                grid=grid,
                region="arctic",
                resolution_label="test",
            )

            layer_group = Path(record.path) / "layers" / "soil_ph"
            layer_group.mkdir(parents=True)

            update_manifest(
                record.path,
                "soil_ph",
                {
                    "region": "arctic",
                    "cube_name": "arctic_30sec",
                    "grid_name": "arctic_30sec",
                    "crs": "EPSG:4326",
                    "resolution_degrees": 1.0,
                    "extent": [-151.0, 67.5, -150.5, 68.0],
                    "missing_value": -9999.0,
                    "resampling": "bilinear",
                    "source_path": "source.tif",
                },
            )
            append_layer_registry_record(
                record.path,
                "soil_ph",
                build_spec_payload={"source_sha256": "abc"},
                provenance={},
                statistics={},
                validation={},
            )
            upsert_stac_item(
                collection.catalog_dir,
                record.path,
                "arctic_30sec",
                "soil_ph",
                grid,
                tmp_path / "source.tif",
            )

            result = collection.delete_layer("arctic_30sec", "soil_ph")

            self.assertEqual(result["status"], "deleted")
            self.assertFalse(layer_group.exists())
            self.assertNotIn("soil_ph", load_manifest(record.path)["layers"])
            self.assertTrue(load_registry(record.path)["layers"]["soil_ph"]["deleted"])


if __name__ == "__main__":
    unittest.main()
