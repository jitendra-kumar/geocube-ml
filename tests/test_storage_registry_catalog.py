from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pystac

from geocube_ml.catalog import delete_stac_item, upsert_stac_item
from geocube_ml.grid import CubeGrid
from geocube_ml.registry import (
    LayerBuildSpec,
    append_layer_registry_record,
    determine_ingest_action,
    load_registry,
    mark_layer_deleted,
    rename_layer_registry,
)
from geocube_ml.storage import (
    layer_group_name,
    layer_group_path,
    list_layer_group_names,
    remove_layer_group,
    validate_layer_name,
)


class StorageTest(unittest.TestCase):
    def test_layer_group_helpers(self):
        with TemporaryDirectory() as tmp:
            cube_path = Path(tmp) / "cube.zarr"
            (cube_path / "layers" / "b").mkdir(parents=True)
            (cube_path / "layers" / "a").mkdir(parents=True)

            self.assertEqual(layer_group_name("soil_ph"), "layers/soil_ph")
            self.assertEqual(
                layer_group_path(cube_path, "soil_ph"),
                cube_path / "layers" / "soil_ph",
            )
            self.assertEqual(list_layer_group_names(cube_path), ["a", "b"])

            remove_layer_group(cube_path, "a")
            self.assertEqual(list_layer_group_names(cube_path), ["b"])

    def test_invalid_layer_names_are_rejected(self):
        for name in ["", ".", "..", "nested/layer"]:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    validate_layer_name(name)


class RegistryTest(unittest.TestCase):
    def test_delete_marks_layer_non_current_for_incremental_ingest(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cube_path = tmp_path / "cube.zarr"
            source_path = tmp_path / "source.tif"
            source_path.write_bytes(b"source")

            spec = LayerBuildSpec(
                source_path=str(source_path),
                source_variable=None,
                layer_name="soil_ph",
                cube_name="cube",
                grid_name="grid",
                region="arctic",
                crs="EPSG:4326",
                resolution_degrees=1.0,
                extent=[0.0, 0.0, 1.0, 1.0],
                resampling="bilinear",
                source_nodata=None,
                missing_value=-9999.0,
            )

            plan = determine_ingest_action(cube_path, spec)
            self.assertEqual(plan["action"], "ingest")

            append_layer_registry_record(
                cube_path,
                "soil_ph",
                plan["current_payload"],
                provenance={},
                statistics={},
                validation={},
            )

            self.assertEqual(determine_ingest_action(cube_path, spec)["action"], "skip")

            mark_layer_deleted(cube_path, "soil_ph")
            self.assertEqual(determine_ingest_action(cube_path, spec)["action"], "ingest")

    def test_rename_creates_tombstone_and_active_layer(self):
        with TemporaryDirectory() as tmp:
            cube_path = Path(tmp) / "cube.zarr"
            append_layer_registry_record(
                cube_path,
                "old_name",
                build_spec_payload={"source_sha256": "abc"},
                provenance={},
                statistics={},
                validation={},
            )

            rename_layer_registry(cube_path, "old_name", "new_name")
            registry = load_registry(cube_path)

            self.assertTrue(registry["layers"]["old_name"]["deleted"])
            self.assertEqual(registry["layers"]["old_name"]["renamed_to"], "new_name")
            self.assertFalse(registry["layers"]["new_name"]["deleted"])
            self.assertEqual(registry["layers"]["new_name"]["layer_name"], "new_name")


class CatalogTest(unittest.TestCase):
    def test_upsert_replaces_existing_item_and_delete_removes_it(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            stac_dir = tmp_path / "catalog"
            cube_path = tmp_path / "cube.zarr"
            source_path = tmp_path / "source.tif"
            grid = CubeGrid("cube", 1.0, 0.0, 0.0, 1.0, 1.0)

            upsert_stac_item(
                stac_dir,
                cube_path,
                "cube",
                "soil_ph",
                grid,
                source_path,
                provenance={"version": 1},
            )
            upsert_stac_item(
                stac_dir,
                cube_path,
                "cube",
                "soil_ph",
                grid,
                source_path,
                provenance={"version": 2},
            )

            catalog = pystac.Catalog.from_file(str(stac_dir / "catalog.json"))
            items = list(catalog.get_items())
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].id, "cube-soil_ph")
            self.assertEqual(items[0].properties["provenance"], {"version": 2})
            self.assertEqual(items[0].properties["zarr_group"], "layers/soil_ph")

            self.assertTrue(delete_stac_item(stac_dir, "cube", "soil_ph"))
            catalog = pystac.Catalog.from_file(str(stac_dir / "catalog.json"))
            self.assertEqual(list(catalog.get_items()), [])


if __name__ == "__main__":
    unittest.main()
