from pathlib import Path
import unittest

import numpy as np
import rasterio


DATA_DIR = Path(__file__).parent / "data"
SOIL_PH_FIXTURE = DATA_DIR / "ph_0-100cm_mean_alaska_test.tif"


class SoilPhFixtureTest(unittest.TestCase):
    def test_alaska_soil_ph_fixture_metadata(self):
        with rasterio.open(SOIL_PH_FIXTURE) as src:
            self.assertEqual(src.crs.to_string(), "EPSG:4326")
            self.assertEqual(src.count, 1)
            self.assertEqual(src.width, 60)
            self.assertEqual(src.height, 60)
            self.assertEqual(src.dtypes[0], "float32")
            self.assertEqual(src.nodata, -9999.0)

            bounds = src.bounds
            self.assertAlmostEqual(bounds.left, -151.0, delta=0.01)
            self.assertAlmostEqual(bounds.right, -150.5, delta=0.01)
            self.assertAlmostEqual(bounds.bottom, 67.5, delta=0.01)
            self.assertAlmostEqual(bounds.top, 68.0, delta=0.01)

            data = src.read(1)

        valid = data[(data != -9999.0) & np.isfinite(data)]
        self.assertEqual(valid.size, 3600)
        self.assertGreater(float(valid.mean()), 5.0)
        self.assertLess(float(valid.mean()), 6.5)
        self.assertGreater(float(valid.max()), 6.0)


if __name__ == "__main__":
    unittest.main()
