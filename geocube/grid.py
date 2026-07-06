from dataclasses import dataclass
import numpy as np
import xarray as xr
import rioxarray  # noqa


@dataclass
class CubeGrid:
    name: str
    resolution: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    crs: str = "EPSG:4326"

    def template(self) -> xr.DataArray:
        x = np.arange(self.xmin + self.resolution / 2, self.xmax, self.resolution)
        y = np.arange(self.ymax - self.resolution / 2, self.ymin, -self.resolution)

        arr = xr.DataArray(
            np.full((len(y), len(x)), np.nan, dtype="float32"),
            dims=("y", "x"),
            coords={"y": y, "x": x},
            name="template",
        )
        arr = arr.rio.write_crs(self.crs)
        arr.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
        return arr
