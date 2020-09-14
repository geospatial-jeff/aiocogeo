from dataclasses import dataclass

import morecantile
import numpy as np
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform_bounds

from .cog import COGReader


DEFAULT_TMS = morecantile.tms.get("WebMercatorQuad")


@dataclass
class COGTiler:
    cog: COGReader

    def __post_init__(self):
        self.profile = self.cog.profile

    async def _warped_read(self, bounds, width, height, out_crs: CRS):
        src_transform = from_bounds(*bounds, width=width, height=height)
        bounds = transform_bounds(
            out_crs,
            CRS.from_epsg(self.cog.epsg),
            *bounds
        )
        dst_transform = from_bounds(*bounds, width=width, height=height)
        arr = await self.cog.read(bounds, shape=(width, height))
        arr, _ = reproject(
            arr,
            destination=np.empty((self.profile['count'], width, height)),
            src_transform=dst_transform,
            dst_transform=src_transform,
            src_crs=CRS.from_epsg(self.cog.epsg),
            dst_crs=out_crs
        )
        return arr.astype(self.profile['dtype'])


    async def tile(self, x: int, y: int, z: int, tile_size: int = 256, tms: morecantile.TileMatrixSet = DEFAULT_TMS):
        tile = morecantile.Tile(x=x, y=y, z=z)
        tile_bounds = tms.xy_bounds(tile)
        width = height = tile_size
        if self.cog.epsg != tms.crs:
            arr = await self._warped_read(tile_bounds, width, height, tms.crs)
        else:
            arr = await self.cog.read(tile_bounds, (width, height))
        return arr