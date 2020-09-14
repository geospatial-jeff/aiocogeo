from dataclasses import dataclass

import mercantile
import numpy as np
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform_bounds

from .cog import COGReader


@dataclass
class COGTiler:
    cog: COGReader

    def __post_init__(self):
        self.profile = self.cog.profile

    async def _warped_read(self, bounds, width, height):
        src_transform = from_bounds(*bounds, width=width, height=height)
        bounds = transform_bounds(
            CRS.from_epsg(3857),
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
            dst_crs=CRS.from_epsg(3857)
        )
        return arr.astype(self.profile['dtype'])


    async def tile(self, x: int, y: int, z: int, tile_size: int = 256):
        tile = mercantile.Tile(x=x, y=y, z=z)
        tile_bounds = mercantile.xy_bounds(tile)
        width = height = tile_size
        if self.cog.epsg != 3857:
            arr = await self._warped_read(tile_bounds, width, height)
        else:
            arr = await self.cog.read(tile_bounds, (width, height))
        return arr