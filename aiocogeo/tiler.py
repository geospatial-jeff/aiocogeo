from dataclasses import dataclass
from typing import List, Tuple

import morecantile
import numpy as np
from PIL import Image
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform_bounds
from rio_tiler.mercator import zoom_for_pixelsize

from .cog import COGReader


DEFAULT_TMS = morecantile.tms.get("WebMercatorQuad")


@dataclass
class COGInfo:
    min_zoom: int
    max_zoom: int
    bounds: List[float]
    dtype: str
    color_interp: str


@dataclass
class COGTiler:
    cog: COGReader

    def __post_init__(self):
        self.profile = self.cog.profile

    async def _warped_read(
        self,
        bounds: Tuple[int, int, int, int],
        width: int,
        height: int,
        bounds_crs: CRS,
        resample_method: int = Image.NEAREST,
    ) -> np.ndarray:
        src_transform = from_bounds(*bounds, width=width, height=height)
        bounds = transform_bounds(bounds_crs, CRS.from_epsg(self.cog.epsg), *bounds)
        dst_transform = from_bounds(*bounds, width=width, height=height)
        arr = await self.cog.read(
            bounds, shape=(width, height), resample_method=resample_method
        )
        arr, _ = reproject(
            arr,
            destination=np.empty((self.profile["count"], width, height)),
            src_transform=dst_transform,
            dst_transform=src_transform,
            src_crs=CRS.from_epsg(self.cog.epsg),
            dst_crs=bounds_crs,
        )
        return arr.astype(self.profile["dtype"])

    async def tile(
        self,
        x: int,
        y: int,
        z: int,
        tile_size: int = 256,
        tms: morecantile.TileMatrixSet = DEFAULT_TMS,
        resample_method: int = Image.NEAREST,
    ) -> np.ndarray:
        tile = morecantile.Tile(x=x, y=y, z=z)
        tile_bounds = tms.xy_bounds(tile)
        width = height = tile_size
        if self.cog.epsg != tms.crs:
            arr = await self._warped_read(
                tile_bounds,
                width,
                height,
                bounds_crs=tms.crs,
                resample_method=resample_method,
            )
        else:
            arr = await self.cog.read(
                tile_bounds, shape=(width, height), resample_method=resample_method
            )
        return arr

    async def info(self) -> COGInfo:
        wgs84_bounds = transform_bounds(
            CRS.from_epsg(self.cog.epsg), CRS.from_epsg(4326), *self.cog.bounds
        )
        mercator_resolution = max(
            self.profile["transform"][0], abs(self.profile["transform"][4])
        )
        max_zoom = zoom_for_pixelsize(mercator_resolution)
        min_zoom = zoom_for_pixelsize(
            mercator_resolution
            * max(self.profile["width"], self.profile["height"])
            / 256
        )

        return COGInfo(
            min_zoom=min_zoom,
            max_zoom=max_zoom,
            bounds=list(wgs84_bounds),
            dtype=self.profile["dtype"],
            color_interp=self.profile["photometric"],
        )
