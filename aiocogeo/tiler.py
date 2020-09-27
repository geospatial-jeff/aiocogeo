import abc
import asyncio
from dataclasses import dataclass
import math
from typing import Any, Awaitable, Callable, Coroutine, Dict, List, Union, Optional, Sequence, Tuple, Type

import numpy as np
from PIL import Image

from .cog import COGReader


try:
    import morecantile
    from morecantile import TileMatrixSet
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject, transform_bounds, transform as transform_coords
    from rio_tiler.mercator import zoom_for_pixelsize
    from rio_tiler.io import AsyncBaseReader
    from rio_tiler.utils import _stats as raster_stats

    DEFAULT_TMS = morecantile.tms.get("WebMercatorQuad")
    WGS84 = CRS.from_epsg(4326)
except ImportError:
    CRS = None
    DEFAULT_TMS = None
    TileMatrixSet = None
    WGS84 = None


@dataclass
class COGInfo:
    min_zoom: int
    max_zoom: int
    bounds: List[float]
    dtype: str
    color_interp: str


@dataclass
class COGTiler(AsyncBaseReader):
    cog: COGReader

    def __post_init__(self):
        self.profile = self.cog.profile
        self.bounds = transform_bounds(
            CRS.from_epsg(self.cog.epsg), CRS.from_epsg(4326), *self.cog.bounds
        )
        self.minzoom, self.maxzoom = self.calculate_zoom_range()

    def calculate_zoom_range(self) -> Tuple[int, int]:
        mercator_resolution = max(
            self.profile["transform"][0], abs(self.profile["transform"][4])
        )
        max_zoom = zoom_for_pixelsize(mercator_resolution)
        min_zoom = zoom_for_pixelsize(
            mercator_resolution
            * max(self.profile["width"], self.profile["height"])
            / 256
        )
        return min_zoom, max_zoom

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
        tms: TileMatrixSet = DEFAULT_TMS,
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


    async def point(
        self,
        coords: Tuple[float, float],
        coords_crs: CRS = WGS84,
        **kwargs: Any
    ) -> np.ndarray:
        if coords_crs != self.cog.epsg:
            coords = [pt[0] for pt in transform_coords(
                coords_crs, CRS.from_epsg(self.cog.epsg), [coords[0]], [coords[1]]
            )]
        arr = await self.cog.point(*coords)
        return arr


    async def part(
        self,
        bounds: Tuple[float, float, float, float],
        bounds_crs: CRS = WGS84,
        width: int = None,
        height: int = None,
        resample_method: int = Image.NEAREST
    ) -> np.ndarray:
        if bounds_crs != self.cog.epsg:
            bounds = transform_bounds(bounds_crs, CRS.from_epsg(self.cog.epsg), *bounds)

        if not height or not width:
            width = math.ceil((bounds[2] - bounds[0]) / self.profile['transform'].a)
            height = math.ceil((bounds[3] - bounds[1]) / -self.profile['transform'].e)

        arr = await self.cog.read(bounds=bounds, shape=(width, height), resample_method=resample_method)
        return arr

    async def preview(
        self,
        width: int = None,
        height: int = None,
        max_size: int = 1024,
        resample_method: int = Image.NEAREST,
    ):
        # https://github.com/cogeotiff/rio-tiler/blob/master/rio_tiler/reader.py#L293-L303
        if not height and not width:
            if max(self.profile["height"], self.profile["width"]) < max_size:
                height, width = self.profile["height"], self.profile["width"]
            else:
                ratio = self.profile["height"] / self.profile["width"]
                if ratio > 1:
                    height = max_size
                    width = math.ceil(height / ratio)
                else:
                    width = max_size
                    height = math.ceil(width * ratio)
        return await self.cog.read(
            bounds=self.cog.bounds,
            shape=(width, height),
            resample_method=resample_method,
        )

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

    async def stats(
        self,
        pmin: float = 2.0,
        pmax: float = 98.0,
        indexes: Optional[Union[Sequence[int], int]] = None,
        width: int = None,
        height: int = None,
        max_size: int = 1024,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        bounds_crs: CRS = CRS.from_epsg(4326),
        resample_method: int = Image.NEAREST
    ) -> Dict:

        if isinstance(indexes, int):
            indexes = (indexes,)

        if indexes is None:
            # TODO: Remove alpha band
            # indexes = non_alpha_indexes(src_dst)
            # if indexes != src_dst.indexes:
            #     warnings.warn(
            #         "Alpha band was removed from the output data array", AlphaBandWarning
            #     )
            indexes = range(self.profile['count'])

        if bounds:
            data = await self.part(bounds, bounds_crs=bounds_crs, width=width, height=height, resample_method=resample_method)
        else:
            data = await self.preview(width=width, height=height, max_size=max_size, resample_method=resample_method)

        data = np.ma.array(data)

        # TODO: Support histogram options
        hist_options = {}
        return {
            indexes[b]: raster_stats(data[b], percentiles=(pmin, pmax), **hist_options)
            for b in range(data.shape[0])
        }





# @dataclass
# class CompositeTiler(TilerMixin):
#     # TODO: Add reducers
#     readers: List[COGTiler]
#
#     async def apply(self, func: Callable) -> List[Any]:
#         futs = [func(reader) for reader in self.readers]
#         return await asyncio.gather(*futs)
#
#     async def tile(
#         self,
#         x: int,
#         y: int,
#         z: int,
#         tile_size: int = 256,
#         tms: TileMatrixSet = DEFAULT_TMS,
#         resample_method: int = Image.NEAREST,
#     ) -> np.ndarray:
#         return await self.apply(
#             func=lambda r: r.tile(x, y, z, tile_size, tms, resample_method)
#         )
#
#     async def part(
#         self,
#         bounds: Tuple[float, float, float, float],
#         bounds_crs: CRS = WGS84,
#         width: int = None,
#         height: int = None
#     ) -> np.ndarray:
#         return await self.apply(
#             func=lambda r: r.part(bounds, bounds_crs, width, height)
#         )
#
#     async def preview(
#         self,
#         width: int = None,
#         height: int = None,
#         max_size: int = 1024,
#         resample_method: int = Image.NEAREST,
#     ):
#         return await self.apply(
#             func=lambda r: r.preview(width, height, max_size, resample_method)
#         )
#
#     async def info(self) -> COGInfo:
#         return await self.apply(
#             func=lambda f: r.info()
#         )