import abc
import asyncio
from dataclasses import dataclass
import math
from typing import Any, Awaitable, Callable, Coroutine, Dict, List, Union, Optional, Sequence, Tuple, Type
import warnings

import numpy as np
from PIL import Image

from .constants import ColorInterp
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
    bounds: Tuple[float, float, float, float]
    center: Tuple[float, float, int]
    minzoom: int
    maxzoom: int
    dtype: str
    colorinterp: List[str]
    nodata_type: str
    colormap: Optional[Dict[int, Sequence[int]]] = None


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

    async def info(self) -> COGInfo:
        # TODO: Make sure this is in EPSG:3857 before getting resolution
        mercator_resolution = max(
            self.profile["transform"][0], abs(self.profile["transform"][4])
        )
        max_zoom = zoom_for_pixelsize(mercator_resolution)
        min_zoom = zoom_for_pixelsize(
            mercator_resolution
            * max(self.profile["width"], self.profile["height"])
            / 256
        )

        if self.cog.has_alpha:
            nodata_type = "Alpha"
        elif self.cog.is_masked:
            nodata_type = "Mask"
        elif self.cog.nodata is not None:
            nodata_type = "Nodata"
        else:
            nodata_type = "None"

        # TODO: Figure out where scale, offset, band_metadata, and band_descriptions come from
        return COGInfo(
            bounds=self.bounds,
            center=self.center,
            minzoom=min_zoom,
            maxzoom=max_zoom,
            dtype=self.profile["dtype"],
            colorinterp=[color.name for color in self.cog.color_interp],
            nodata_type=nodata_type,
            colormap=self.cog.colormap
        )

    async def stats(
        self,
        pmin: float = 2.0,
        pmax: float = 98.0,
        hist_options: Optional[Dict] = None,
        indexes: Optional[Union[Sequence[int], int]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        max_size: int = 1024,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        bounds_crs: CRS = CRS.from_epsg(4326),
        resample_method: int = Image.NEAREST
    ) -> Dict:

        hist_options = hist_options or {}

        if self.cog.colormap and not "bins" not in hist_options:
            hist_options["bins"] = [
                k for k, v in self.cog.colormap.items() if v != (0, 0, 0, 255)
            ]

        if isinstance(indexes, int):
            indexes = (indexes,)

        if indexes is None:
            indexes = [idx for idx, b in enumerate(self.cog.color_interp) if b != ColorInterp.alpha]
            if len(indexes) != self.profile['count']:
                warnings.warn(
                    "Alpha band was removed from the output data array"
                )
            indexes = range(self.profile['count'])

        if bounds:
            data = await self.part(bounds, bounds_crs=bounds_crs, width=width, height=height, resample_method=resample_method)
        else:
            data = await self.preview(width=width, height=height, max_size=max_size, resample_method=resample_method)

        data = np.ma.array(data)

        return {
            indexes[b]: raster_stats(data[b], percentiles=(pmin, pmax), **hist_options)
            for b in range(data.shape[0])
        }

    async def tile(
        self,
        tile_x: int,
        tile_y: int,
        tile_z: int,
        tilesize: int = 256,
        resample_method: int = Image.NEAREST,
        tms: TileMatrixSet = DEFAULT_TMS,
    ) -> Coroutine[Any, Any, Tuple[np.ndarray, np.ndarray]]:
        tile = morecantile.Tile(x=tile_x, y=tile_y, z=tile_z)
        tile_bounds = tms.xy_bounds(tile)
        width = height = tilesize
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
        return arr.data, arr.mask

    async def part(
        self,
        bbox: Tuple[float, float, float, float],
        bbox_crs: CRS = WGS84,
        width: int = None,
        height: int = None,
        resample_method: int = Image.NEAREST
    ) -> np.ndarray:
        if bbox_crs != self.cog.epsg:
            bounds = transform_bounds(bbox_crs, CRS.from_epsg(self.cog.epsg), *bbox)

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

    async def point(
        self, lon: float, lat: float, **kwargs: Any
    ) -> Coroutine[Any, Any, List]:
        coords = [lon, lat]
        if self.cog.epsg != 4326:
            coords = [pt[0] for pt in transform_coords(
                CRS.from_epsg(4326), CRS.from_epsg(self.cog.epsg), [coords[0]], [coords[1]]
            )]
        arr = await self.cog.point(*coords)
        return arr.tolist()





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