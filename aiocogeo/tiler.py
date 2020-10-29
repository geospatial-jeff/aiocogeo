import abc
import asyncio
from dataclasses import dataclass, field
import math
from typing import Any, Awaitable, Callable, Coroutine, Dict, List, Union, Optional, Sequence, Tuple, Type
import warnings

import numpy as np
from PIL import Image

from .filesystems import Filesystem
from .constants import ColorInterp
from .cog import COGReader
from .errors import InvalidTiffError

# Rasterio <-> PIL resampling modes
Resampling = {
    "nearest": 0,
    "bilinear": 2,
    "cubic": 3,
    "lanczos": 1,
}

try:
    import morecantile
    from morecantile import TileMatrixSet
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds
    from rasterio.rio.overview import get_maximum_overview_level
    from rasterio.warp import calculate_default_transform
    from rasterio.warp import reproject, transform_bounds, transform as transform_coords
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
    band_metadata: List[Tuple[int, Dict]]
    band_descriptions: List[Tuple[int, str]]
    dtype: str
    colorinterp: List[str]
    nodata_type: str
    colormap: Optional[Dict[int, Sequence[int]]] = None


@dataclass
class TileResponse:
    arr: np.ndarray
    mask: Optional[np.ma.masked_array] = None

    def __iter__(self):
        """
        Allow for variable expansion (``arr, mask = TileResponse``)

        """
        for i in (self.arr, self.mask * 255):
            yield i

    @classmethod
    def from_array(cls, array: Union[np.ndarray, np.ma.masked_array]) -> "TileResponse":
        if isinstance(array, np.ma.masked_array):
            return cls(array.data, array.mask)
        return cls(array.data)


@dataclass
class COGTiler(COGReader, AsyncBaseReader):

    tms: TileMatrixSet = field(default=DEFAULT_TMS)

    async def __aenter__(self):
        """Open the image and read the header"""
        async with Filesystem.create_from_filepath(self.filepath, **self.kwargs) as file_reader:
            self._file_reader = file_reader
            if (await file_reader.read(2)) == b"MM":
                file_reader._endian = ">"
            version = await file_reader.read(2, cast_to_int=True)
            if version == 42:
                first_ifd = await file_reader.read(4, cast_to_int=True)
                file_reader.seek(first_ifd)
                await self._read_header()
            elif version == 43:
                raise NotImplementedError("BigTiff is not yet supported")
            else:
                raise InvalidTiffError("Not a valid TIFF")

            self.bounds = transform_bounds(
                CRS.from_epsg(self.epsg), CRS.from_epsg(4326), *self.geo_bounds
            )
            self.minzoom, self.maxzoom = self.get_zooms()

        return self

    def get_zooms(self, tilesize: int = 256) -> Tuple[int, int]:
        """Calculate raster min/max zoom level."""
        dst_affine, w, h = calculate_default_transform(
            CRS.from_epsg(self.epsg),
            self.tms.crs,
            self.profile["width"],
            self.profile["height"],
            *self.geo_bounds,
        )
        resolution = max(abs(dst_affine[0]), abs(dst_affine[4]))
        maxzoom = self.tms.zoom_for_res(resolution)

        overview_level = get_maximum_overview_level(w, h, minsize=tilesize)
        ovr_resolution = resolution * (2 ** overview_level)
        minzoom = self.tms.zoom_for_res(ovr_resolution)

        return minzoom, maxzoom

    async def _warped_read(
        self,
        bounds: Tuple[int, int, int, int],
        width: int,
        height: int,
        bounds_crs: CRS,
        resample_method: int = Image.NEAREST,
    ) -> np.ndarray:
        src_transform = from_bounds(*bounds, width=width, height=height)
        bounds = transform_bounds(bounds_crs, CRS.from_epsg(self.epsg), *bounds)
        dst_transform = from_bounds(*bounds, width=width, height=height)
        arr = await self.read(
            bounds, shape=(width, height), resample_method=resample_method
        )
        arr, _ = reproject(
            arr,
            destination=np.empty((self.profile["count"], width, height)),
            src_transform=dst_transform,
            dst_transform=src_transform,
            src_crs=CRS.from_epsg(self.epsg),
            dst_crs=bounds_crs,
        )
        return arr.astype(self.profile["dtype"])

    async def info(self) -> Dict:
        if self.has_alpha:
            nodata_type = "Alpha"
        elif self.is_masked:
            nodata_type = "Mask"
        elif self.nodata is not None:
            nodata_type = "Nodata"
        else:
            nodata_type = "None"

        # TODO: Figure out where scale, offset, band_metadata, and band_descriptions come from
        band_descr = [(ix, f"band{ix}") for ix in self.indexes]
        band_meta = [(ix, {}) for ix in self.indexes]

        return dict(
            bounds=self.bounds,
            center=self.center,
            minzoom=self.minzoom,
            maxzoom=self.maxzoom,
            band_metadata=band_meta,
            band_descriptions=band_descr,
            dtype=self.profile["dtype"],
            colorinterp=[color.name for color in self.color_interp],
            nodata_type=nodata_type,
            colormap=self.colormap
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
        resampling_method: str = "nearest",
    ) -> Dict:

        hist_options = hist_options or {}

        if self.colormap and not "bins" not in hist_options:
            hist_options["bins"] = [
                k for k, v in self.colormap.items() if v != (0, 0, 0, 255)
            ]

        if isinstance(indexes, int):
            indexes = (indexes,)

        if indexes is None:
            indexes = [idx for idx, b in enumerate(self.color_interp) if b != ColorInterp.alpha]
            if len(indexes) != self.profile['count']:
                warnings.warn(
                    "Alpha band was removed from the output data array"
                )
            indexes = range(self.profile['count'])

        if bounds:
            data = await self.part(bounds, bbox_crs=bounds_crs, width=width, height=height, resampling_method=resampling_method)
        else:
            data = await self.preview(width=width, height=height, max_size=max_size, resampling_method=resampling_method)

        data = np.ma.array(data.arr)

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
        indexes: Optional[Union[int, Sequence]] = None,
        expression: Optional[str] = "",  # placeholder
        resampling_method: str = "nearest",
    ) -> TileResponse:

        if isinstance(indexes, int):
            indexes = (indexes,)

        tile = morecantile.Tile(x=tile_x, y=tile_y, z=tile_z)
        tile_bounds = self.tms.xy_bounds(*tile)

        # resampling_method -> resample_method
        resample_method = Resampling[resampling_method]

        width = height = tilesize
        if self.epsg != self.tms.crs.to_epsg():
            arr = await self._warped_read(
                tile_bounds,
                width,
                height,
                bounds_crs=self.tms.crs,
                resample_method=resample_method,
            )
        else:
            arr = await self.read(
                tile_bounds, shape=(width, height), resample_method=resample_method
            )

        # should be handled at read level
        if indexes:
            arr = arr[[r - 1 for r in indexes]]

        mask = np.zeros((arr.shape[1], arr.shape[2]), dtype=np.int8) + 255

        return arr, mask

    async def part(
        self,
        bbox: Tuple[float, float, float, float],
        bbox_crs: CRS = WGS84,
        width: int = None,
        height: int = None,
        resampling_method: str = "nearest",
    ) -> TileResponse:
        if bbox_crs != CRS.from_epsg(self.epsg):
            bbox = transform_bounds(bbox_crs, CRS.from_epsg(self.epsg), *bbox)

        if not height or not width:
            width = math.ceil((bbox[2] - bbox[0]) / self.profile['transform'].a)
            height = math.ceil((bbox[3] - bbox[1]) / -self.profile['transform'].e)

        # resampling_method -> resample_method
        resample_method = Resampling[resampling_method]

        arr = await self.read(bounds=bbox, shape=(width, height), resample_method=resample_method)
        return TileResponse.from_array(arr)

    async def preview(
        self,
        width: int = None,
        height: int = None,
        max_size: int = 1024,
        resampling_method: str = "nearest",
    ) -> TileResponse:
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

        # resampling_method -> resample_method
        resample_method = Resampling[resampling_method]

        arr = await self.read(
            bounds=self.geo_bounds,
            shape=(width, height),
            resample_method=resample_method,
        )
        return TileResponse.from_array(arr)

    async def point(
        self, lon: float, lat: float, **kwargs: Any
    ) -> List:
        coords = [lon, lat]
        if self.epsg != 4326:
            coords = [pt[0] for pt in transform_coords(
                WGS84, CRS.from_epsg(self.epsg), [coords[0]], [coords[1]]
            )]
        arr = await self.read_point(*coords)
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
