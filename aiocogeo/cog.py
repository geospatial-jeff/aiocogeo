import asyncio
from dataclasses import dataclass, field
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin
import uuid

from aiocache import cached, Cache
import affine
import numpy as np

from . import config
from .constants import PHOTOMETRIC
from .errors import InvalidTiffError, TileNotFoundError
from .filesystems import Filesystem
from .ifd import IFD, ImageIFD, MaskIFD
from .partial_reads import PartialReadInterface

logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)

def config_cache(fn: Callable) -> Callable:
    """
    Inject cache config params (https://aiocache.readthedocs.io/en/latest/decorators.html#aiocache.cached)
    """
    def wrap_function(*args, **kwargs):
        kwargs['cache_read'] = kwargs['cache_write'] = config.ENABLE_BLOCK_CACHE
        return fn(*args, **kwargs)
    return wrap_function

@dataclass
class COGReader(PartialReadInterface):
    filepath: str
    ifds: Optional[List[ImageIFD]] = field(default_factory=lambda: [])
    mask_ifds: Optional[List[MaskIFD]] = field(default_factory=lambda: [])

    _version: Optional[int] = 42
    _big_tiff: Optional[bool] = False


    async def __aenter__(self):
        """Open the image and read the header"""
        async with Filesystem.create_from_filepath(self.filepath) as file_reader:
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
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._file_reader._close()

    def __iter__(self):
        """Iterate through image IFDs"""
        for ifd in self.ifds:
            yield ifd

    @property
    def profile(self) -> Dict[str, Any]:
        """Return a rasterio-style image profile"""
        # TODO: Support nodata value
        return {
            "driver": "GTiff",
            "width": self.ifds[0].ImageWidth.value,
            "height": self.ifds[0].ImageHeight.value,
            "count": self.ifds[0].bands,
            "dtype": str(self.ifds[0].dtype),
            "transform": self.geotransform(),
            "blockxsize": self.ifds[0].TileWidth.value,
            "blockysize": self.ifds[0].TileHeight.value,
            "compress": self.ifds[0].compression,
            "interleave": self.ifds[0].interleave,
            "crs": f"EPSG:{self.epsg}",
            "tiled": True,
            "photometric": PHOTOMETRIC[self.ifds[0].PhotometricInterpretation.value],
        }

    @property
    def epsg(self) -> int:
        """Return the EPSG code representing the crs of the image"""
        ifd = self.ifds[0]
        for idx in range(0, len(ifd.GeoKeyDirectoryTag), 4):
            # 2048 is geographic crs
            # 3072 is projected crs
            if ifd.GeoKeyDirectoryTag[idx] in (2048, 3072):
                return ifd.GeoKeyDirectoryTag[idx + 3]

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Return the bounds of the image in native crs"""
        gt = self.geotransform()
        tlx = gt.c
        tly = gt.f
        brx = tlx + (gt.a * self.ifds[0].ImageWidth.value)
        bry = tly + (gt.e * self.ifds[0].ImageHeight.value)
        return (tlx, bry, brx, tly)

    @property
    def overviews(self) -> List[int]:
        """Return decimation factor for each overview (2**zoom)"""
        return [2 ** (ifd + 1) for ifd in range(len(self.ifds) - 1)]

    @property
    def requests(self) -> Dict[str, Union[int, List[Tuple[int]]]]:
        """Return statistics about http requests made during context lifecycle"""
        return {
            'count': self._file_reader._total_requests,
            'byte_count': self._file_reader._total_bytes_requested,
            'ranges': self._file_reader._requested_ranges
        }

    @property
    def is_masked(self) -> bool:
        """Check if the image has an internal mask"""
        return True if self.mask_ifds else False

    async def _read_header(self) -> None:
        """Internal method to read image header and parse into IFDs and Tags"""
        next_ifd_offset = 1
        while next_ifd_offset != 0:
            ifd = await IFD.read(self._file_reader)
            logger.debug(f" Opened {ifd.ImageHeight.value}x{ifd.ImageWidth.value} overview")
            next_ifd_offset = ifd.next_ifd_offset
            self._file_reader.seek(next_ifd_offset)

            if isinstance(ifd, MaskIFD):
                self.mask_ifds.append(ifd)
            else:
                self.ifds.append(ifd)


    def geotransform(self, ovr_level: int = 0) -> affine.Affine:
        """Return the geotransform of the image at a specific overview level (defaults to native resolution)"""
        # Calculate overview for source image
        gt = affine.Affine(
            self.ifds[0].ModelPixelScaleTag[0],
            0.0,
            self.ifds[0].ModelTiepointTag[3],
            0.0,
            -self.ifds[0].ModelPixelScaleTag[1],
            self.ifds[0].ModelTiepointTag[4],
        )
        # Decimate the geotransform if an overview is requested
        if ovr_level > 0:
            bounds = self.bounds
            ifd = self.ifds[ovr_level]
            gt = affine.Affine.translation(bounds[0], bounds[3]) * affine.Affine.scale(
                (bounds[2] - bounds[0]) / ifd.ImageWidth.value,
                (bounds[1] - bounds[3]) / ifd.ImageHeight.value,
            )
        return gt

    @config_cache
    @cached(
        cache=Cache.MEMORY,
        key_builder=lambda fn,*args,**kwargs: f"{args[0].filepath}-{args[1]}-{args[2]}-{args[3]}"
    )
    async def get_tile(self, x: int, y: int, z: int) -> np.ndarray:

        """
        Request an internal image tile at the specified row (x), column (y), and overview (z).  Based on COGDumper:
        https://github.com/mapbox/COGDumper/blob/master/cogdumper/cog_tiles.py#L337-L365
        """
        futures = []
        if z > len(self.ifds):
            raise TileNotFoundError(f"Overview {z} does not exist.")
        ifd = self.ifds[z]
        idx = (y * ifd.tile_count[0]) + x
        if idx > len(ifd.TileOffsets):
            raise TileNotFoundError(f"Tile {x} {y} {z} does not exist")

        # Request the tile
        futures.append(
            asyncio.create_task(ifd._get_tile(x, y))
        )

        # Request the mask
        if self.is_masked:
            mask_ifd = self.mask_ifds[z]
            futures.append(
                asyncio.create_task(mask_ifd._get_tile(x, y))
            )

        tile = await asyncio.gather(*futures)
        if self.is_masked:
            # Apply mask
            tile[1] = np.invert(np.broadcast_to(tile[1], tile[0].shape))
            return np.ma.masked_array(*tile)
        return tile[0]

    async def read(self, bounds: Tuple[float, float, float, float], shape: Tuple[int, int]) -> Union[np.ndarray, np.ma.masked_array]:
        """
        Perform a partial read.  All pixels within the specified bounding box are read from the image and the array is
        resampled to match the desired shape.
        """
        # Determine which tiles intersect the request bounds
        ovr_level = self._get_overview_level(bounds, shape[1], shape[0])
        ifd = self.ifds[ovr_level]
        img_tiles = self._calculate_image_tiles(
            bounds,
            tile_width=ifd.TileWidth.value,
            tile_height=ifd.TileHeight.value,
            band_count=ifd.bands,
            ovr_level=ovr_level,
            dtype=ifd.dtype
        )

        # Request those tiles
        if config.HTTP_MERGE_CONSECUTIVE_RANGES:
            img_arr = await self._request_merged_tiles(img_tiles)
        else:
            img_arr = await self._request_tiles(img_tiles)

        # Postprocess the array (clip to bounds and resize to requested shape)
        postprocessed = self._postprocess(img_arr, img_tiles, shape)

        return postprocessed

    def create_tile_matrix_set(self, identifier: str = None) -> Dict[str, Any]:
        """Create an OGC TileMatrixSet where each TileMatrix corresponds to an overview"""
        matrices = []
        for idx, ifd in enumerate(self.ifds):
            gt = self.geotransform(idx)
            matrix = {
                "identifier": str(len(self.ifds) - idx - 1),
                "topLeftCorner": [gt.c, gt.f],
                "tileWidth": ifd.TileWidth.value,
                "tileHeight": ifd.TileHeight.value,
                "matrixWidth": ifd.tile_count[0],
                "matrixHeight": ifd.tile_count[1],
                "scaleDenominator": gt.a / 0.28e-3,
            }
            matrices.append(matrix)

        tms = {
            "title": f"Tile matrix for {self.filepath}",
            "identifier": identifier or str(uuid.uuid4()),
            "supportedCRS": urljoin(f"http://www.opengis.net", f"/def/crs/EPSG/0/{self.epsg}"),
            "tileMatrix": list(reversed(matrices))
        }
        return tms
