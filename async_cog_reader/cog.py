import asyncio
from dataclasses import dataclass, field
from functools import partial
import math
from typing import List, Optional

import aiohttp
import affine
import numpy as np
from skimage.transform import resize

from .constants import COMPRESSIONS, HEADER_OFFSET, INTERLEAVE, PHOTOMETRIC
from .compression import Compressions
from .counter import BytesReader
from .errors import InvalidTiffError, TileNotFoundError
from .ifd import IFD


@dataclass
class COGReader:
    filepath: str
    session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self._session_keep_alive = True
        if not self.session:
            self._session_keep_alive = False
            self.session = aiohttp.ClientSession()
        bytes_reader = BytesReader(b"", self.filepath, self.session)
        bytes_reader.data = await bytes_reader.range_request(0, HEADER_OFFSET)
        if (await bytes_reader.read(2)) == b'MM':
            bytes_reader._endian = ">"
        version = await bytes_reader.read(2, cast_to_int=True)
        if version == 42:
            first_ifd = await bytes_reader.read(4, cast_to_int=True)
            bytes_reader.seek(first_ifd)
            async with COGTiff(
                filepath=self.filepath,
                session=self.session,
                _bytes_reader=bytes_reader,
                _session_keep_alive=self._session_keep_alive
            ) as cog:
                return cog
        elif version == 43:
            async with COGBigTiff() as cog:
                return cog
        else:
            raise InvalidTiffError("Not a valid TIFF")

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Don't close session if it was instantiated outside the class
        if not self._session_keep_alive:
            await self.session.close()


@dataclass
class COGBigTiff(COGReader):

    async def __aenter__(self):
        ...

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        ...

@dataclass
class COGTiff(COGReader):
    ifds: Optional[List[IFD]] = field(default_factory=lambda: [])

    _bytes_reader: Optional[BytesReader] = None
    _version: Optional[int] = 42
    _big_tiff: Optional[bool] = False

    _session_keep_alive: Optional[bool] = True


    @property
    def profile(self):
        # TODO: Support nodata value
        return {
            "driver": "GTiff",
            "width": self.ifds[0].ImageWidth.value,
            "height": self.ifds[0].ImageHeight.value,
            "count": self.ifds[0].SamplesPerPixel.value,
            "dtype": str(self.ifds[0].dtype),
            "transform": self.geotransform(),
            "blockxsize": self.ifds[0].TileWidth.value,
            "blockysize": self.ifds[0].TileHeight.value,
            "compress": COMPRESSIONS[self.ifds[0].Compression.value],
            "interleave": INTERLEAVE[self.ifds[0].PlanarConfiguration.value],
            "crs": f"EPSG:{self.epsg}",
            "tiled": True,
            "photometric": PHOTOMETRIC[self.ifds[0].PhotometricInterpretation.value]
        }

    @property
    def epsg(self):
        ifd = self.ifds[0]
        for idx in range(0, len(ifd.GeoKeyDirectoryTag), 4):
            # 2048 is geographic crs
            # 3072 is projected crs
            if ifd.GeoKeyDirectoryTag[idx] in (2048, 3072):
                return ifd.GeoKeyDirectoryTag[idx+3]

    @property
    def bounds(self):
        gt = self.geotransform()
        tlx = gt.c
        tly = gt.f
        brx = tlx + (gt.a * self.ifds[0].ImageWidth.value)
        bry = tly + (gt.e * self.ifds[0].ImageHeight.value)
        return (tlx, bry, brx, tly)

    @property
    def overviews(self):
        return [2 ** (ifd+1) for ifd in range(len(self.ifds)-1)]

    async def read_header(self):
        next_ifd_offset = 1
        while next_ifd_offset != 0:
            ifd = await IFD.read(self._bytes_reader)
            next_ifd_offset = ifd.next_ifd_offset
            self._bytes_reader.seek(next_ifd_offset)
            self.ifds.append(ifd)

    def geotransform(self, ovr_level: int = 0):
        # Calculate overview for source image
        gt = affine.Affine(
            self.ifds[0].ModelPixelScaleTag[0],
            0.0,
            self.ifds[0].ModelTiepointTag[3],
            0.0,
            -self.ifds[0].ModelPixelScaleTag[1],
            self.ifds[0].ModelTiepointTag[4]
        )
        # Decimate the geotransform if an overview is requested
        if ovr_level > 0:
            bounds = self.bounds
            ifd = self.ifds[ovr_level]
            gt = affine.Affine.translation(bounds[0], bounds[3]) * affine.Affine.scale(
                (bounds[2] - bounds[0]) / ifd.ImageWidth.value, (bounds[1] - bounds[3]) / ifd.ImageHeight.value
            )
        return gt

    def _get_overview_level(self, bounds, width, height):
        """
        https://github.com/cogeotiff/rio-tiler/blob/v2/rio_tiler/utils.py#L79-L135
        """
        src_res = self.geotransform().a
        target_gt = affine.Affine.translation(bounds[0], bounds[3]) * affine.Affine.scale(
            (bounds[2] - bounds[0]) / width, (bounds[1] - bounds[3]) / height
        )
        target_res = target_gt.a

        ovr_level = 0
        if target_res > src_res:
            # Decimated resolution at each overview
            overviews = [src_res * decim for decim in self.overviews]
            for ovr_level in range(ovr_level, len(overviews) - 1):
                ovr_res = src_res if ovr_level == 0 else overviews[ovr_level]
                if (ovr_res < target_res) and (overviews[ovr_level+1] > target_res):
                    break
                if abs(ovr_res - target_res) < 1e-1:
                    break
            else:
                ovr_level = len(overviews) - 1

        return ovr_level

    async def get_tile(self, x: int, y: int, z: int) -> bytes:
        """
        https://github.com/mapbox/COGDumper/blob/master/cogdumper/cog_tiles.py#L337-L365
        """
        if z > len(self.ifds):
            raise TileNotFoundError(f"Overview {z} does not exist.")
        ifd = self.ifds[z]
        idx = (y * ifd.tile_count[0]) + x
        if idx > len(ifd.TileOffsets):
            raise TileNotFoundError(f"Tile {x} {y} {z} does not exist")
        offset = ifd.TileOffsets[idx]
        byte_count = ifd.TileByteCounts[idx] - 1
        tile = await self._bytes_reader.range_request(offset, byte_count)
        decoded = Compressions(ifd, self._bytes_reader, tile).decompress()
        return decoded

    def _calculate_image_tiles(self, bounds, ovr_level):
        geotransform = self.geotransform(ovr_level)
        invgt = ~geotransform
        tile_width = self.ifds[ovr_level].TileWidth.value
        tile_height = self.ifds[ovr_level].TileHeight.value

        # Project request bounds to pixel coordinates relative to geotransform of the overview
        tlx, tly = invgt * (bounds[0], bounds[3])
        brx, bry = invgt * (bounds[2], bounds[1])

        # Calculate tiles
        xmin = math.floor(tlx / tile_width)
        xmax = math.floor(brx / tile_width)
        ymax = math.floor(bry / tile_height)
        ymin = math.floor(tly / tile_height)

        tile_bounds = (xmin * tile_width, ymin * tile_height, (xmax+1) * tile_width, (ymax+1) * tile_height)

        return {
            'tile_ranges': (xmin, ymin, xmax, ymax),
            'tile_bounds': tile_bounds,
            'request_bounds': (tlx, bry, brx, tly),
            'xtransform': (tlx - tile_bounds[0]) / float(tile_bounds[2] - tile_bounds[0]),
            'ytransform': (bry - tile_bounds[1]) / float(tile_bounds[3] - tile_bounds[1])
        }


    @staticmethod
    def stitch_image_tile(fut, fused_arr, idx, idy, tile_width, tile_height):
        img_arr = fut.result()
        fused_arr[idy * tile_height:(idy + 1) * tile_height, idx * tile_width:(idx + 1) * tile_width, :] = img_arr

    async def read(self, bounds, shape):
        # Determine which tiles intersect the request bounds
        ovr_level = self._get_overview_level(bounds, shape[1], shape[0])
        ifd = self.ifds[ovr_level]
        tile_height = ifd.TileHeight.value
        tile_width = ifd.TileWidth.value
        img_tiles = self._calculate_image_tiles(bounds, ovr_level)
        xmin, ymin, xmax, ymax = img_tiles['tile_ranges']

        # Request those tiles
        tile_tasks = []
        fused = np.zeros(((ymax+1-ymin)*tile_height, (xmax+1-xmin)*tile_width, 3)).astype(ifd.dtype)
        for idx, xtile in enumerate(range(xmin, xmax+1)):
            for idy, ytile in enumerate(range(ymin, ymax+1)):
                get_tile_task = asyncio.create_task(self.get_tile(xtile, ytile, ovr_level))
                get_tile_task.add_done_callback(partial(self.stitch_image_tile, fused_arr=fused, idx=idx, idy=idy, tile_width=tile_width, tile_height=tile_height))
                tile_tasks.append(get_tile_task)
        await asyncio.gather(*tile_tasks)

        # Clip the requested tiles to the extent of the request bounds
        request_height = math.floor(img_tiles['request_bounds'][1]-img_tiles['request_bounds'][3])
        request_width = math.floor(img_tiles['request_bounds'][2]-img_tiles['request_bounds'][0])
        yorigin = fused.shape[0] - int(round(fused.shape[0] * img_tiles['ytransform']))
        xorigin = int(round(fused.shape[1] * img_tiles['xtransform']))
        clipped = fused[yorigin:yorigin+request_height, xorigin:xorigin+request_width,:]

        # Resample to match the requested shape
        resized = resize(clipped, output_shape=shape, preserve_range=True, anti_aliasing=True).astype(ifd.dtype)

        return resized


    async def __aenter__(self):
        await self.read_header()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        ...

    def __iter__(self):
        for ifd in self.ifds:
            yield ifd


