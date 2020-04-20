from dataclasses import dataclass, field
from typing import List, Optional

import aiohttp
import affine
from rasterio.crs import CRS
from rasterio.transform import array_bounds, from_bounds
from rasterio.warp import calculate_default_transform

from .constants import HEADER_OFFSET, WEB_MERCATOR_EPSG
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
            session_keep_alive = False
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
    def geotransform(self):
        # xres, xtilt, tlx, yres, ytilt, tlx
        ifd = self.ifds[0]
        return affine.Affine(
            ifd.ModelPixelScaleTag[0],
            0.0,
            ifd.ModelTiepointTag[3],
            0.0,
            -ifd.ModelPixelScaleTag[1],
            ifd.ModelTiepointTag[4]
        )

    @property
    def epsg(self):
        ifd = self.ifds[0]
        for idx in range(0, len(ifd.GeoKeyDirectoryTag), 4):
            # 2048 is geographic crs
            # 3072 is projected crs
            if ifd.GeoKeyDirectoryTag[idx] in (2048, 3072):
                return ifd.GeoKeyDirectoryTag[idx+3]


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

    def _get_overview_level(self, width, height):
        """
        https://github.com/cogeotiff/rio-tiler/blob/v2/rio_tiler/utils.py#L79-L135
        """
        native_bounds = array_bounds(self.ifds[0].ImageHeight.value, self.ifds[0].ImageWidth.value, self.geotransform)

        proj_transform, _, _ = calculate_default_transform(
            CRS.from_epsg(self.epsg),
            CRS.from_epsg(WEB_MERCATOR_EPSG),
            self.ifds[0].ImageWidth.value,
            self.ifds[0].ImageHeight.value,
            *native_bounds
        )
        native_res = proj_transform.a

        dst_transform = from_bounds(*native_bounds, width, height)
        target_res = dst_transform.a

        ovr_idx = -1
        if target_res > native_res:
            res = [native_res * decim for decim in self.overviews]

            for ovr_idx in range(ovr_idx, len(res) - 1):
                ovrRes = native_res if ovr_idx < 0 else res[ovr_idx]
                nextRes = res[ovr_idx + 1]
                if (ovrRes < target_res) and (nextRes > target_res):
                    break
                if abs(ovrRes - target_res) < 1e-1:
                    break
            else:
                ovr_idx = len(res) - 1
        return ovr_idx

    # https://github.com/mapbox/COGDumper/blob/master/cogdumper/cog_tiles.py#L337-L365
    async def get_tile(self, x: int, y: int, z: int) -> bytes:
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


    async def __aenter__(self):
        await self.read_header()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        ...

    def __iter__(self):
        for ifd in self.ifds:
            yield ifd


