from dataclasses import dataclass, field
import struct
from typing import List, Optional

from .constants import COMPRESSIONS
from .counter import BytesReader
from .errors import InvalidTiffError, TileNotFoundError
from .ifd import IFD

import aiohttp
from .constants import HEADER_OFFSET, SAMPLE_DTYPES
import imagecodecs
import numpy as np
from io import BytesIO
from PIL import Image


# TODO: Move this to a `utils` file.  I imagine we'll have a bunch of compression-specific helper methods
# https://github.com/mapbox/COGDumper/tree/master/cogdumper
def insert_tables(data, tables):
    if tables:
        if data[0] == 0xFF and data[1] == 0xd8:
            # insert tables, first removing the SOI and EOI
            return data[0:2] + tables[2:-2] + data[2:]
        else:
            raise Exception('Missing SOI marker for JPEG tile')
    else:
        # no-op as per the spec, segment contains all of the JPEG data required
        return data


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
        if bytes_reader.read(2) == b'MM':
            bytes_reader._endian = ">"
        version = bytes_reader.read(2, cast_to_int=True)
        if version == 42:
            first_ifd = bytes_reader.read(4, cast_to_int=True)
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
        return (
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

    async def read_header(self):
        next_ifd_offset = 1
        while next_ifd_offset != 0:
            ifd = await IFD.read(self._bytes_reader)
            next_ifd_offset = ifd.next_ifd_offset
            self._bytes_reader.seek(next_ifd_offset)
            self.ifds.append(ifd)

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
        compression = COMPRESSIONS[ifd.Compression.value]

        # Assuming all bands are the same
        dtype = np.dtype(SAMPLE_DTYPES[(ifd.SampleFormat.value[0], ifd.BitsPerSample.value[0])])

        if compression == "lzw":
            decoded = imagecodecs.lzw_decode(tile)
            decoded = np.frombuffer(decoded, dtype).reshape(ifd.TileHeight.value, ifd.TileWidth.value, ifd.SamplesPerPixel.value)
            # Unpredict if there is horizontal differencing
            if ifd.Predictor.value == 2:
                imagecodecs.delta_decode(decoded, out=decoded, axis=-1)
        elif compression == "jpeg":
            jpeg_tables = ifd.JPEGTables
            jpeg_table_bytes = struct.pack(
                f"{self._bytes_reader._endian}{jpeg_tables.count}{jpeg_tables.tag_type.format}", *ifd.JPEGTables.value
            )
            tile = insert_tables(tile, jpeg_table_bytes)
            decoded = imagecodecs.jpeg_decode(tile)
        else:
            raise NotImplementedError(f"{compression} compression is not currently supported")

        return decoded


    async def __aenter__(self):
        await self.read_header()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        ...

    def __iter__(self):
        for ifd in self.ifds:
            yield ifd


