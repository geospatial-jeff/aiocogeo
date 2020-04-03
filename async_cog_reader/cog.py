from dataclasses import dataclass, field
import struct
from typing import List, Optional

from .constants import COMPRESSIONS
from .counter import BytesCounter
from .ifd import IFD

import aiohttp
from rasterio.io import MemoryFile
import lzw


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
    ifds: Optional[List[IFD]] = field(default_factory=lambda: [])

    _header: Optional[BytesCounter] = None
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
            -ifd.ModelPixelScaleTag[1],
            0.0,
            ifd.ModelTiepointTag[4]
        )

    async def range_request(self, start, offset):
        range_header = {"Range": f"bytes={start}-{start+offset}"}
        async with self.session.get(self.filepath, headers=range_header) as cog:
            data = await cog.content.read()
        return data

    def read_header(self):
        if self._header.read(2) == b'MM':
            self._header._endian = ">"
        self.version = self._header.read(2, cast_to_int=True)
        if self.version == 42:
            self._big_tiff = False
            first_ifd = self._header.read(4, cast_to_int=True)
            self._header.seek(first_ifd)
        elif self.version == 43:
            # TODO: Support BIGTIFF (https://github.com/mapbox/COGDumper/blob/master/cogdumper/cog_tiles.py#L233-L241)
            raise NotImplementedError
        else:
            # TODO: Throw custom exception
            raise Exception("Not a valid TIFF")

        # IFD structure is:
        #   - 2 bytes for the number of tags
        #   - 12 bytes for the tag data itself (one for each tag)
        #   - 4 bytes for the offset to next IFD
        # Each IFD is 2 + (12 * N) + 4 bytes
        next_ifd_offset = 1
        while next_ifd_offset != 0:
            ifd = IFD.read(self._header)
            next_ifd_offset = ifd.next_ifd_offset
            self._header.seek(next_ifd_offset)
            self.ifds.append(ifd)

    # https://github.com/mapbox/COGDumper/blob/master/cogdumper/cog_tiles.py#L337-L365
    async def get_tile(self, x: int, y: int, z: int) -> bytes:
        if z > len(self.ifds):
            raise Exception(f"Overview {z} does not exist.")
        ifd = self.ifds[z]
        idx = (y * ifd.tile_count[0]) + x
        if idx > len(ifd.TileOffsets):
            raise Exception(f"Tile {x} {y} {z} does not exist")
        offset = ifd.TileOffsets[idx]
        byte_count = ifd.TileByteCounts[idx] - 1
        tile = await self.range_request(offset, byte_count)
        if COMPRESSIONS[ifd.Compression.value] == "jpeg":
            # fix up jpeg tile with missing quantization tables
            jpeg_tables = ifd.JPEGTables
            # TODO: clean this up
            jpeg_table_bytes = struct.pack(f"{self._header._endian}{jpeg_tables.count}{jpeg_tables.tag_type.format}", *ifd.JPEGTables.value)
            # TODO: read mask ifds
            tile = insert_tables(tile, jpeg_table_bytes)
        return tile

    async def __aenter__(self):
        if not self.session:
            self._session_keep_alive = False
            self.session = aiohttp.ClientSession()
        # Increasing initial offset for now.  We need to request more bytes if the tag value offset is higher than ~16k.
        # TODO: Implement smarter request logic w/ cache
        # self._header = BytesCounter(await self.range_request(0, offset=16384))
        self._header = BytesCounter(await self.range_request(0, offset=85000))
        self.read_header()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Don't close session if it was instantiated outside the class
        if not self._session_keep_alive:
            await self.session.close()

    def __iter__(self):
        for ifd in self.ifds:
            yield ifd