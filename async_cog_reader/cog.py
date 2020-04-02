from dataclasses import dataclass, field
from typing import List, Optional

from .counter import BytesCounter
from .ifd import IFD

import aiohttp


@dataclass
class COGReader:
    filepath: str
    session: Optional[aiohttp.ClientSession] = None
    ifds: Optional[List[IFD]] = field(default_factory=lambda: [])

    _header: Optional[BytesCounter] = None
    _version: Optional[int] = 42
    _big_tiff: Optional[bool] = False

    _session_keep_alive: Optional[bool] = True

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

    async def __aenter__(self):
        if not self.session:
            self._session_keep_alive = False
            self.session = aiohttp.ClientSession()
        self._header = BytesCounter(await self.range_request(0, offset=16384))
        self.read_header()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Don't close session if it was instantiated outside the class
        if not self._session_keep_alive:
            await self.session.close()