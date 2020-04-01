from dataclasses import dataclass
from typing import Optional

from .ifd import IFD

import aiohttp

@dataclass
class BytesCounter:
    """
    Duck-typed file-like object.
    """
    data: bytes

    # Counter to keep track of our current offset within `data`
    _offset: int = 0
    _endian: str = 'little'

    def read(self, offset, cast_to_int=False):
        """
        Read <offset> number of bytes past the current `self._offset` and increment `self._offset`.
        """
        data = self.data[self._offset:self._offset+offset]
        self.incr(offset)
        return int.from_bytes(data, self._endian) if cast_to_int else data

    def incr(self, offset):
        """
        Increment the offset.
        """
        self._offset += offset

    def seek(self, offset):
        """
        Change offset position.
        """
        self._offset = offset

    def tell(self):
        """
        Returns current offset position.
        """
        return self._offset


@dataclass
class COGReader:
    filepath: str
    session: Optional[aiohttp.ClientSession] = None

    _header: BytesCounter = None
    _version: int = 42
    _big_tiff: bool = False

    _offset = 0
    _session_keep_alive = True

    async def range_request(self, start, offset):
        range_header = {"Range": f"bytes={start}-{start+offset}"}
        async with self.session.get(self.filepath, headers=range_header) as cog:
            data = await cog.content.read()
        return data

    async def __aenter__(self):
        if not self.session:
            self._session_keep_alive = False
            self.session = aiohttp.ClientSession()

        # Read first ~16kb of file
        self._header = BytesCounter(await self.range_request(0, offset=16384))

        # TODO: Wrap a lot of this in a method to make more readable
        # Read first 4 bytes to determine tiff or bigtiff and byte order
        if self._header.read(2) == b'MM':
            self._header._endian = "big"

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
            print(ifd.tag_count)
            next_ifd_offset = ifd.next_ifd_offset
            self._header.seek(next_ifd_offset)


    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Don't close session if it was instantiated outside the class
        if not self._session_keep_alive:
            await self.session.close()