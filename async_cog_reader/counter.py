from dataclasses import dataclass

import aiohttp

@dataclass
class BytesReader:
    """
    Duck-typed file-like object.
    """
    data: bytes
    filepath: str
    session: aiohttp.ClientSession

    # Counter to keep track of our current offset within `data`
    _offset: int = 0
    _endian: str = '<'


    async def range_request(self, start, offset):
        range_header = {"Range": f"bytes={start}-{start + offset}"}
        async with self.session.get(self.filepath, headers=range_header) as cog:
            data = await cog.content.read()
        return data

    def read(self, offset, cast_to_int=False):
        """
        Read <offset> number of bytes past the current `self._offset` and increment `self._offset`.
        """
        data = self.data[self._offset:self._offset+offset]
        self.incr(offset)
        order = 'little' if self._endian == '<' else 'big'
        return int.from_bytes(data, order) if cast_to_int else data

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