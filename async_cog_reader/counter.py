from dataclasses import dataclass

@dataclass
class BytesCounter:
    """
    Duck-typed file-like object.
    """
    data: bytes

    # Counter to keep track of our current offset within `data`
    _offset: int = 0
    _endian: str = '<'

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