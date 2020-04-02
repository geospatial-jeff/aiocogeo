from dataclasses import dataclass
import struct
import warnings

from typing import Any, Tuple, Union

from .constants import TIFF_TAGS
from .counter import BytesCounter

@dataclass
class TagType:
    """
    Represents the type of a TIFF tag.  Also responsible for reading the tag since this is dependent on the tag's type.
    """
    format: str
    size: int

    def _read(self, header: BytesCounter, count: int) -> Tuple[Any]:
        return struct.unpack(f"<{count}{self.format}", header.read(self.size * count))

    def _read_tag_value(self, header: BytesCounter) -> Tuple[Tuple[Any], int, int]:
        count = header.read(4, cast_to_int=True)
        length = self.size * count
        if length <= 4:
            value = self._read(header, count)
            header.incr(4-length)
        else:
            value_offset = header.read(4, cast_to_int=True)
            end_of_tag = header._offset
            header.seek(value_offset)
            value = self._read(header, count)
            header.seek(end_of_tag)
        return value, length, count


FIELD_TYPES = {
    1: TagType(format='B', size=1), # TIFFByte
    2: TagType(format='c', size=1), # TIFFascii
    3: TagType(format='H', size=2), # TIFFshort
    4: TagType(format='L', size=4), # TIFFlong
    5: TagType(format='f', size=4), # TIFFrational
    7: TagType(format='B', size=1), # undefined
    12: TagType(format='d', size=8), # TIFFdouble
    16: TagType(format='Q', size=8), # TIFFlong8
}


@dataclass
class Tag:
    code: int
    name: str
    field_type: TagType
    count: int
    length: int
    value: Tuple[Any]

    @classmethod
    def read(cls, header: BytesCounter) -> Union[None, "Tag"]:
        code = header.read(2, cast_to_int=True)
        if code not in TIFF_TAGS:
            warnings.warn(f"TIFF tag {code} is not supported")
            header.incr(10)
            return None
        name = TIFF_TAGS[code]
        field_type = FIELD_TYPES[header.read(2, cast_to_int=True)]
        value, length, count = field_type._read_tag_value(header)
        return Tag(
            code=code,
            name=name,
            field_type=field_type,
            count=count,
            length=length,
            value=value
        )