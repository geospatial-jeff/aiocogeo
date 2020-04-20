from dataclasses import dataclass
import struct
import warnings

from typing import Any, Tuple, Union

from .constants import TIFF_TAGS, HEADER_OFFSET
from .counter import BytesReader

@dataclass
class TagType:
    """
    Represents the type of a TIFF tag.  Also responsible for reading the tag since this is dependent on the tag's type.
    """
    format: str
    size: int

    async def _read(self, reader: BytesReader, count: int) -> Tuple[Any]:
        offset = self.size * count
        if reader.tell() + offset > HEADER_OFFSET:
            data = await reader.range_request(reader.tell(), offset-1)
        else:
            data = await reader.read(offset)
        return struct.unpack(f"{reader._endian}{count}{self.format}", data)

    async def _read_tag_value(self, reader: BytesReader) -> Tuple[Tuple[Any], int, int]:
        # 4-8 bytes contain number of tag values
        count = await reader.read(4, cast_to_int=True)
        length = self.size * count
        # 8-12 bytes contain either (a) tag value or (b) offset to tag value
        if length <= 4:
            value = await self._read(reader, count)
            reader.incr(4-length)
        else:
            value_offset = await reader.read(4, cast_to_int=True)
            end_of_tag = reader.tell()
            reader.seek(value_offset)
            value = await self._read(reader, count)
            reader.seek(end_of_tag)
        value = value[0] if count == 1 else value
        return value, length, count


TAG_TYPES = {
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
    tag_type: TagType
    count: int
    length: int
    value: Tuple[Any]

    def __getitem__(self, key):
        return self.value[key]

    def __len__(self):
        return self.count


    @classmethod
    async def read(cls, reader: BytesReader) -> Union[None, "Tag"]:
        # 0-2 bytes of tag contains TAG NUMBER
        code = await reader.read(2, cast_to_int=True)
        if code not in TIFF_TAGS:
            warnings.warn(f"TIFF tag {code} is not supported")
            reader.incr(12)
            return None
        name = TIFF_TAGS[code]
        # 2-4 bytes contain data type
        tag_type = TAG_TYPES[(await reader.read(2, cast_to_int=True))]
        value, length, count = await tag_type._read_tag_value(reader)
        return Tag(
            code=code,
            name=name,
            tag_type=tag_type,
            count=count,
            length=length,
            value=value
        )