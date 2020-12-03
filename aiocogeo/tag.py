from dataclasses import dataclass
import logging
import struct

from typing import Any, Optional, Tuple, Union

from . import config
from .constants import GEO_KEYS, TIFF_TAGS
from .filesystems import Filesystem


logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


@dataclass
class TagType:
    """
    Represents the type of a TIFF tag.  Also responsible for reading the tag since this is dependent on the tag's type.
    """

    format: str
    size: int


TAG_TYPES = {
    1: TagType(format="B", size=1),  # TIFFByte
    2: TagType(format="c", size=1),  # TIFFascii
    3: TagType(format="H", size=2),  # TIFFshort
    4: TagType(format="L", size=4),  # TIFFlong
    5: TagType(format="f", size=4),  # TIFFrational
    7: TagType(format="B", size=1),  # undefined
    12: TagType(format="d", size=8),  # TIFFdouble
    16: TagType(format="Q", size=8),  # TIFFlong8
}

@dataclass
class BaseTag:
    code: int
    name: str
    count: int

@dataclass
class Tag(BaseTag):
    tag_type: TagType
    length: int
    value: Union[Any, Tuple[Any]]

    def __getitem__(self, key):
        return self.value[key]

    def __len__(self):
        return self.count

    @classmethod
    async def read(cls, reader: Filesystem) -> Optional["Tag"]:
        """Read a TIFF Tag"""
        # 0-2 bytes of tag are tag name
        code = await reader.read(2, cast_to_int=True)
        if code not in TIFF_TAGS:
            logger.warning(f"TIFF TAG {code} is not supported.")
            reader.incr(10)
            return None
        name = TIFF_TAGS[code]
        # 2-4 bytes are field type
        field_type = TAG_TYPES[(await reader.read(2, cast_to_int=True))]
        # 4-8 bytes are number of values
        count = await reader.read(4, cast_to_int=True)
        length = field_type.size * count
        if length <= 4:
            data = await reader.read(length)
            value = struct.unpack(f"{reader._endian}{count}{field_type.format}", data)
            reader.incr(4 - length)
            # Interpret both bits of NewSubfileType independently, even though the tiff spec says there is
            # a single value If we need to keep adding more custom logic here for specific tags we should switch to
            # something more declarative where each tag defines how to read its data.
            if name == "NewSubfileType":
                bit32 = '{:032b}'.format(value[0])
                value = [[int(x) for x in str(int(bit32)).zfill(3)]]

        else:
            # value is elsewhere in the file, `value_offset` tells us where it is
            value_offset = await reader.read(4, cast_to_int=True)
            end_of_tag = reader.tell()

            # read more data if we need to
            current_size = len(reader.data)
            if value_offset + length > current_size:

                # we coerce the chunk size to be big enough to read the full tag value
                chunk_size = max(value_offset + length - current_size, config.HEADER_CHUNK_SIZE)
                reader.data += await reader.range_request(len(reader.data), chunk_size, is_header=True)

            # read the tag value
            reader.seek(value_offset)
            data = await reader.read(length)
            value = struct.unpack(f"{reader._endian}{count}{field_type.format}", data)

            reader.seek(end_of_tag)
        value = value[0] if count == 1 else value

        tag = Tag(
            code=code,
            name=name,
            tag_type=field_type,
            count=count,
            length=length,
            value=value,
        )
        return tag


@dataclass
class GeoKey(BaseTag):
    """http://docs.opengeospatial.org/is/19-008r4/19-008r4.html#_geokey"""
    tag_location: int
    value: Any

    @classmethod
    def read(cls, key: Tuple[int, int, int, int]):
        return cls(
            code=key[0],
            tag_location=key[1],
            count=key[2],
            value=key[3],
            name=GEO_KEYS[key[0]]
        )


@dataclass
class GeoKeyDirectory:
    """http://docs.opengeospatial.org/is/19-008r4/19-008r4.html#_requirements_class_geokeydirectorytag"""
    RasterType: GeoKey
    GeographicType: Optional[GeoKey] = None
    ProjectedType: Optional[GeoKey] = None

    @classmethod
    def read(cls, tag: Tag) -> "GeoKeyDirectory":
        """Parse GeoKeyDirectoryTag"""
        geokeys = {}
        assert tag.name == 'GeoKeyDirectoryTag'
        for idx in range(0, len(tag), 4):
            if tag[idx] in list(GEO_KEYS):
                geokeys[GEO_KEYS[tag[idx]]] = GeoKey.read(tag[idx:idx+4])
        return cls(**geokeys)

    @property
    def epsg(self) -> int:
        """Return the EPSG code representing the crs of the image"""
        return self.ProjectedType.value if self.ProjectedType else self.GeographicType.value