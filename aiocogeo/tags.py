import abc
import struct
from dataclasses import dataclass
from typing import Any, Optional

from aiocogeo.constants import TIFF_TAGS
from aiocogeo.tag import TAG_TYPES, TagType


@dataclass
class Tag(abc.ABC):

    code: int
    name: str
    count: int
    offset: int
    length: int
    value: bytes
    tag_type: TagType

    @abc.abstractmethod
    async def get_value(self) -> Any:
        ...

    @abc.abstractmethod
    @classmethod
    def read(cls, data: bytes, offset: int):
        ...

    def write(self) -> bytes:
        raise NotImplementedError


@dataclass
class SimpleTag(Tag):
    @classmethod
    def read(cls, data: bytes, offset: int):
        code = data[:2]
        name = TIFF_TAGS[code]
        field_type = TAG_TYPES[2:4]
        count = data[4:8]
        value = data[8:12]
        length = field_type.size * count
        return cls(
            code=code,
            name=name,
            count=count,
            offset=offset,
            length=length,
            value=value,
            tag_type=field_type,
        )

    async def get_value(self) -> Any:
        if self.length > 4:
            raise NotImplementedError
        return struct.unpack(f"<{self.count}{self.tag_type.format}", self.value)


class NewSubfileType(SimpleTag):
    pass


class ImageWidth(SimpleTag):
    pass


class ImageHeight(SimpleTag):
    pass


class BitsPerSample(SimpleTag):
    pass


class Compression(SimpleTag):
    pass


class PhotometricInterpretation(SimpleTag):
    pass


class DocumentName(SimpleTag):
    pass


class ImageDescription(SimpleTag):
    pass


class SamplesPerPixel(SimpleTag):
    pass


class MinSampleValue(SimpleTag):
    pass


class MaxSampleValue(SimpleTag):
    pass


class XResolution(SimpleTag):
    pass


class YResolution(SimpleTag):
    pass


class PlanarConfiguration(SimpleTag):
    pass


class ResolutionUnit(SimpleTag):
    pass


class Software(SimpleTag):
    pass


class DateTime(SimpleTag):
    pass


class Artist(SimpleTag):
    pass


class HostComputer(SimpleTag):
    pass


class Predictor(SimpleTag):
    pass


class ColorMap(SimpleTag):
    pass


class TileWidth(SimpleTag):
    pass


class TileHeight(SimpleTag):
    pass


class TileOffsets(SimpleTag):
    pass


class TileByteCounts(SimpleTag):
    pass


class ExtraSamples(SimpleTag):
    pass


class SampleFormat(SimpleTag):
    pass


class JPEGTables(SimpleTag):
    pass


class Copyright(SimpleTag):
    pass


class ModelPixelScaleTag(SimpleTag):
    pass


class ModelTiepointTag(SimpleTag):
    pass


class GeoKeyDirectoryTag(Tag):
    def read(cls, data: bytes, offset: int):
        raise NotImplementedError


class GdalMetadata(SimpleTag):
    pass


class Nodata(SimpleTag):
    pass
