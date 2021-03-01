import abc
from dataclasses import dataclass

from aiocogeo.constants import TIFF_TAGS
from aiocogeo.tag import TAG_TYPES, TagType


@dataclass
class Tag(abc.ABC):

    code: int
    name: str
    count: int
    length: int
    tag_type: TagType

    @abc.abstractmethod
    @classmethod
    def read(cls, data: bytes):
        ...

    def write(self) -> bytes:
        raise NotImplementedError


@dataclass
class SimpleTag(Tag):
    @classmethod
    def read(cls, data: bytes):
        code = data[:2]
        name = TIFF_TAGS[code]
        field_type = TAG_TYPES[2:4]
        count = data[4:8]
        length = field_type.size * count
        return cls(
            code=code, name=name, count=count, length=length, tag_type=field_type
        )


class NewSubfileType(SimpleTag):
    def read(cls, data: bytes):
        raise NotImplementedError


class ImageWidth(SimpleTag):
    def read(cls, data: bytes):
        raise NotImplementedError


class ImageHeight(SimpleTag):
    def read(cls, data: bytes):
        raise NotImplementedError


class BitsPerSample(SimpleTag):
    def read(cls, data: bytes):
        raise NotImplementedError


class Compression(SimpleTag):
    def read(cls, data: bytes):
        raise NotImplementedError


class PhotometricInterpretation(SimpleTag):
    def read(cls, data: bytes):
        raise NotImplementedError


class DocumentName(SimpleTag):
    def read(cls, data: bytes):
        raise NotImplementedError


class ImageDescription(SimpleTag):
    def read(cls, data: bytes):
        raise NotImplementedError


class SamplesPerPixel(SimpleTag):
    def read(cls, data: bytes):
        raise NotImplementedError


class MinSampleValue(SimpleTag):
    def read(cls, data: bytes):
        raise NotImplementedError


class MaxSampleValue(SimpleTag):
    def read(cls, data: bytes):
        raise NotImplementedError


class XResolution(SimpleTag):
    def read(cls, data: bytes):
        raise NotImplementedError


class YResolution(SimpleTag):
    def read(cls, data: bytes):
        raise NotImplementedError


class PlanarConfiguration(SimpleTag):
    def read(cls, data: bytes):
        raise NotImplementedError


class ResolutionUnit(SimpleTag):
    def read(cls, data: bytes):
        raise NotImplementedError


class Software(SimpleTag):
    def read(cls, data: bytes):
        raise NotImplementedError


class DateTime(SimpleTag):
    def read(cls, data: bytes):
        raise NotImplementedError


class Artist(SimpleTag):
    def read(cls, data: bytes):
        raise NotImplementedError


class HostComputer(SimpleTag):
    def read(cls, data: bytes):
        raise NotImplementedError


class Predictor(SimpleTag):
    def read(cls, data: bytes):
        raise NotImplementedError


class ColorMap(SimpleTag):
    def read(cls, data: bytes):
        raise NotImplementedError


class TileWidth(SimpleTag):
    def read(cls, data: bytes):
        raise NotImplementedError


class TileHeight(SimpleTag):
    def read(cls, data: bytes):
        raise NotImplementedError


class TileOffsets(SimpleTag):
    def read(cls, data: bytes):
        raise NotImplementedError


class TileByteCounts(SimpleTag):
    def read(cls, data: bytes):
        raise NotImplementedError


class ExtraSamples(SimpleTag):
    def read(cls, data: bytes):
        raise NotImplementedError


class SampleFormat(SimpleTag):
    def read(cls, data: bytes):
        raise NotImplementedError


class JPEGTables(SimpleTag):
    def read(cls, data: bytes):
        raise NotImplementedError


class Copyright(SimpleTag):
    def read(cls, data: bytes):
        raise NotImplementedError


class ModelPixelScaleTag(SimpleTag):
    def read(cls, data: bytes):
        raise NotImplementedError


class ModelTiepointTag(SimpleTag):
    def read(cls, data: bytes):
        raise NotImplementedError


class GeoKeyDirectoryTag(Tag):
    def read(cls, data: bytes):
        raise NotImplementedError


class GdalMetadata(SimpleTag):
    def read(cls, data: bytes):
        raise NotImplementedError


class Nodata(SimpleTag):
    def read(cls, data: bytes):
        raise NotImplementedError
