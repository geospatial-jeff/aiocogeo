import abc
from dataclasses import dataclass

from aiocogeo.tag import TagType


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


class NewSubfileType(Tag):
    def read(cls, data: bytes):
        raise NotImplementedError


class ImageWidth(Tag):
    def read(cls, data: bytes):
        raise NotImplementedError


class ImageHeight(Tag):
    def read(cls, data: bytes):
        raise NotImplementedError


class BitsPerSample(Tag):
    def read(cls, data: bytes):
        raise NotImplementedError


class Compression(Tag):
    def read(cls, data: bytes):
        raise NotImplementedError


class PhotometricInterpretation(Tag):
    def read(cls, data: bytes):
        raise NotImplementedError


class DocumentName(Tag):
    def read(cls, data: bytes):
        raise NotImplementedError


class ImageDescription(Tag):
    def read(cls, data: bytes):
        raise NotImplementedError


class SamplesPerPixel(Tag):
    def read(cls, data: bytes):
        raise NotImplementedError


class MinSampleValue(Tag):
    def read(cls, data: bytes):
        raise NotImplementedError


class MaxSampleValue(Tag):
    def read(cls, data: bytes):
        raise NotImplementedError


class XResolution(Tag):
    def read(cls, data: bytes):
        raise NotImplementedError


class YResolution(Tag):
    def read(cls, data: bytes):
        raise NotImplementedError


class PlanarConfiguration(Tag):
    def read(cls, data: bytes):
        raise NotImplementedError


class ResolutionUnit(Tag):
    def read(cls, data: bytes):
        raise NotImplementedError


class Software(Tag):
    def read(cls, data: bytes):
        raise NotImplementedError


class DateTime(Tag):
    def read(cls, data: bytes):
        raise NotImplementedError


class Artist(Tag):
    def read(cls, data: bytes):
        raise NotImplementedError


class HostComputer(Tag):
    def read(cls, data: bytes):
        raise NotImplementedError


class Predictor(Tag):
    def read(cls, data: bytes):
        raise NotImplementedError


class ColorMap(Tag):
    def read(cls, data: bytes):
        raise NotImplementedError


class TileWidth(Tag):
    def read(cls, data: bytes):
        raise NotImplementedError


class TileHeight(Tag):
    def read(cls, data: bytes):
        raise NotImplementedError


class TileOffsets(Tag):
    def read(cls, data: bytes):
        raise NotImplementedError


class TileByteCounts(Tag):
    def read(cls, data: bytes):
        raise NotImplementedError


class ExtraSamples(Tag):
    def read(cls, data: bytes):
        raise NotImplementedError


class SampleFormat(Tag):
    def read(cls, data: bytes):
        raise NotImplementedError


class JPEGTables(Tag):
    def read(cls, data: bytes):
        raise NotImplementedError


class Copyright(Tag):
    def read(cls, data: bytes):
        raise NotImplementedError


class ModelPixelScaleTag(Tag):
    def read(cls, data: bytes):
        raise NotImplementedError


class ModelTiepointTag(Tag):
    def read(cls, data: bytes):
        raise NotImplementedError


class GeoKeyDirectoryTag(Tag):
    def read(cls, data: bytes):
        raise NotImplementedError


class GdalMetadata(Tag):
    def read(cls, data: bytes):
        raise NotImplementedError


class Nodata(Tag):
    def read(cls, data: bytes):
        raise NotImplementedError
