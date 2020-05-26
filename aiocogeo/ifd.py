from dataclasses import dataclass
import math

import numpy as np

from .compression import Compression
from .constants import COMPRESSIONS, INTERLEAVE, SAMPLE_DTYPES
from .filesystems import Filesystem
from .tag import Tag

@dataclass
class BaseIFD:
    next_ifd_offset: int
    tag_count: int
    _file_reader: Filesystem

@dataclass
class RequiredTags:
    BitsPerSample: Tag
    Compression: Tag
    ImageHeight: Tag
    ImageWidth: Tag
    PhotometricInterpretation: Tag
    PlanarConfiguration: Tag
    SampleFormat: Tag
    SamplesPerPixel: Tag
    TileByteCounts: Tag
    TileHeight: Tag
    TileOffsets: Tag
    TileWidth: Tag

@dataclass
class OptionalTags:
    NewSubfileType: Tag = None
    Predictor: Tag = None
    JPEGTables: Tag = None

    GeoKeyDirectoryTag: Tag = None
    ModelPixelScaleTag: Tag = None
    ModelTiepointTag: Tag = None

@dataclass
class IFD(OptionalTags, Compression, RequiredTags, BaseIFD):

    @property
    def compression(self):
        return COMPRESSIONS[self.Compression.value]

    @property
    def bands(self):
        return self.SamplesPerPixel.value

    @property
    def dtype(self):
        if self.bands == 1:
            return np.dtype(
                SAMPLE_DTYPES[(self.SampleFormat.value, self.BitsPerSample.value)]
            )
        else:
            return np.dtype(
                SAMPLE_DTYPES[(self.SampleFormat.value[0], self.BitsPerSample.value[0])]
            )

    @property
    def interleave(self):
        return "band" if self.bands == 1 else INTERLEAVE[self.PlanarConfiguration.value]

    @property
    def is_full_resolution(self):
        if not self.NewSubfileType:
            return True
        elif self.NewSubfileType.value[0] == 0:
            return False
        return True

    @property
    def is_mask(self):
        # # https://www.awaresystems.be/imaging/tiff/tifftags/newsubfiletype.html
        # # https://gdal.org/drivers/raster/gtiff.html#internal-nodata-masks
        if self.NewSubfileType:
            if self.NewSubfileType.value[2] == 1 and self.PhotometricInterpretation.value == 4 and self.compression == "deflate":
                return True
        return False


    @property
    def tile_count(self):
        return (
            math.ceil(self.ImageWidth.value / float(self.TileWidth.value)),
            math.ceil(self.ImageHeight.value / float(self.TileHeight.value)),
        )

    def __iter__(self):
        for (k, v) in self.__dict__.items():
            if k not in ("next_ifd_offset", "tag_count", "_file_reader") and v:
                yield v

    @classmethod
    async def read(cls, file_reader: Filesystem) -> "IFD":
        ifd_start = file_reader.tell()
        tag_count = await file_reader.read(2, cast_to_int=True)
        tiff_tags = {}
        # Read tags
        for idx in range(tag_count):
            tag = await Tag.read(file_reader)
            if tag:
                tiff_tags[tag.name] = tag
        file_reader.seek(ifd_start + (12 * tag_count) + 2)
        next_ifd_offset = await file_reader.read(4, cast_to_int=True)
        return cls(next_ifd_offset, tag_count, file_reader, **tiff_tags)
