from dataclasses import dataclass
import math
from typing import Dict, Optional

import numpy as np

from .constants import INTERLEAVE, SAMPLE_DTYPES
from .counter import BytesReader
from .tag import Tag


@dataclass
class IFD:
    next_ifd_offset: int
    tag_count: int

    # Required tiff tags
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

    NewSubfileType: Optional[Tag] = None
    Predictor: Optional[Tag] = None
    JPEGTables: Optional[Tag] = None

    GeoKeyDirectoryTag: Optional[Tag] = None
    ModelPixelScaleTag: Optional[Tag] = None
    ModelTiepointTag: Optional[Tag] = None

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
    def tile_count(self):
        return (
            math.ceil(self.ImageWidth.value / float(self.TileWidth.value)),
            math.ceil(self.ImageHeight.value / float(self.TileHeight.value)),
        )

    def __iter__(self):
        for (k, v) in self.__dict__.items():
            if k not in ("next_ifd_offset", "tag_count") and v:
                yield v

    @classmethod
    async def read(cls, reader: BytesReader) -> "IFD":
        ifd_start = reader.tell()
        tag_count = await reader.read(2, cast_to_int=True)
        tiff_tags = {}
        # Read tags
        for idx in range(tag_count):
            tag = await Tag.read(reader)
            if tag:
                tiff_tags[tag.name] = tag
        reader.seek(ifd_start + (12 * tag_count) + 2)
        next_ifd_offset = await reader.read(4, cast_to_int=True)
        return cls(next_ifd_offset, tag_count, **tiff_tags)
