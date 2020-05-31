import asyncio
import abc
from dataclasses import dataclass
import math
from typing import Dict, Tuple, Union

import numpy as np

from .compression import Compression
from .constants import COMPRESSIONS, INTERLEAVE, SAMPLE_DTYPES
from .errors import TileNotFoundError
from .filesystems import Filesystem
from .tag import Tag

@dataclass
class IFD:
    next_ifd_offset: int
    tag_count: int
    _file_reader: Filesystem

    @staticmethod
    def _is_masked(tiff_tags: Dict[str, Tag]) -> bool:
        """Check if an IFD is masked based on a dictionary of tiff tags"""
        # # https://www.awaresystems.be/imaging/tiff/tifftags/newsubfiletype.html
        # # https://gdal.org/drivers/raster/gtiff.html#internal-nodata-masks
        if "NewSubfileType" in tiff_tags:
            compression = tiff_tags['Compression'].value
            photo_interp = tiff_tags['PhotometricInterpretation'].value
            subfile_type = tiff_tags['NewSubfileType'].value
            if (subfile_type[0] == 1 or subfile_type[2] == 1) and photo_interp == 4 and compression == 8:
                return True
        return False

    @classmethod
    async def read(cls, file_reader: Filesystem) -> Union["ImageIFD", "MaskIFD"]:
        """Read the IFD"""
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

        # Check if mask
        if cls._is_masked(tiff_tags):
            return MaskIFD(next_ifd_offset, tag_count, file_reader, **tiff_tags)
        return ImageIFD(next_ifd_offset, tag_count, file_reader, **tiff_tags)

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
class ImageIFD(OptionalTags, Compression, RequiredTags, IFD):

    @property
    def compression(self) -> str:
        """Return the compression of the IFD"""
        return COMPRESSIONS[self.Compression.value]

    @property
    def bands(self) -> int:
        """Return the number of image bands"""
        return self.SamplesPerPixel.value

    @property
    def dtype(self) -> np.dtype:
        """Return the data type of the IFD"""
        if self.bands == 1:
            return np.dtype(
                SAMPLE_DTYPES[(self.SampleFormat.value, self.BitsPerSample.value)]
            )
        else:
            return np.dtype(
                SAMPLE_DTYPES[(self.SampleFormat.value[0], self.BitsPerSample.value[0])]
            )

    @property
    def interleave(self) -> str:
        """Return the interleave of the IFD"""
        return "band" if self.bands == 1 else INTERLEAVE[self.PlanarConfiguration.value]

    @property
    def is_full_resolution(self) -> bool:
        """Check if the IFD contains a full resolution image (not an overview)"""
        if not self.NewSubfileType:
            return True
        elif self.NewSubfileType.value[0] == 0:
            return False
        return True

    async def _get_tile(self, x: int, y: int) -> np.ndarray:
        """Read the requested tile from the IFD"""
        idx = (y * self.tile_count[0]) + x
        if idx > len(self.TileOffsets):
            raise TileNotFoundError(f"Tile {x} {y} does not exist")
        offset = self.TileOffsets[idx]
        byte_count = self.TileByteCounts[idx] - 1
        img_bytes = await self._file_reader.range_request(offset, byte_count)
        return self._decompress(img_bytes)

    @property
    def tile_count(self) -> Tuple[int, int]:
        """Return the number of x/y tiles in the IFD"""
        return (
            math.ceil(self.ImageWidth.value / float(self.TileWidth.value)),
            math.ceil(self.ImageHeight.value / float(self.TileHeight.value)),
        )

    def __iter__(self):
        """Iterate through TIFF Tags"""
        for (k, v) in self.__dict__.items():
            if k not in ("next_ifd_offset", "tag_count", "_file_reader") and v:
                yield v


class MaskIFD(ImageIFD):

    async def _get_tile(self, x: int, y: int) -> np.ndarray:
        """Read the requested tile from the IFD"""
        idx = (y * self.tile_count[0]) + x
        if idx > len(self.TileOffsets):
            raise TileNotFoundError(f"Tile {x} {y} does not exist")
        offset = self.TileOffsets[idx]
        byte_count = self.TileByteCounts[idx] - 1
        img_bytes = await self._file_reader.range_request(offset, byte_count)
        return self._decompress_mask(img_bytes)