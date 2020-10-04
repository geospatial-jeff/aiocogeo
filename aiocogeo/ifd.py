from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple, Union

import numpy as np
import xmltodict

from .compression import Compression
from .constants import COMPRESSIONS, GDAL_METADATA_TAGS, INTERLEAVE, RASTER_TYPE, SAMPLE_DTYPES
from .filesystems import Filesystem
from .tag import GeoKeyDirectory, Tag
from .utils import run_in_background

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

        if 'GeoKeyDirectoryTag' in tiff_tags:
            tiff_tags['geo_keys'] = GeoKeyDirectory.read(tiff_tags['GeoKeyDirectoryTag'])

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
    # TIFF standard tags
    NewSubfileType: Tag = None
    Predictor: Tag = None
    JPEGTables: Tag = None
    ExtraSamples: Tag = None
    ColorMap: Tag = None
    ImageDescription: Tag = None
    DocumentName: Tag = None
    Software: Tag = None
    DateTime: Tag = None
    Artist: Tag = None
    HostComputer: Tag = None
    Copyright: Tag = None
    XResolution: Tag = None
    YResolution: Tag = None
    ResolutionUnit: Tag = None
    MinSampleValue: Tag = None
    MaxSampleValue: Tag = None

    # GeoTiff
    GeoKeyDirectoryTag: Tag = None
    ModelPixelScaleTag: Tag = None
    ModelTiepointTag: Tag = None

    # GDAL private tags
    NoData: Tag = None
    GdalMetadata: Tag = None


@dataclass
class ImageIFD(OptionalTags, Compression, RequiredTags, IFD):
    _is_alpha: bool = False
    geo_keys: Optional[GeoKeyDirectory] = None

    @property
    def is_alpha(self) -> bool:
        """Return if the ifd is an alpha band"""
        return self._is_alpha

    @is_alpha.setter
    def is_alpha(self, value):
        """is_alpha setter"""
        self._is_alpha = value

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
    def nodata(self) -> Optional[int]:
        return int(self.NoData.value[0]) if self.NoData else None

    @property
    def has_extra_samples(self):
        return True if self.ExtraSamples else False


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
        offset = self.TileOffsets[idx]
        byte_count = self.TileByteCounts[idx] - 1
        img_bytes = await self._file_reader.range_request(offset, byte_count)
        return await run_in_background(self._decompress, img_bytes)

    @property
    def tile_count(self) -> Tuple[int, int]:
        """Return the number of x/y tiles in the IFD"""
        return (
            math.ceil(self.ImageWidth.value / float(self.TileWidth.value)),
            math.ceil(self.ImageHeight.value / float(self.TileHeight.value)),
        )

    @property
    def gdal_metadata(self) -> Dict:
        """Return gdal metadata"""
        meta = {}
        for tag in GDAL_METADATA_TAGS:
            inst = getattr(self, tag)
            if inst is not None:
                if isinstance(inst.value, tuple):
                    # TODO: Maybe we are reading one extra byte
                    val = b"".join(inst.value)[:-1].decode('utf-8')
                else:
                    val = inst.value
                meta[f"TIFFTAG_{tag.upper()}"] = val

        if self.GdalMetadata:
            xml = b''.join(self.GdalMetadata.value[:-1]).decode('utf-8')
            parsed = xmltodict.parse(xml)
            tags = parsed['GDALMetadata']['Item']
            if isinstance(tags, list):
                meta.update({tag['@name']:tag['#text'] for tag in tags})
            else:
                meta.update({tags['@name']:tags['#text']})

        if self.geo_keys:
            meta['AREA_OR_POINT'] = RASTER_TYPE[self.geo_keys.RasterType.value]

        return meta

    def __iter__(self):
        """Iterate through TIFF Tags"""
        for (k, v) in self.__dict__.items():
            if k not in ("next_ifd_offset", "tag_count", "_file_reader", "geo_keys") and v:
                yield v


class MaskIFD(ImageIFD):

    async def _get_tile(self, x: int, y: int) -> np.ndarray:
        """Read the requested tile from the IFD"""
        idx = (y * self.tile_count[0]) + x
        offset = self.TileOffsets[idx]
        byte_count = self.TileByteCounts[idx] - 1
        img_bytes = await self._file_reader.range_request(offset, byte_count)
        return await run_in_background(self._decompress_mask, img_bytes)