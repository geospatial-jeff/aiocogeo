import pytest

import rasterio
from rasterio import Affine

from async_cog_reader.ifd import IFD
from async_cog_reader.tag import Tag
from async_cog_reader.constants import COMPRESSIONS, INTERLEAVE
from async_cog_reader.errors import InvalidTiffError

from .conftest import TEST_DATA

@pytest.mark.asyncio
@pytest.mark.parametrize("infile", TEST_DATA)
async def test_cog_metadata(infile, create_cog_reader):
    async with create_cog_reader(infile) as cog:
        first_ifd = cog.ifds[0]

        with rasterio.open(infile) as ds:
            profile = ds.profile
            assert profile['width'] == first_ifd.ImageWidth.value
            assert profile['height'] == first_ifd.ImageHeight.value
            assert profile['transform'] == Affine(*cog.geotransform)
            assert profile['blockxsize'] == first_ifd.TileWidth.value
            assert profile['blockysize'] == first_ifd.TileHeight.value
            assert profile['compress'] == COMPRESSIONS[first_ifd.Compression.value]
            assert profile['interleave'] == INTERLEAVE[first_ifd.PlanarConfiguration.value]


@pytest.mark.asyncio
@pytest.mark.parametrize("infile", [TEST_DATA[0]])
async def test_cog_metadata_iter(infile, create_cog_reader):
    async with create_cog_reader(infile) as cog:
        for ifd in cog:
            assert isinstance(ifd, IFD)
            for tag in ifd:
                assert isinstance(tag, Tag)


@pytest.mark.asyncio
async def test_cog_not_a_tiff(create_cog_reader):
    infile = "https://async-cog-reader-test-data.s3.amazonaws.com/not_a_tiff.png"
    with pytest.raises(InvalidTiffError):
        async with create_cog_reader(infile) as cog:
            ...
