import aiohttp
import pytest
from stac_pydantic.shared import MimeTypes

from aiocogeo.errors import MissingAssets
from aiocogeo.stac import STACReader

STAC_ITEM = "https://canada-spot-ortho.s3.amazonaws.com/canada_spot_orthoimages/canada_spot5_orthoimages/S5_2007/S5_11055_6057_20070622/S5_11055_6057_20070622.json"


@pytest.mark.asyncio
async def test_stac_reader():
    async with STACReader(filepath=STAC_ITEM,) as reader:
        assert len(reader.readers) == 5


@pytest.mark.asyncio
async def test_stac_reader_include_types():
    async with STACReader(filepath=STAC_ITEM, include_types={MimeTypes.cog}) as reader:
        assert len(reader.readers) == 5

    with pytest.raises(MissingAssets):
        async with STACReader(filepath=STAC_ITEM, include_types={MimeTypes.geopackage}):
            ...


@pytest.mark.asyncio
async def test_stac_reader_reuse_session():
    async with aiohttp.ClientSession() as session:
        async with STACReader(filepath=STAC_ITEM, kwargs={"session": session}):
            pass
        assert not session.closed
