import pytest
import aiohttp

from async_cog_reader import COGReader


TEST_DATA = [
    "https://async-cog-reader-test-data.s3.amazonaws.com/0097d134-9be4-47f6-816d-edb77c9ed79e.tif", # 3 band, JPEG ycbcr
    "https://async-cog-reader-test-data.s3.amazonaws.com/lzw_cog.tif" # 3 band lzw (RGBA)
]

@pytest.fixture
async def client_session():
    async with aiohttp.ClientSession() as session:
        yield session

@pytest.fixture
def create_cog_reader(client_session):
    def _create_reader(infile):
        return COGReader(filepath=infile, session=client_session)
    return _create_reader

