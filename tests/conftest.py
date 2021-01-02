import asyncio
from concurrent.futures import ProcessPoolExecutor
import os

import aiohttp
import pytest
from typer.testing import CliRunner

from aiocogeo import config
from aiocogeo import COGReader


asyncio.get_event_loop().set_default_executor(ProcessPoolExecutor())


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

TEST_DATA = [
    "https://async-cog-reader-test-data.s3.amazonaws.com/0097d134-9be4-47f6-816d-edb77c9ed79e.tif",  # 3 band, JPEG ycbcr, 0.6 meter res
    "https://async-cog-reader-test-data.s3.amazonaws.com/lzw_cog.tif",  # 3 band lzw (RGBA)
    "https://async-cog-reader-test-data.s3.amazonaws.com/webp_cog.tif",
    "s3://async-cog-reader-test-data/lzw_cog.tif",
    "https://async-cog-reader-test-data.s3.amazonaws.com/deflate_cog.tif",
    "https://async-cog-reader-test-data.s3.amazonaws.com/naip_image_nodata.tif",
    "https://async-cog-reader-test-data.s3.amazonaws.com/naip_image.tif",
    "https://async-cog-reader-test-data.s3.amazonaws.com/naip_image_masked.tif",
    "http://async-cog-reader-test-data.s3.amazonaws.com/packbits_cog.tif",
    "https://async-cog-reader-test-data.s3.amazonaws.com/cog_alpha_band.tif",
    "https://async-cog-reader-test-data.s3.amazonaws.com/int16_deflate.tif",
    os.path.join(DATA_DIR, "cog.tif"),
]


@pytest.fixture
async def client_session():
    async with aiohttp.ClientSession() as session:
        yield session


@pytest.fixture
def create_cog_reader(client_session, monkeypatch):
    monkeypatch.setattr(config, "ENABLE_BLOCK_CACHE", False)
    monkeypatch.setattr(config, "ENABLE_HEADER_CACHE", False)

    def _create_reader(infile, **kwargs):
        return COGReader(filepath=infile, **kwargs)

    return _create_reader


@pytest.fixture
def cli_runner():
    return CliRunner()
