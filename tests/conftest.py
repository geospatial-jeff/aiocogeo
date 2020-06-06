import os

import aiohttp
import pytest
from typer.testing import CliRunner

from aiocogeo import config
from aiocogeo import COGReader



DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

TEST_DATA = [
    "https://async-cog-reader-test-data.s3.amazonaws.com/0097d134-9be4-47f6-816d-edb77c9ed79e.tif",  # 3 band, JPEG ycbcr
    "https://async-cog-reader-test-data.s3.amazonaws.com/lzw_cog.tif",  # 3 band lzw (RGBA)
    "https://async-cog-reader-test-data.s3.amazonaws.com/webp_cog.tif",
    "s3://async-cog-reader-test-data/lzw_cog.tif",
    "https://async-cog-reader-test-data.s3.amazonaws.com/naip_image.tif",
    "https://async-cog-reader-test-data.s3.amazonaws.com/naip_image_masked.tif",
    "https://async-cog-reader-test-data.s3.amazonaws.com/int16_deflate.tif",
    os.path.join(DATA_DIR, "cog.tif"),
]


@pytest.fixture
async def client_session():
    async with aiohttp.ClientSession() as session:
        yield session


@pytest.fixture
def create_cog_reader(client_session, monkeypatch):
    monkeypatch.setattr(config, "ENABLE_CACHE", False)
    def _create_reader(infile):
        return COGReader(filepath=infile)

    return _create_reader


@pytest.fixture
def cli_runner():
    return CliRunner()