import pytest

from aiocogeo import config


@pytest.mark.asyncio
async def test_block_cache_enabled(create_cog_reader, monkeypatch):
    # Cache is disabled for tests
    monkeypatch.setattr(config, "ENABLE_BLOCK_CACHE", True)
    infile = "https://async-cog-reader-test-data.s3.amazonaws.com/lzw_cog.tif"
    async with create_cog_reader(infile) as cog:
        await cog.get_tile(0, 0, 0)

    async with create_cog_reader(infile) as cog:
        await cog.get_tile(0, 0, 0)
        # Confirm all requests are cached
        assert cog.requests["count"] == 18


@pytest.mark.asyncio
async def test_block_cache_disabled(create_cog_reader):
    infile = "https://async-cog-reader-test-data.s3.amazonaws.com/lzw_cog.tif"
    async with create_cog_reader(infile) as cog:
        await cog.get_tile(0, 0, 0)
        request_count = cog.requests["count"]

        await cog.get_tile(0, 0, 0)
        assert cog.requests["count"] == request_count + 1


@pytest.mark.asyncio
async def test_header_cache_enabled(create_cog_reader, monkeypatch):
    # Cache is disabled for tests
    monkeypatch.setattr(config, "ENABLE_HEADER_CACHE", True)
    infile = "https://async-cog-reader-test-data.s3.amazonaws.com/webp_cog.tif"
    async with create_cog_reader(infile) as cog:
        assert cog.requests["count"] == 20

    async with create_cog_reader(infile) as cog:
        assert cog.requests["count"] == 2

    async with create_cog_reader(infile) as cog:
        await cog.get_tile(0, 0, 0)
        assert cog.requests["count"] == 3


@pytest.mark.asyncio
async def test_header_cache_disabled(create_cog_reader):
    infile = "https://async-cog-reader-test-data.s3.amazonaws.com/webp_cog.tif"
    async with create_cog_reader(infile) as cog:
        assert cog.requests["count"] == 20

    async with create_cog_reader(infile) as cog:
        assert cog.requests["count"] == 20