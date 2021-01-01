import asyncio

import pytest

from aiocogeo import COGReader, CompositeReader


@pytest.fixture
async def readers():
    readers = await asyncio.gather(
        *[
            COGReader(
                "https://async-cog-reader-test-data.s3.amazonaws.com/webp_cog.tif"
            ).__aenter__()
            for _ in range(2)
        ]
    )
    yield readers
    for reader in readers:
        await reader._file_reader._close()


@pytest.mark.asyncio
async def test_composite_reader(readers):
    composite_reader = CompositeReader(readers=readers)
    for reader in composite_reader:
        assert isinstance(reader, COGReader)


@pytest.mark.asyncio
async def test_composite_reader_get_tile(readers):
    composite_reader = CompositeReader(readers=readers)
    tiles = await composite_reader.get_tile(x=0, y=0, z=0)
    assert (tiles[0] == tiles[1]).all()


@pytest.mark.asyncio
async def test_composite_reader_read(readers):
    composite_reader = CompositeReader(readers=readers)
    bounds = readers[0].native_bounds
    tiles = await composite_reader.read(bounds=bounds, shape=(256, 256))
    assert (tiles[0] == tiles[1]).all()
