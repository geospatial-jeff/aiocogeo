import random

import aiohttp
import mercantile
import numpy as np
import pytest
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from rio_tiler.io.cogeo import COGReader as cogeo_reader
from rio_tiler import utils as rio_tiler_utils
from rio_tiler.mercator import get_zooms
from shapely.geometry import Polygon

from aiocogeo import config
from aiocogeo.errors import TileNotFoundError

from .conftest import TEST_DATA


@pytest.mark.asyncio
@pytest.mark.parametrize("infile", TEST_DATA)
async def test_cog_read_internal_tile(infile, create_cog_reader):
    async with create_cog_reader(infile) as cog:
        # Read top left tile at native resolution
        tile = await cog.get_tile(0, 0, 0)
        ifd = cog.ifds[0]

        # Make sure tile is the right size
        assert tile.shape == (
            ifd.SamplesPerPixel.value,
            ifd.TileHeight.value,
            ifd.TileWidth.value,
        )

        with rasterio.open(infile) as src:
            window = Window(0, 0, ifd.TileWidth.value, ifd.TileHeight.value)
            rio_tile = src.read(window=window)
            # Internal mask
            if np.ma.is_masked(tile) and cog.is_masked:
                assert cog.is_masked
                tile_arr = np.ma.getdata(tile)
                tile_mask = np.ma.getmask(tile)
                rio_mask = src.read_masks(1, window=window)

                # Make sure image data is the same
                assert pytest.approx(np.min(rio_tile), 2) == np.min(tile_arr)
                assert pytest.approx(np.mean(rio_tile), 2) == np.mean(tile_arr)
                assert pytest.approx(np.max(rio_tile), 2) == np.max(tile_arr)

                # Make sure mask data is the same
                rio_mask_counts = np.unique(rio_mask, return_counts=True)
                tile_mask_counts = np.unique(tile_mask, return_counts=True)
                assert rio_mask_counts[0].all() == tile_mask_counts[0].all()
                assert rio_mask_counts[1].all() == tile_mask_counts[1].all()
            # Nodata
            elif ifd.nodata is not None:
                # Mask rio array to match aiocogeo output
                rio_tile = np.ma.masked_where(
                    rio_tile == src.profile["nodata"], rio_tile
                )
                assert pytest.approx(np.min(rio_tile), 2) == np.min(tile)
                assert pytest.approx(np.mean(rio_tile), 2) == np.mean(tile)
                assert pytest.approx(np.max(rio_tile), 2) == np.max(tile)
            else:
                # Make sure image data is the same
                assert pytest.approx(np.min(rio_tile), 2) == np.min(tile)
                assert pytest.approx(np.mean(rio_tile), 2) == np.mean(tile)
                assert pytest.approx(np.max(rio_tile), 2) == np.max(tile)


@pytest.mark.asyncio
async def test_cog_read_internal_tile_nodata(create_cog_reader):
    infile_nodata = (
        "https://async-cog-reader-test-data.s3.amazonaws.com/naip_image_nodata.tif"
    )
    async with create_cog_reader(infile_nodata) as cog:
        nodata_tile = await cog.get_tile(0, 0, 0)
        # Confirm output array is masked using nodata value
        assert np.ma.is_masked(nodata_tile)
        assert not np.any(nodata_tile == cog.profile["nodata"])

    # Confirm nodata mask is comparable to same image with internal mask
    infile_internal_mask = (
        "https://async-cog-reader-test-data.s3.amazonaws.com/naip_image_masked.tif"
    )
    async with create_cog_reader(infile_internal_mask) as cog:
        mask_tile = await cog.get_tile(0, 0, 0)
        # Confirm same number of masked values
        assert nodata_tile.count() == mask_tile.count()


@pytest.mark.asyncio
@pytest.mark.parametrize("infile", TEST_DATA[:-1])
async def test_cog_calculate_image_tiles(infile, create_cog_reader):
    async with create_cog_reader(infile) as cog:
        ovr_level = 0
        gt = cog.geotransform(ovr_level)
        ifd = cog.ifds[ovr_level]

        # Find bounds of top left tile at native res
        bounds = (
            gt.c,
            gt.f + ifd.TileHeight.value * gt.e,
            gt.c + ifd.TileWidth.value * gt.a,
            gt.f,
        )

        img_tile = cog._calculate_image_tiles(
            bounds,
            tile_width=ifd.TileWidth.value,
            tile_height=ifd.TileHeight.value,
            band_count=ifd.bands,
            ovr_level=ovr_level,
            dtype=ifd.dtype,
        )
        assert img_tile.tlx == img_tile.tly == 0
        assert img_tile.width == cog.ifds[0].TileWidth.value
        assert img_tile.height == cog.ifds[0].TileHeight.value
        assert img_tile.xmin == 0
        assert img_tile.ymin == 0
        assert img_tile.xmax == 1
        assert img_tile.ymax == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("infile", TEST_DATA[:-3])
async def test_cog_read(infile, create_cog_reader):
    async with create_cog_reader(infile) as cog:
        with rasterio.open(infile) as ds:
            _, zoom = get_zooms(ds)
        centroid = Polygon.from_bounds(
            *transform_bounds(cog.epsg, "EPSG:4326", *cog.native_bounds)
        ).centroid
        tile = mercantile.tile(centroid.x, centroid.y, zoom)

        tile_native_bounds = transform_bounds(
            "EPSG:4326", cog.epsg, *mercantile.bounds(tile)
        )

        arr = await cog.read(tile_native_bounds, (256, 256))

        with cogeo_reader(infile) as ds:
            rio_tile_arr, rio_tile_mask = ds.tile(tile.x, tile.y, tile.z, tilesize=256, resampling_method="bilinear")

        if cog.is_masked:
            tile_arr = np.ma.getdata(arr)
            tile_mask = np.ma.getmask(arr)

            # Make sure image data is the same
            assert pytest.approx(tile_arr - rio_tile_arr, 1) == np.zeros(tile_arr.shape)

            # Make sure mask data is the same
            rio_mask_counts = np.unique(rio_tile_mask, return_counts=True)
            tile_mask_counts = np.unique(tile_mask, return_counts=True)
            assert len(rio_mask_counts[0]) == len(tile_mask_counts[0])
            assert (
                rio_mask_counts[1][0] * cog.profile["count"] == tile_mask_counts[1][0]
            )

        else:
            assert pytest.approx(arr - rio_tile_arr, 1) == np.zeros(arr.shape)


@pytest.mark.asyncio
async def test_cog_read_single_band(create_cog_reader):
    infile = "https://async-cog-reader-test-data.s3.amazonaws.com/int16_deflate.tif"
    async with create_cog_reader(infile) as cog:
        assert cog.profile["count"] == 1
        with rasterio.open(infile) as ds:
            _, zoom = get_zooms(ds)
        centroid = Polygon.from_bounds(
            *transform_bounds(cog.epsg, "EPSG:4326", *cog.native_bounds)
        ).centroid
        tile = mercantile.tile(centroid.x, centroid.y, zoom)
        tile_native_bounds = transform_bounds(
            "EPSG:4326", cog.epsg, *mercantile.bounds(tile)
        )
        arr = await cog.read(tile_native_bounds, (256, 256))
        assert arr.shape == (256, 256)
        assert arr.dtype == cog.profile["dtype"]


@pytest.mark.asyncio
async def test_cog_read_internal_mask(create_cog_reader):
    async with create_cog_reader(
        "https://async-cog-reader-test-data.s3.amazonaws.com/naip_image_masked.tif"
    ) as cog:
        tile = await cog.read(
            bounds=(-10526706.9, 4445561.5, -10526084.1, 4446144.0), shape=(512, 512)
        )
        assert np.ma.is_masked(tile)

        # Confirm proportion of masked pixels
        valid_data = tile[tile.mask == False]
        frequencies = np.asarray(np.unique(valid_data, return_counts=True)).T
        assert pytest.approx(frequencies[0][1] / np.prod(tile.shape), abs=0.002) == 0


@pytest.mark.asyncio
async def test_cog_read_nodata_value(create_cog_reader):
    infile_nodata = (
        "https://async-cog-reader-test-data.s3.amazonaws.com/naip_image_nodata.tif"
    )
    async with create_cog_reader(infile_nodata) as cog:
        nodata_tile = await cog.read(
            bounds=(-10526706.9, 4445561.5, -10526084.1, 4446144.0), shape=(512, 512)
        )
        assert np.ma.is_masked(nodata_tile)

    # Confirm nodata mask is comparable to same image with internal mask
    infile_internal_mask = (
        "https://async-cog-reader-test-data.s3.amazonaws.com/naip_image_masked.tif"
    )
    async with create_cog_reader(infile_internal_mask) as cog:
        mask_tile = await cog.read(
            bounds=(-10526706.9, 4445561.5, -10526084.1, 4446144.0), shape=(512, 512)
        )
        assert np.ma.is_masked(mask_tile)

        # Nodata values wont be exactly the same because of difference in compression but they should be similar
        # proportional to number of masked values
        proportion_masked = abs(nodata_tile.count() - mask_tile.count()) / max(
            nodata_tile.count(), mask_tile.count()
        )
        assert pytest.approx(proportion_masked, abs=0.003) == 0


@pytest.mark.asyncio
async def test_cog_read_merge_range_requests(create_cog_reader, monkeypatch):
    infile = "https://async-cog-reader-test-data.s3.amazonaws.com/lzw_cog.tif"
    bounds = (368461, 3770591, 368796, 3770921)
    shape = (512, 512)

    # Do a request without merged range requests
    async with create_cog_reader(infile) as cog:
        tile_data = await cog.read(bounds=bounds, shape=shape)
        request_count = cog.requests["count"]
        bytes_requested = cog.requests["byte_count"]

    # Do a request with merged range requests
    monkeypatch.setattr(config, "HTTP_MERGE_CONSECUTIVE_RANGES", True)
    async with create_cog_reader(infile) as cog:
        tile_data_merged = await cog.read(bounds=bounds, shape=shape)
        merged_request_count = cog.requests["count"]
        merged_bytes_requested = cog.requests["byte_count"]

    # Confirm we got the same tile with fewer requests
    assert merged_request_count < request_count
    assert bytes_requested == merged_bytes_requested
    assert tile_data.all() == tile_data_merged.all()
    assert tile_data.shape == tile_data_merged.shape


@pytest.mark.asyncio
async def test_cog_read_merge_range_requests_with_internal_nodata_mask(
    create_cog_reader, monkeypatch
):
    infile = "https://async-cog-reader-test-data.s3.amazonaws.com/naip_image_masked.tif"
    bounds = (-10526706.9, 4445561.5, -10526084.1, 4446144.0)
    shape = (512, 512)

    # Do a request without merged range requests
    async with create_cog_reader(infile) as cog:
        tile_data = await cog.read(bounds=bounds, shape=shape)
        # assert np.ma.is_masked(tile_data)
        request_count = cog.requests["count"]
        bytes_requested = cog.requests["byte_count"]

    # Do a request with merged range requests
    monkeypatch.setattr(config, "HTTP_MERGE_CONSECUTIVE_RANGES", True)
    async with create_cog_reader(infile) as cog:
        tile_data_merged = await cog.read(bounds=bounds, shape=shape)
        # assert np.ma.is_masked(tile_data_merged)
        merged_request_count = cog.requests["count"]
        merged_bytes_requested = cog.requests["byte_count"]

    # Confirm we got the same tile with fewer requests
    assert merged_request_count < request_count
    assert bytes_requested == merged_bytes_requested
    assert tile_data.all() == tile_data_merged.all()
    assert tile_data.shape == tile_data_merged.shape


@pytest.mark.asyncio
async def test_boundless_read(create_cog_reader, monkeypatch):
    infile = (
        "http://async-cog-reader-test-data.s3.amazonaws.com/webp_web_optimized_cog.tif"
    )
    tile = mercantile.Tile(x=701, y=1634, z=12)
    bounds = mercantile.xy_bounds(tile)

    # Confirm an exception is raised if boundless reads are disabled
    monkeypatch.setattr(config, "BOUNDLESS_READ", False)

    async with create_cog_reader(infile) as cog:
        with pytest.raises(TileNotFoundError):
            tile = await cog.read(bounds=bounds, shape=(256, 256))

    monkeypatch.setattr(config, "BOUNDLESS_READ", True)
    async with create_cog_reader(infile) as cog:
        await cog.read(bounds=bounds, shape=(256, 256))


@pytest.mark.asyncio
async def test_boundless_read_fill_value(create_cog_reader, monkeypatch):
    infile = (
        "http://async-cog-reader-test-data.s3.amazonaws.com/webp_web_optimized_cog.tif"
    )
    tile = mercantile.Tile(x=701, y=1634, z=12)
    bounds = mercantile.xy_bounds(tile)

    async with create_cog_reader(infile) as cog:
        # Count number of pixels with a value of 1
        tile = await cog.read(bounds=bounds, shape=(256, 256))
        counts = dict(zip(*np.unique(tile, return_counts=True)))
        assert counts[1] == 713

        # Set fill value of 1
        monkeypatch.setattr(config, "BOUNDLESS_READ_FILL_VALUE", 1)

        # Count number of pixels with a value of 1
        tile = await cog.read(bounds=bounds, shape=(256, 256))
        counts = dict(zip(*np.unique(tile, return_counts=True)))
        assert counts[1] == 154889


@pytest.mark.asyncio
@pytest.mark.parametrize("infile", TEST_DATA)
async def test_boundless_get_tile(create_cog_reader, infile, monkeypatch):
    async with create_cog_reader(infile) as cog:
        fill_value = random.randint(0, 100)
        monkeypatch.setattr(config, "BOUNDLESS_READ_FILL_VALUE", fill_value)

        # Test reading tiles outside of IFD when boundless reads is enabled
        tile = await cog.get_tile(x=-1, y=-1, z=0)
        counts = dict(zip(*np.unique(tile, return_counts=True)))
        assert counts[fill_value] == tile.shape[0] * tile.shape[1] * tile.shape[2]


@pytest.mark.asyncio
@pytest.mark.parametrize("infile", TEST_DATA)
async def test_read_not_in_bounds(create_cog_reader, infile):
    tile = mercantile.Tile(x=0, y=0, z=25)
    bounds = mercantile.xy_bounds(tile)

    async with create_cog_reader(infile) as cog:
        if cog.epsg != 3857:
            bounds = transform_bounds("EPSG:3857", f"EPSG:{cog.epsg}", *bounds)
        with pytest.raises(TileNotFoundError):
            await cog.read(bounds=bounds, shape=(256, 256))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "width,height,expected_ovr", [(500, 500, 4), (1000, 1000, 3), (5000, 5000, 1), (10000, 10000, 0)]
)
async def test_cog_get_overview_level(create_cog_reader, width, height, expected_ovr):
    # available resolutions: [ 0.6,  1.2,  2.4,  4.8,  9.6, 19.2, 38.4]
    # target resolutions: [12.336 ,  6.168 ,  1.2336,  0.6168]
    # `expected_ovr` is index of available resolution that most closely matches the target resolution
    async with create_cog_reader(TEST_DATA[0]) as cog:
        ovr = cog._get_overview_level(cog.native_bounds, width, height)
        assert ovr == expected_ovr

@pytest.mark.asyncio
async def test_inject_session(create_cog_reader):
    async with aiohttp.ClientSession() as session:
        async with create_cog_reader(
            "https://async-cog-reader-test-data.s3.amazonaws.com/webp_cog.tif",
            kwargs={"session": session},
        ):
            pass
        # Confirm session is still open
        assert not session.closed
        assert session._trace_configs
