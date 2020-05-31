import math

from morecantile.models import TileMatrixSet
import mercantile
import numpy as np
import pytest
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from rio_tiler.io import cogeo
from rio_tiler import utils as rio_tiler_utils
from shapely.geometry import Polygon

from aiocogeo import config
from aiocogeo.ifd import IFD
from aiocogeo.tag import Tag
from aiocogeo.errors import InvalidTiffError

from .conftest import TEST_DATA


@pytest.mark.asyncio
@pytest.mark.parametrize("infile", TEST_DATA)
async def test_cog_metadata(infile, create_cog_reader):
    async with create_cog_reader(infile) as cog:
        with rasterio.open(infile) as ds:
            rio_profile = ds.profile
            cog_profile = cog.profile

            # Don't compare nodata, its not supported yet
            cog_profile.pop("nodata", None)
            rio_profile.pop("nodata", None)

            # Don't compare photometric, rasterio seems to not always report color interp
            cog_profile.pop("photometric", None)
            rio_profile.pop("photometric", None)

            assert rio_profile == cog_profile
            assert ds.overviews(1) == cog.overviews


@pytest.mark.asyncio
@pytest.mark.parametrize("infile", TEST_DATA)
async def test_cog_metadata_overviews(infile, create_cog_reader):
    async with create_cog_reader(infile) as cog:
        for idx, ifd in enumerate(cog.ifds):
            width = ifd.ImageWidth.value
            height = ifd.ImageHeight.value
            try:
                # Test decimation of 2
                next_ifd = cog.ifds[idx + 1]
                next_width = next_ifd.ImageWidth.value
                next_height = next_ifd.ImageHeight.value
                assert pytest.approx(width / next_width, 5) == 2.0
                assert pytest.approx(height / next_height, 5) == 2.0

                # Test number of tiles
                tile_count = ifd.tile_count[0] * ifd.tile_count[1]
                next_tile_count = next_ifd.tile_count[0] * next_ifd.tile_count[1]
                assert (
                    pytest.approx(
                        (
                            max(tile_count, next_tile_count)
                            / min(tile_count, next_tile_count)
                        ),
                        3,
                    )
                    == 4.0
                )
            except IndexError:
                pass


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
            rio_tile = src.read(
                window=window
            )
            if np.ma.is_masked(tile):
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
            else:
                # Make sure image data is the same
                assert pytest.approx(np.min(rio_tile), 2) == np.min(tile)
                assert pytest.approx(np.mean(rio_tile), 2) == np.mean(tile)
                assert pytest.approx(np.max(rio_tile), 2) == np.max(tile)




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
            gt.f
        )

        img_tile = cog._calculate_image_tiles(
            bounds,
            tile_width=ifd.TileWidth.value,
            tile_height=ifd.TileHeight.value,
            band_count=ifd.bands,
            ovr_level=ovr_level,
            dtype=ifd.dtype
        )
        assert img_tile.tlx == img_tile.tly == 0
        assert img_tile.width == cog.ifds[0].TileWidth.value
        assert img_tile.height == cog.ifds[0].TileHeight.value
        assert img_tile.xmin == 0
        assert img_tile.ymin == 0
        assert img_tile.xmax == 1
        assert img_tile.ymax == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("infile", TEST_DATA[:-2])
async def test_cog_read(infile, create_cog_reader):
    async with create_cog_reader(infile) as cog:
        zoom = math.floor(math.log2((2 * math.pi * 6378137 / 256) / cog.geotransform().a))
        centroid = Polygon.from_bounds(*transform_bounds(cog.epsg, "EPSG:4326", *cog.bounds)).centroid
        tile = mercantile.tile(centroid.x, centroid.y, zoom)

        tile_native_bounds = transform_bounds("EPSG:4326", cog.epsg, *mercantile.bounds(tile))

        arr = await cog.read(tile_native_bounds, (256, 256))
        rio_tile_arr, rio_tile_mask = cogeo.tile(infile, tile.x, tile.y, tile.z, tilesize=256, resampling_method="bilinear")

        if np.ma.is_masked(arr):
            tile_arr = np.ma.getdata(tile)
            tile_mask = np.ma.getmask(tile)

            # Make sure image data is the same
            assert pytest.approx(np.min(tile_arr), 2) == np.min(rio_tile_arr)
            assert pytest.approx(np.mean(tile_arr), 2) == np.mean(rio_tile_arr)
            assert pytest.approx(np.max(tile_arr), 2) == np.max(rio_tile_arr)

            # Make sure mask data is the same
            rio_mask_counts = np.unique(rio_tile_mask, return_counts=True)
            tile_mask_counts = np.unique(tile_mask, return_counts=True)
            assert rio_mask_counts[0].all() == tile_mask_counts[0].all()
            assert rio_mask_counts[1].all() == tile_mask_counts[1].all()
        else:
            assert pytest.approx(np.min(tile), 2) == np.min(rio_tile_arr)
            assert pytest.approx(np.mean(tile), 2) == np.mean(rio_tile_arr)
            assert pytest.approx(np.max(tile), 2) == np.max(rio_tile_arr)


@pytest.mark.asyncio
async def test_cog_read_internal_mask(create_cog_reader):
    async with create_cog_reader("https://async-cog-reader-test-data.s3.amazonaws.com/naip_image_masked.tif") as cog:
        tile = await cog.read(bounds=(-10526706.9, 4445561.5, -10526084.1, 4446144.0), shape=(512,512))
        assert np.ma.is_masked(tile)

        # Confirm proportion of masked pixels
        valid_data = tile[tile.mask == False]
        frequencies = np.asarray(np.unique(valid_data, return_counts=True)).T
        assert pytest.approx(frequencies[0][1] / np.prod(tile.shape), abs=0.004) == 0


@pytest.mark.asyncio
async def test_cog_read_merge_range_requests(create_cog_reader, monkeypatch):
    infile = "https://async-cog-reader-test-data.s3.amazonaws.com/lzw_cog.tif"
    bounds = (368461,3770591,368796,3770921)
    shape = (512, 512)

    # Do a request without merged range requests
    async with create_cog_reader(infile) as cog:
        tile_data = await cog.read(bounds=bounds, shape=shape)
        request_count = cog._file_reader._total_requests
        bytes_requested = cog._file_reader._total_bytes_requested

    # Do a request with merged range requests
    monkeypatch.setattr(config, "HTTP_MERGE_CONSECUTIVE_RANGES", "TRUE")
    async with create_cog_reader(infile) as cog:
        tile_data_merged = await cog.read(bounds=bounds, shape=shape)
        merged_request_count = cog._file_reader._total_requests
        merged_bytes_requested = cog._file_reader._total_bytes_requested

    # Confirm we got the same tile with fewer requests
    assert merged_request_count < request_count
    assert bytes_requested == merged_bytes_requested
    assert tile_data.all() == tile_data_merged.all()
    assert tile_data.shape == tile_data_merged.shape


@pytest.mark.asyncio
async def test_cog_read_merge_range_requests_with_internal_nodata_mask(create_cog_reader, monkeypatch):
    infile = "https://async-cog-reader-test-data.s3.amazonaws.com/naip_image_masked.tif"
    bounds = (-10526706.9, 4445561.5, -10526084.1, 4446144.0)
    shape = (512, 512)

    # Do a request without merged range requests
    async with create_cog_reader(infile) as cog:
        tile_data = await cog.read(bounds=bounds, shape=shape)
        # assert np.ma.is_masked(tile_data)
        request_count = cog._file_reader._total_requests
        bytes_requested = cog._file_reader._total_bytes_requested

    # Do a request with merged range requests
    monkeypatch.setattr(config, "HTTP_MERGE_CONSECUTIVE_RANGES", "TRUE")
    async with create_cog_reader(infile) as cog:
        tile_data_merged = await cog.read(bounds=bounds, shape=shape)
        # assert np.ma.is_masked(tile_data_merged)
        merged_request_count = cog._file_reader._total_requests
        merged_bytes_requested = cog._file_reader._total_bytes_requested

    # Confirm we got the same tile with fewer requests
    assert merged_request_count < request_count
    assert bytes_requested == merged_bytes_requested
    assert tile_data.all() == tile_data_merged.all()
    assert tile_data.shape == tile_data_merged.shape


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "width,height", [(500, 500), (1000, 1000), (5000, 5000), (10000, 10000)]
)
async def test_cog_get_overview_level(create_cog_reader, width, height):
    async with create_cog_reader(TEST_DATA[0]) as cog:
        ovr = cog._get_overview_level(cog.bounds, width, height)

        with rasterio.open(TEST_DATA[0]) as src:
            expected_ovr = rio_tiler_utils.get_overview_level(
                src, src.bounds, height, width
            )
            # Our index for source data is 0 while rio tiler uses -1
            expected_ovr = 0 if expected_ovr == -1 else expected_ovr
            assert ovr == expected_ovr

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "infile", TEST_DATA
)
async def test_cog_tile_matrix_set(infile, create_cog_reader):
    async with create_cog_reader(infile) as cog:
        tile_matrix_set = cog.create_tile_matrix_set()
        TileMatrixSet(**tile_matrix_set)

@pytest.mark.asyncio
@pytest.mark.parametrize("infile", [TEST_DATA[0]])
async def test_cog_metadata_iter(infile, create_cog_reader):
    async with create_cog_reader(infile) as cog:
        for ifd in cog:
            assert isinstance(ifd, IFD)
            for tag in ifd:
                assert isinstance(tag, Tag)

@pytest.mark.asyncio
async def test_block_cache_enabled(create_cog_reader):
    # Cache is enabled by default
    infile = "https://async-cog-reader-test-data.s3.amazonaws.com/lzw_cog.tif"
    async with create_cog_reader(infile) as cog:
        await cog.get_tile(0,0,0)
        request_count = cog._file_reader._total_requests

        await cog.get_tile(0,0,0)
        assert cog._file_reader._total_requests == request_count


@pytest.mark.asyncio
async def test_block_cache_disabled(create_cog_reader, monkeypatch):
    monkeypatch.setattr(config, "ENABLE_BLOCK_CACHE", False)

    infile = "https://async-cog-reader-test-data.s3.amazonaws.com/lzw_cog.tif"
    async with create_cog_reader(infile) as cog:
        await cog.get_tile(0,0,0)
        request_count = cog._file_reader._total_requests

        await cog.get_tile(0,0,0)
        assert cog._file_reader._total_requests == request_count + 1



@pytest.mark.asyncio
async def test_cog_not_a_tiff(create_cog_reader):
    infile = "https://async-cog-reader-test-data.s3.amazonaws.com/not_a_tiff.png"
    with pytest.raises(InvalidTiffError):
        async with create_cog_reader(infile) as cog:
            ...
