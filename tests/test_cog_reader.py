import random

import aiohttp
from morecantile.models import TileMatrixSet
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

from aiocogeo import config, COGReader
from aiocogeo.ifd import IFD
from aiocogeo.tag import BaseTag
from aiocogeo.errors import InvalidTiffError, TileNotFoundError
from aiocogeo.constants import MaskFlags

from .conftest import TEST_DATA


@pytest.mark.asyncio
@pytest.mark.parametrize("infile", TEST_DATA)
async def test_cog_metadata(infile, create_cog_reader):
    async with create_cog_reader(infile) as cog:
        with rasterio.open(infile) as ds:
            rio_profile = ds.profile
            cog_profile = cog.profile

            # Don't compare photometric, rasterio seems to not always report color interp
            cog_profile.pop("photometric", None)
            rio_profile.pop("photometric", None)

            assert [member.value for member in ds.colorinterp] == [member.value for member in cog.color_interp]
            assert rio_profile == cog_profile
            assert ds.overviews(1) == cog.overviews

            rio_tags = ds.tags()
            gdal_metadata = cog.gdal_metadata

            for (k,v) in rio_tags.items():
                if k in ("TIFFTAG_XRESOLUTION", "TIFFTAG_YRESOLUTION", "TIFFTAG_RESOLUTIONUNIT"):
                    continue
                assert str(gdal_metadata[k]) == v


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
            *transform_bounds(cog.epsg, "EPSG:4326", *cog.bounds)
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
            *transform_bounds(cog.epsg, "EPSG:4326", *cog.bounds)
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
        assert counts[1] == 351

        # Set fill value of 1
        monkeypatch.setattr(config, "BOUNDLESS_READ_FILL_VALUE", 1)

        # Count number of pixels with a value of 1
        tile = await cog.read(bounds=bounds, shape=(256, 256))
        counts = dict(zip(*np.unique(tile, return_counts=True)))
        assert counts[1] == 167583


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
async def test_point(create_cog_reader, infile):
    async with create_cog_reader(infile) as cog:
        bounds = cog.bounds
        pt = await cog.point(x=bounds[0], y=bounds[3])
        tile = await cog.get_tile(0, 0, 0)
        if cog.is_masked or cog.nodata is not None:
            assert np.equal(pt.data, tile.data[:, 0, 0]).all()
        else:
            assert np.equal(pt, tile[:, 0, 0]).all()


@pytest.mark.asyncio
@pytest.mark.parametrize("infile", TEST_DATA[:-1])
async def test_preview(create_cog_reader, infile):
    async with create_cog_reader(infile) as cog:
        profile = cog.profile
        preview = await cog.preview(max_size=1024)

        src_aspect_ratio = profile["height"] / profile["width"]
        dst_aspect_ratio = preview.shape[-2] / preview.shape[-1]
        assert pytest.approx(src_aspect_ratio, 0.001) == dst_aspect_ratio
        assert preview.shape[-2] <= 1024
        assert preview.shape[-1] <= 1024


@pytest.mark.asyncio
async def test_preview_width_height(create_cog_reader):
    async with create_cog_reader(
        "https://async-cog-reader-test-data.s3.amazonaws.com/webp_cog.tif"
    ) as cog:
        preview = await cog.preview(width=512, height=512)
        assert preview.shape == (3, 512, 512)


@pytest.mark.asyncio
async def test_cog_has_alpha_band(create_cog_reader):
    async with create_cog_reader("https://async-cog-reader-test-data.s3.amazonaws.com/cog_alpha_band.tif") as cog:
        assert cog.has_alpha

    async with create_cog_reader(TEST_DATA[0]) as cog:
        assert not cog.has_alpha


@pytest.mark.asyncio
@pytest.mark.parametrize("infile,expected", zip(TEST_DATA, [
    [[MaskFlags.all_valid], [MaskFlags.all_valid], [MaskFlags.all_valid]],
    [[MaskFlags.all_valid], [MaskFlags.all_valid], [MaskFlags.all_valid]],
    [[MaskFlags.all_valid], [MaskFlags.all_valid], [MaskFlags.all_valid]],
    [[MaskFlags.all_valid], [MaskFlags.all_valid], [MaskFlags.all_valid]],
    [[MaskFlags.all_valid], [MaskFlags.all_valid], [MaskFlags.all_valid]],
    [[MaskFlags.nodata], [MaskFlags.nodata], [MaskFlags.nodata]],
    [[MaskFlags.nodata], [MaskFlags.nodata], [MaskFlags.nodata]],
    [[MaskFlags.per_dataset], [MaskFlags.per_dataset], [MaskFlags.per_dataset]],
    [[MaskFlags.nodata], [MaskFlags.nodata], [MaskFlags.nodata]],
    [
        [MaskFlags.per_dataset, MaskFlags.alpha],
        [MaskFlags.per_dataset, MaskFlags.alpha],
        [MaskFlags.per_dataset, MaskFlags.alpha],
        [MaskFlags.all_valid]
    ],
    [[MaskFlags.per_dataset]],
    [[MaskFlags.all_valid], [MaskFlags.all_valid], [MaskFlags.all_valid]],

]))
async def test_cog_mask_flags(create_cog_reader, infile, expected):
    async with create_cog_reader(infile) as cog:
        mask_flags = cog.mask_flags
    assert expected == mask_flags

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
async def test_cog_palette(create_cog_reader):
    infile = "https://async-cog-reader-test-data.s3.amazonaws.com/cog_cmap.tif"
    async with create_cog_reader(infile) as cog:
        with rasterio.open(infile) as ds:
            cog_interp = cog.color_interp
            rio_interp = ds.colorinterp
            assert cog_interp[0].value == rio_interp[0].value
            assert cog.colormap == ds.colormap(1)


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
@pytest.mark.parametrize("infile", TEST_DATA)
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
                assert isinstance(tag, BaseTag)


@pytest.mark.asyncio
async def test_block_cache_enabled(create_cog_reader, monkeypatch):
    # Cache is disabled for tests
    monkeypatch.setattr(config, "ENABLE_CACHE", True)
    infile = "https://async-cog-reader-test-data.s3.amazonaws.com/lzw_cog.tif"
    async with create_cog_reader(infile) as cog:
        await cog.get_tile(0, 0, 0)

    async with create_cog_reader(infile) as cog:
        await cog.get_tile(0, 0, 0)
        # Confirm all requests are cached
        assert cog.requests["count"] == 0


@pytest.mark.asyncio
async def test_block_cache_disabled(create_cog_reader):
    infile = "https://async-cog-reader-test-data.s3.amazonaws.com/lzw_cog.tif"
    async with create_cog_reader(infile) as cog:
        await cog.get_tile(0, 0, 0)
        request_count = cog.requests["count"]

        await cog.get_tile(0, 0, 0)
        assert cog.requests["count"] == request_count + 1


@pytest.mark.asyncio
@pytest.mark.parametrize("infile", TEST_DATA)
async def test_cog_request_metadata(create_cog_reader, infile):
    async with create_cog_reader(infile) as cog:
        request_metadata = cog.requests

    assert len(request_metadata["ranges"]) == request_metadata["count"]
    assert (
        sum([end - start + 1 for (start, end) in request_metadata["ranges"]])
        == request_metadata["byte_count"]
    )


@pytest.mark.asyncio
async def test_cog_not_a_tiff(create_cog_reader):
    infile = "https://async-cog-reader-test-data.s3.amazonaws.com/not_a_tiff.png"
    with pytest.raises(InvalidTiffError):
        async with create_cog_reader(infile) as cog:
            ...


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "infile",
    [
        "/local/file/does/not/exist",
        "https://cogsarecool.com/cog.tif",  # Invalid host
        "https://async-cog-reader-test-data.s3.amazonaws.com/file-does-not-exist.tif",  # valid host invalid path
        "s3://nobucket/badkey.tif",
    ],
)
async def test_file_not_found(create_cog_reader, infile):
    with pytest.raises(FileNotFoundError):
        async with create_cog_reader(infile) as cog:
            ...


@pytest.mark.asyncio
async def test_inject_session():
    async with aiohttp.ClientSession() as session:
        async with COGReader(
            "https://async-cog-reader-test-data.s3.amazonaws.com/webp_cog.tif",
            kwargs={"session": session},
        ):
            pass
        # Confirm session is still open
        assert not session.closed
        assert session._trace_configs
