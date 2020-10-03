import mercantile
import numpy as np
import pytest
from PIL import Image
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from rio_tiler.io.cogeo import COGReader as cogeo_reader
from rio_tiler.mercator import get_zooms
from shapely.geometry import Polygon

from aiocogeo.tiler import COGTiler



@pytest.mark.asyncio
async def test_cog_tiler_tile(create_cog_reader):
    infile = "https://async-cog-reader-test-data.s3.amazonaws.com/webp_cog.tif"
    async with create_cog_reader(infile) as cog:
        tiler = COGTiler(cog)

        with rasterio.open(infile) as ds:
            _, zoom = get_zooms(ds)

        centroid = Polygon.from_bounds(
            *transform_bounds(cog.epsg, "EPSG:4326", *cog.bounds)
        ).centroid
        tile = mercantile.tile(centroid.x, centroid.y, zoom)

        arr, mask = await tiler.tile(
            tile.x, tile.y, tile.z, tilesize=256, resample_method=Image.BILINEAR
        )

        with cogeo_reader(infile) as ds:
            rio_tile_arr, rio_tile_mask = ds.tile(
                tile.x, tile.y, tile.z, tilesize=256, resampling_method="bilinear"
            )

        if cog.is_masked:
            # Make sure image data is the same
            assert pytest.approx(arr - rio_tile_arr, 1) == np.zeros(arr.shape)

            # Make sure mask data is the same
            rio_mask_counts = np.unique(rio_tile_mask, return_counts=True)
            tile_mask_counts = np.unique(mask, return_counts=True)
            assert len(rio_mask_counts[0]) == len(tile_mask_counts[0])
            assert (
                rio_mask_counts[1][0] * cog.profile["count"] == tile_mask_counts[1][0]
            )

        else:
            assert pytest.approx(arr - rio_tile_arr, 1) == np.zeros(arr.shape)


@pytest.mark.asyncio
async def test_cog_tiler_point(create_cog_reader):
    infile = "https://async-cog-reader-test-data.s3.amazonaws.com/webp_cog.tif"
    async with create_cog_reader(infile) as cog:
        tiler = COGTiler(cog)
        centroid = Polygon.from_bounds(
            *transform_bounds(cog.epsg, "EPSG:4326", *cog.bounds)
        ).centroid
        val = await tiler.point(
            lat=centroid.y, lon=centroid.x
        )
        assert list(val) == [50, 69, 74]


@pytest.mark.asyncio
async def test_cog_tiler_part(create_cog_reader):
    infile_nodata = (
        "https://async-cog-reader-test-data.s3.amazonaws.com/naip_image_nodata.tif"
    )
    async with create_cog_reader(infile_nodata) as cog:
        tiler = COGTiler(cog)
        tile = await tiler.part(
            bbox=(-10526706.9, 4445561.5, -10526084.1, 4446144.0),
            bbox_crs=CRS.from_epsg(cog.epsg),
        )
        assert tile.arr.shape == (3, 976, 1043)


@pytest.mark.asyncio
async def test_cog_tiler_part_dimensions(create_cog_reader):
    infile_nodata = (
        "https://async-cog-reader-test-data.s3.amazonaws.com/naip_image_nodata.tif"
    )
    async with create_cog_reader(infile_nodata) as cog:
        tiler = COGTiler(cog)
        tile = await tiler.part(
            bbox=(-10526706.9, 4445561.5, -10526084.1, 4446144.0),
            bbox_crs=CRS.from_epsg(cog.epsg),
            width=500,
            height=500,
        )
        assert tile.arr.shape == (3, 500, 500)


@pytest.mark.asyncio
async def test_cog_tiler_preview(create_cog_reader):
    infile = "https://async-cog-reader-test-data.s3.amazonaws.com/webp_cog.tif"
    async with create_cog_reader(infile) as cog:
        tiler = COGTiler(cog)
        tile = await tiler.preview()
        assert tile.arr.shape == (3, 1024, 864)


@pytest.mark.asyncio
async def test_cog_tiler_preview_max_size(create_cog_reader):
    infile = "https://async-cog-reader-test-data.s3.amazonaws.com/webp_cog.tif"
    async with create_cog_reader(infile) as cog:
        tiler = COGTiler(cog)
        tile = await tiler.preview(max_size=512)
        assert tile.arr.shape == (3, 512, 432)


@pytest.mark.asyncio
async def test_cog_tiler_preview_dimensions(create_cog_reader):
    infile = "https://async-cog-reader-test-data.s3.amazonaws.com/webp_cog.tif"
    async with create_cog_reader(infile) as cog:
        tiler = COGTiler(cog)
        tile = await tiler.preview(width=512, height=512)
        assert tile.arr.shape == (3, 512, 512)


@pytest.mark.asyncio
async def test_cog_tiler_info(create_cog_reader):
    infile = "https://async-cog-reader-test-data.s3.amazonaws.com/webp_cog.tif"
    async with create_cog_reader(infile) as cog:
        tiler = COGTiler(cog)
        info = await tiler.info()

        profile = cog.profile

        assert info.maxzoom > info.minzoom
        assert info.dtype == profile["dtype"]