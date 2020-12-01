
from morecantile.models import TileMatrixSet
import pytest
import rasterio

from aiocogeo.ifd import IFD
from aiocogeo.tag import BaseTag
from aiocogeo.errors import InvalidTiffError
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
async def test_cog_palette(create_cog_reader):
    infile = "https://async-cog-reader-test-data.s3.amazonaws.com/cog_cmap.tif"
    async with create_cog_reader(infile) as cog:
        with rasterio.open(infile) as ds:
            cog_interp = cog.color_interp
            rio_interp = ds.colorinterp
            assert cog_interp[0].value == rio_interp[0].value
            assert cog.colormap == ds.colormap(1)


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
async def test_cog_has_alpha_band(create_cog_reader):
    async with create_cog_reader("https://async-cog-reader-test-data.s3.amazonaws.com/cog_alpha_band.tif") as cog:
        assert cog.has_alpha

    async with create_cog_reader(TEST_DATA[0]) as cog:
        assert not cog.has_alpha


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