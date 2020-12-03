import abc
import asyncio
from dataclasses import dataclass, field
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin
import uuid

import affine
from PIL import Image
import numpy as np

from . import config
from .constants import ColorInterp, MaskFlags, PHOTOMETRIC
from .errors import InvalidTiffError, TileNotFoundError
from .filesystems import Filesystem
from .ifd import IFD, ImageIFD, MaskIFD
from .partial_reads import PartialReadInterface

logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


@dataclass
class ReaderMixin(abc.ABC):

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        ...

    @abc.abstractmethod
    async def get_tile(self, x: int, y: int, z: int) -> Union[np.ndarray, List[np.ndarray]]:
        ...

    @abc.abstractmethod
    async def read(
        self,
        bounds: Tuple[float, float, float, float],
        shape: Tuple[int, int],
        resample_method: int = Image.NEAREST,
    ) -> Union[Union[np.ndarray, np.ma.masked_array], List[Union[np.ndarray, np.ma.masked_array]]]:
        ...


@dataclass
class COGReader(ReaderMixin, PartialReadInterface):
    filepath: str
    ifds: Optional[List[ImageIFD]] = field(default_factory=lambda: [])
    mask_ifds: Optional[List[MaskIFD]] = field(default_factory=lambda: [])

    _version: Optional[int] = 42
    _big_tiff: Optional[bool] = False

    kwargs: Optional[Dict] = field(default_factory=dict)

    async def __aenter__(self):
        """Open the image and read the header"""
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._file_reader._close()

    def __iter__(self):
        """Iterate through image IFDs"""
        for ifd in self.ifds:
            yield ifd

    async def open(self):
        await self._open()

    async def _open(self):
        """internal method to open the cog by reading the file header"""
        async with Filesystem.create_from_filepath(self.filepath, **self.kwargs) as file_reader:
            self._file_reader = file_reader
            # Do the first request
            self._file_reader.data += await self._file_reader.range_request(0, config.INGESTED_BYTES_AT_OPEN, is_header=True)
            if (await file_reader.read(2)) == b"MM":
                file_reader._endian = ">"
            version = await file_reader.read(2, cast_to_int=True)
            if version == 42:
                first_ifd = await file_reader.read(4, cast_to_int=True)
                file_reader.seek(first_ifd)
                await self._read_header()
            elif version == 43:
                raise NotImplementedError("BigTiff is not yet supported")
            else:
                raise InvalidTiffError("Not a valid TIFF")

    @property
    def profile(self) -> Dict[str, Any]:
        """Return a rasterio-style image profile"""
        ifd = self.ifds[0]
        return {
            "driver": "GTiff",
            "width": ifd.ImageWidth.value,
            "height": ifd.ImageHeight.value,
            "count": ifd.bands,
            "dtype": str(ifd.dtype),
            "transform": self.geotransform(),
            "blockxsize": ifd.TileWidth.value,
            "blockysize": ifd.TileHeight.value,
            "compress": ifd.compression,
            "interleave": ifd.interleave,
            "crs": f"EPSG:{self.epsg}",
            "nodata": ifd.nodata,
            "tiled": True,
            "photometric": self.photometric,
        }

    @property
    def epsg(self) -> int:
        """Return the EPSG code representing the crs of the image"""
        return self.ifds[0].geo_keys.epsg

    @property
    def native_bounds(self) -> Tuple[float, float, float, float]:
        """Return the bounds of the image in native crs"""
        gt = self.geotransform()
        tlx = gt.c
        tly = gt.f
        brx = tlx + (gt.a * self.ifds[0].ImageWidth.value)
        bry = tly + (gt.e * self.ifds[0].ImageHeight.value)
        return (tlx, bry, brx, tly)

    @property
    def indexes(self) -> Tuple[int]:
        """Return rasterio style band indexes"""
        return tuple([r + 1 for r in range(self.ifds[0].bands)])

    @property
    def overviews(self) -> List[int]:
        """Return decimation factor for each overview (2**zoom)"""
        return [2 ** (ifd + 1) for ifd in range(len(self.ifds) - 1)]

    @property
    def requests(self) -> Dict[str, Union[int, List[Tuple[int]]]]:
        """Return statistics about http requests made during context lifecycle"""
        return {
            'count': self._file_reader._total_requests,
            'byte_count': self._file_reader._total_bytes_requested,
            'ranges': self._file_reader._requested_ranges,
            'header_size': self._file_reader._header_size
        }

    @property
    def is_masked(self) -> bool:
        """Check if the image has an internal mask"""
        return True if self.mask_ifds else False

    @property
    def mask_flags(self):
        """
        https://gdal.org/doxygen/classGDALRasterBand.html#a181a931c6ecbdd8c84c5478e4aa48aaf
        https://trac.osgeo.org/gdal/wiki/rfc15_nodatabitmask
        """
        bands = self.ifds[0].bands
        flags = set()
        if self.nodata is not None:
            flags.add(MaskFlags.nodata)
        if self.has_alpha:
            flags.add(MaskFlags.per_dataset)
            flags.add(MaskFlags.alpha)
        if self.mask_ifds:
            flags.add(MaskFlags.per_dataset)
        if not any([self.nodata is not None, self.has_alpha, self.mask_ifds]):
            flags.add(MaskFlags.all_valid)

        flags = list(flags)
        if self.has_alpha:
            extra_samples = self.ifds[0].ExtraSamples.count or 0
            band_flags = [flags for _ in range(bands - extra_samples)]
            for _ in range(extra_samples):
                band_flags.append([MaskFlags.all_valid])
            return band_flags
        return [flags for _ in range(bands)]

    @property
    def photometric(self):
        return PHOTOMETRIC[self.ifds[0].PhotometricInterpretation.value]

    @property
    def colormap(self) -> Optional[Dict[int, Tuple[int, int, int]]]:
        """https://www.awaresystems.be/imaging/tiff/tifftags/colormap.html"""
        if self.ifds[0].ColorMap:
            colormap = {}
            count = 2 ** self.ifds[0].BitsPerSample.value

            nodata_val = None
            if self.has_alpha or self.nodata is not None:
                nodata_val = 0 if self.has_alpha else self.nodata

            transform = lambda val: int((val / 65535) * 255)
            for idx in range(count):
                color = [transform(self.ifds[0].ColorMap.value[idx + i * count]) for i in range(3)]
                if nodata_val is not None:
                    color.append(0 if idx == nodata_val else 255)
                colormap[idx] = tuple(color)
            return colormap
        return None

    @property
    def color_interp(self):
        """
        https://gdal.org/user/raster_data_model.html#raster-band
        https://trac.osgeo.org/gdal/ticket/4547#comment:1
        """
        photometric = self.photometric
        if photometric == "rgb":
            interp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue]
            if self.has_alpha:
                interp.append(ColorInterp.alpha)
        elif photometric == "minisblack" or photometric == "miniswhite":
            interp = [ColorInterp.gray]
        elif photometric == "palette":
            interp = [ColorInterp.palette]
        elif photometric == "cmyk":
            interp = [ColorInterp.cyan, ColorInterp.magenta, ColorInterp.yellow, ColorInterp.black]
        elif photometric == "ycbcr":
            interp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue]
        elif photometric == "cielab" or photometric == "icclab" or photometric == "itulab":
            interp = [ColorInterp.lightness, ColorInterp.lightness, ColorInterp.lightness]
        else:
            interp = [ColorInterp.undefined for _ in range(self.profile['count'])]
        return interp

    @property
    def has_alpha(self) -> bool:
        """Check if the image has an alpha band"""
        if self.mask_ifds:
            for ifd in self.mask_ifds:
                if ifd.is_alpha:
                    return True
        return False

    @property
    def nodata(self) -> Optional[int]:
        return self.ifds[0].nodata

    @property
    def gdal_metadata(self) -> Dict:
        return self.ifds[0].gdal_metadata

    async def _read_header(self) -> None:
        """Internal method to read image header and parse into IFDs and Tags"""
        next_ifd_offset = 1
        while next_ifd_offset != 0:
            ifd = await IFD.read(self._file_reader)
            logger.debug(f" Opened {ifd.ImageHeight.value}x{ifd.ImageWidth.value} overview")
            next_ifd_offset = ifd.next_ifd_offset
            self._file_reader.seek(next_ifd_offset)

            if isinstance(ifd, MaskIFD):
                self.mask_ifds.append(ifd)
            else:
                self.ifds.append(ifd)

        # TODO: Explicitely associate the image with it's mask
        # Label alpha bands
        if self.mask_ifds:
            for (image, mask) in zip(self.ifds, self.mask_ifds):
                if image.has_extra_samples:
                    mask.is_alpha = True

    def geotransform(self, ovr_level: int = 0) -> affine.Affine:
        """Return the geotransform of the image at a specific overview level (defaults to native resolution)"""
        # Calculate overview for source image
        gt = affine.Affine(
            self.ifds[0].ModelPixelScaleTag[0],
            0.0,
            self.ifds[0].ModelTiepointTag[3],
            0.0,
            -self.ifds[0].ModelPixelScaleTag[1],
            self.ifds[0].ModelTiepointTag[4],
        )
        # Decimate the geotransform if an overview is requested
        if ovr_level > 0:
            bounds = self.native_bounds
            ifd = self.ifds[ovr_level]
            gt = affine.Affine.translation(bounds[0], bounds[3]) * affine.Affine.scale(
                (bounds[2] - bounds[0]) / ifd.ImageWidth.value,
                (bounds[1] - bounds[3]) / ifd.ImageHeight.value,
            )
        return gt


    async def get_tile(self, x: int, y: int, z: int) -> np.ndarray:

        """
        Request an internal image tile at the specified row (x), column (y), and overview (z).  Based on COGDumper:
        https://github.com/mapbox/COGDumper/blob/master/cogdumper/cog_tiles.py#L337-L365
        """
        futures = []
        if z > len(self.ifds):
            raise TileNotFoundError(f"Overview {z} does not exist.")
        ifd = self.ifds[z]
        xmax, ymax = ifd.tile_count

        # Return an empty array if tile is outside bounds of image
        if x < 0 or y < 0 or x >= xmax or y >= ymax:
            if not config.BOUNDLESS_READ:
                raise TileNotFoundError(f"Internal tile {z}/{x}/{y} does not exist")
            tile = np.full(
                (ifd.bands, ifd.TileHeight.value, ifd.TileWidth.value),
                fill_value=config.BOUNDLESS_READ_FILL_VALUE
            )
            return tile

        # Request the tile
        futures.append(
            asyncio.create_task(ifd._get_tile(x, y))
        )

        # Request the mask
        if self.is_masked:
            mask_ifd = self.mask_ifds[z]
            futures.append(
                asyncio.create_task(mask_ifd._get_tile(x, y))
            )

        tile = await asyncio.gather(*futures)

        # Prioritize internal mask over nodata
        if self.is_masked:
            # Apply mask
            tile[1] = np.invert(np.broadcast_to(tile[1], tile[0].shape))
            return np.ma.masked_array(*tile)
        # Explicitly check for None because nodata is often 0
        if ifd.nodata is not None:
            return np.ma.masked_where(tile[0] == ifd.nodata, tile[0])
        return tile[0]

    async def read(
        self,
        bounds: Tuple[float, float, float, float],
        shape: Tuple[int, int],
        resample_method: int = Image.NEAREST,
    ) -> Union[np.ndarray, np.ma.masked_array]:
        """
        Perform a partial read.  All pixels within the specified bounding box are read from the image and the array is
        resampled to match the desired shape.
        """
        # Determine which tiles intersect the request bounds
        ovr_level = self._get_overview_level(bounds, shape[1], shape[0])
        ifd = self.ifds[ovr_level]
        img_tiles = self._calculate_image_tiles(
            bounds,
            tile_width=ifd.TileWidth.value,
            tile_height=ifd.TileHeight.value,
            band_count=ifd.bands,
            ovr_level=ovr_level,
            dtype=ifd.dtype
        )

        if not self._intersect_bounds(bounds, self.native_bounds):
            raise TileNotFoundError("Partial read is outside bounds of the image")

        # Request those tiles
        if config.HTTP_MERGE_CONSECUTIVE_RANGES:
            img_arr = await self._request_merged_tiles(img_tiles)
        else:
            img_arr = await self._request_tiles(img_tiles)

        # Postprocess the array (clip to bounds and resize to requested shape)
        postprocessed = self._postprocess(
            arr=img_arr,
            img_tiles=img_tiles,
            out_shape=shape,
            resample_method=resample_method
        )

        return postprocessed


    def create_tile_matrix_set(self, identifier: str = None) -> Dict[str, Any]:
        """Create an OGC TileMatrixSet where each TileMatrix corresponds to an overview"""
        matrices = []
        for idx, ifd in enumerate(self.ifds):
            gt = self.geotransform(idx)
            matrix = {
                "identifier": str(len(self.ifds) - idx - 1),
                "topLeftCorner": [gt.c, gt.f],
                "tileWidth": ifd.TileWidth.value,
                "tileHeight": ifd.TileHeight.value,
                "matrixWidth": ifd.tile_count[0],
                "matrixHeight": ifd.tile_count[1],
                "scaleDenominator": gt.a / 0.28e-3,
            }
            matrices.append(matrix)

        tms = {
            "title": f"Tile matrix for {self.filepath}",
            "identifier": identifier or str(uuid.uuid4()),
            "supportedCRS": urljoin(f"http://www.opengis.net", f"/def/crs/EPSG/0/{self.epsg}"),
            "tileMatrix": list(reversed(matrices))
        }
        return tms


FilterType = Callable[[COGReader], Any]
MapType = Callable[[COGReader], Any]
ReduceType = Callable[[List[Union[np.ndarray, np.ma.masked_array]]], Any]

@dataclass
class CompositeReader(ReaderMixin):
    readers: Optional[List[COGReader]] = field(default_factory=list)
    filter: FilterType = lambda a: a
    default_reducer: ReduceType = lambda r: r

    def __iter__(self):
        return iter(self.readers)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def map(self, func: MapType) -> List[Any]:
        futs = [func(reader) for reader in filter(self.filter, self.readers)]
        return await asyncio.gather(*futs)

    async def get_tile(self, x: int, y: int, z: int, reducer: Optional[ReduceType] = None) -> List[np.ndarray]:
        """Fetch a tile from all readers"""
        tiles = await self.map(
            func=lambda r: r.get_tile(x, y, z),
        )
        reducer = reducer or self.default_reducer
        return reducer(tiles)

    async def read(
        self,
        bounds: Tuple[float, float, float, float],
        shape: Tuple[int, int],
        resample_method: int = Image.NEAREST,
        reducer: Optional[ReduceType] = None
    ):
        reads = await self.map(
            func=lambda r: r.read(bounds, shape, resample_method)
        )
        reducer = reducer or self.default_reducer
        return reducer(reads)