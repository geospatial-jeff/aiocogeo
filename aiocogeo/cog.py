import asyncio
from dataclasses import dataclass, field
from functools import partial
import math
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin
import uuid

from aiocache import cached, Cache
import affine
import numpy as np
from skimage.transform import resize

from . import config
from .constants import PHOTOMETRIC
from .errors import InvalidTiffError, TileNotFoundError
from .filesystems import Filesystem
from .ifd import IFD


@dataclass
class COGReader:
    filepath: str
    ifds: Optional[List[IFD]] = field(default_factory=lambda: [])
    mask_ifds: Optional[List[IFD]] = field(default_factory=lambda: [])

    _version: Optional[int] = 42
    _big_tiff: Optional[bool] = False

    async def __aenter__(self):
        """Open the image and read the header"""
        async with Filesystem.create_from_filepath(self.filepath) as file_reader:
            self._file_reader = file_reader
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
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._file_reader._close()

    def __iter__(self):
        """Iterate through image IFDs"""
        for ifd in self.ifds:
            yield ifd

    @property
    def profile(self) -> Dict[str, Any]:
        """Return a rasterio-style image profile"""
        # TODO: Support nodata value
        return {
            "driver": "GTiff",
            "width": self.ifds[0].ImageWidth.value,
            "height": self.ifds[0].ImageHeight.value,
            "count": self.ifds[0].bands,
            "dtype": str(self.ifds[0].dtype),
            "transform": self.geotransform(),
            "blockxsize": self.ifds[0].TileWidth.value,
            "blockysize": self.ifds[0].TileHeight.value,
            "compress": self.ifds[0].compression,
            "interleave": self.ifds[0].interleave,
            "crs": f"EPSG:{self.epsg}",
            "tiled": True,
            "photometric": PHOTOMETRIC[self.ifds[0].PhotometricInterpretation.value],
        }

    @property
    def epsg(self) -> int:
        """Return the EPSG code representing the crs of the image"""
        ifd = self.ifds[0]
        for idx in range(0, len(ifd.GeoKeyDirectoryTag), 4):
            # 2048 is geographic crs
            # 3072 is projected crs
            if ifd.GeoKeyDirectoryTag[idx] in (2048, 3072):
                return ifd.GeoKeyDirectoryTag[idx + 3]

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Return the bounds of the image in native crs"""
        gt = self.geotransform()
        tlx = gt.c
        tly = gt.f
        brx = tlx + (gt.a * self.ifds[0].ImageWidth.value)
        bry = tly + (gt.e * self.ifds[0].ImageHeight.value)
        return (tlx, bry, brx, tly)

    @property
    def overviews(self) -> List[int]:
        """Return decimation factor for each overview (2**zoom)"""
        return [2 ** (ifd + 1) for ifd in range(len(self.ifds) - 1)]

    @property
    def is_masked(self) -> bool:
        """Check if the image has an internal mask"""
        return True if self.mask_ifds else False

    async def _read_header(self) -> None:
        """Internal method to read image header and parse into IFDs and Tags"""
        next_ifd_offset = 1
        while next_ifd_offset != 0:
            ifd = await IFD.read(self._file_reader)
            next_ifd_offset = ifd.next_ifd_offset
            self._file_reader.seek(next_ifd_offset)
            # Hack to correctly identify full resolution mask ifds in deflate compressed images
            # This assumes that the image ifd always comes before the mask ifd
            try:
                if ifd.is_mask or (
                    ifd.is_full_resolution
                    and ifd.ImageHeight.value == self.ifds[0].ImageHeight.value
                ):
                    self.mask_ifds.append(ifd)
                else:
                    self.ifds.append(ifd)
            except IndexError:
                self.ifds.append(ifd)

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
            bounds = self.bounds
            ifd = self.ifds[ovr_level]
            gt = affine.Affine.translation(bounds[0], bounds[3]) * affine.Affine.scale(
                (bounds[2] - bounds[0]) / ifd.ImageWidth.value,
                (bounds[1] - bounds[3]) / ifd.ImageHeight.value,
            )
        return gt

    def _get_overview_level(
        self, bounds: Tuple[float, float, float, float], width: int, height: int
    ) -> int:
        """
        Calculate appropriate overview level given request bounds and shape (width + height).  Based on rio-tiler:
        https://github.com/cogeotiff/rio-tiler/blob/v2/rio_tiler/utils.py#L79-L135
        """
        src_res = self.geotransform().a
        target_gt = affine.Affine.translation(
            bounds[0], bounds[3]
        ) * affine.Affine.scale(
            (bounds[2] - bounds[0]) / width, (bounds[1] - bounds[3]) / height
        )
        target_res = target_gt.a

        ovr_level = 0
        if target_res > src_res:
            # Decimated resolution at each overview
            overviews = [src_res * decim for decim in self.overviews]
            for ovr_level in range(ovr_level, len(overviews) - 1):
                ovr_res = src_res if ovr_level == 0 else overviews[ovr_level]
                if (ovr_res < target_res) and (overviews[ovr_level + 1] > target_res):
                    break
                if abs(ovr_res - target_res) < 1e-1:
                    break
            else:
                ovr_level = len(overviews) - 1

        return ovr_level

    @cached(
        cache=Cache.MEMORY,
        # Cache key comes from filepath and x/y/z coordinates of image tile
        key_builder=lambda fn, *args, **kwargs: f"{args[0].filepath}-{args[1]}-{args[2]}-{args[3]}",
    )
    async def get_tile(self, x: int, y: int, z: int) -> np.ndarray:
        """
        Request an internal image tile at the specified row (x), column (y), and overview (z).  Based on COGDumper:
        https://github.com/mapbox/COGDumper/blob/master/cogdumper/cog_tiles.py#L337-L365
        """
        futures = []
        if z > len(self.ifds):
            raise TileNotFoundError(f"Overview {z} does not exist.")
        ifd = self.ifds[z]
        idx = (y * ifd.tile_count[0]) + x
        if idx > len(ifd.TileOffsets):
            raise TileNotFoundError(f"Tile {x} {y} {z} does not exist")
        offset = ifd.TileOffsets[idx]
        byte_count = ifd.TileByteCounts[idx] - 1
        tile_task = asyncio.create_task(
            self._file_reader.range_request(offset, byte_count)
        )
        futures.append(tile_task)

        if self.is_masked:
            mask_ifd = self.mask_ifds[z]
            offset = mask_ifd.TileOffsets[idx]
            byte_count = mask_ifd.TileByteCounts[idx] - 1
            mask_task = asyncio.create_task(
                self._file_reader.range_request(offset, byte_count)
            )
            futures.append(mask_task)

        img_bytes = await asyncio.gather(*futures)

        decoded = ifd._decompress(img_bytes[0])
        if self.is_masked:
            mask = mask_ifd._decompress_mask(img_bytes[1])
            decoded = np.ma.masked_where(
                np.broadcast_to(mask, decoded.shape) == 0, decoded
            )

        return decoded

    @staticmethod
    def merge_range_requests(
        ranges: List[Tuple[int, int, int, int]]
    ) -> Tuple[Tuple[int, int], Tuple[Tuple[int, int, int, int]]]:
        """
        Helper function to merge consecutive range requests while keeping track of the initial ranges.  Returns an
        iterator which yields a tuple of tuples, where the first is a tuple with the start/end of a merged request and
        the second is another tuple of tuples where each tuple contains the original ranges and tile indices.

        For example, if we have two ranges A->B and B->C representing tiles in the top row of a COG, this
        method will yield:

            ((A, C), ((A, B, 0, 0), (B, C, 1, 0)))

        """
        # Create our start position (start, end)
        saved = [ranges[0][0], ranges[0][0] + ranges[0][1]]
        indices = []
        for offset, byte_count, idx, idy in sorted(ranges):
            # Get the next range
            start = offset
            end = offset + byte_count

            # Merge ranges
            if start <= saved[1]:
                saved[1] = max(saved[1], end)
                # Keep track of initial range
                indices.append((offset, byte_count, idx, idy))
            else:
                yield tuple(saved), tuple(indices)
                # Keep track of initial range
                indices = [(offset, byte_count, idx, idy)]
                saved[0] = start
                saved[1] = end
        yield tuple(saved), tuple(indices)

    def _calculate_image_tiles(
        self, bounds: Tuple[float, float, float, float], ovr_level: int
    ) -> Dict[str, Any]:
        """
        Internal method to calculate which images tiles need to be requested for a partial read.  Also returns metadata
        about those image tiles.
        """
        geotransform = self.geotransform(ovr_level)
        invgt = ~geotransform
        tile_width = self.ifds[ovr_level].TileWidth.value
        tile_height = self.ifds[ovr_level].TileHeight.value

        # Project request bounds to pixel coordinates relative to geotransform of the overview
        tlx, tly = invgt * (bounds[0], bounds[3])
        brx, bry = invgt * (bounds[2], bounds[1])

        # Calculate tiles
        xmin = math.floor((tlx + 1e-6) / tile_width)
        xmax = math.floor((brx + 1e-6) / tile_width)
        ymax = math.floor((bry + 1e-6) / tile_height)
        ymin = math.floor((tly + 1e-6) / tile_height)

        tile_bounds = (
            xmin * tile_width,
            ymin * tile_height,
            (xmax + 1) * tile_width,
            (ymax + 1) * tile_height,
        )

        # Create geotransform for the fused image
        _tlx, _tly = geotransform * (tile_bounds[0], tile_bounds[1])
        fused_gt = affine.Affine(
            geotransform.a, geotransform.b, _tlx, geotransform.d, geotransform.e, _tly
        )
        inv_fused_gt = ~fused_gt
        xorigin, yorigin = [round(v) for v in inv_fused_gt * (bounds[0], bounds[3])]
        return {
            "tlx": xorigin,
            "tly": yorigin,
            "width": round(brx - tlx),
            "height": round(bry - tly),
            "tile_ranges": (xmin, ymin, xmax, ymax),
        }

    @staticmethod
    def _stitch_image_tile(
        tile_arr: np.ndarray,
        fused_arr: np.ndarray,
        idx: int,
        idy: int,
        tile_width: int,
        tile_height: int,
    ) -> None:
        """
        Helper method to mosaic a tile into a larger numpy array based on its position with respect to the larger
        array
        """
        fused_arr[
            :,
            idy * tile_height : (idy + 1) * tile_height,
            idx * tile_width : (idx + 1) * tile_width,
        ] = tile_arr

    def _stitch_merged_image_tile(
        self,
        fut: asyncio.Future,
        fused: np.ndarray,
        offset: int,
        range_indices: Tuple[int, int, int, int],
        tile_width: int,
        tile_height: int,
        ovr_level: int,
    ) -> None:
        """Internal asyncio callback to mosaic merged ranged requests into a larger array"""
        img_bytes = fut.result()
        # Compression is applied to each block, so we need to "unmerge" the blocks after performing the merged range
        # request so we can decompress each block independently
        # TODO: Each tile is discrete so we should be able to do this in parallel for a minor speed up
        for (tile_offset, byte_count, idx, idy) in range_indices:
            # Find the start/end of the tile with respect to the merged request
            tile_start = tile_offset - offset
            tile_end = tile_start + byte_count
            # Extract the tile
            tile_bytes = img_bytes[tile_start:tile_end]
            # Decompress and mosaic
            decoded = self.ifds[ovr_level]._decompress(tile_bytes)
            self._stitch_image_tile(decoded, fused, idx, idy, tile_width, tile_height)

    def _stitch_image_tile_callback(self, fut: asyncio.Future, *args, **kwargs):
        """Internal asyncio callback to mosaic an image tile into a larger array."""
        tile_arr = fut.result()
        self._stitch_image_tile(tile_arr, *args, **kwargs)

    async def read(
        self, bounds: Tuple[float, float, float, float], shape: Tuple[int, int]
    ) -> Union[np.ndarray, np.ma.masked_array]:
        """
        Perform a partial read.  All pixels within the specified bounding box are read from the image and the array is
        resampled to match the desired shape.
        """
        # Determine which tiles intersect the request bounds
        ovr_level = self._get_overview_level(bounds, shape[1], shape[0])
        ifd = self.ifds[ovr_level]
        tile_height = ifd.TileHeight.value
        tile_width = ifd.TileWidth.value
        img_tiles = self._calculate_image_tiles(bounds, ovr_level)
        xmin, ymin, xmax, ymax = img_tiles["tile_ranges"]

        # Request those tiles
        tile_tasks = []
        fused = np.zeros(
            (
                ifd.bands,
                (ymax + 1 - ymin) * tile_height,
                (xmax + 1 - xmin) * tile_width,
            )
        ).astype(ifd.dtype)

        if config.HTTP_MERGE_CONSECUTIVE_RANGES == "TRUE":
            # Aggregate ranges
            ranges = []
            for idx, xtile in enumerate(range(xmin, xmax + 1)):
                for idy, ytile in enumerate(range(ymin, ymax + 1)):
                    offset = ifd.TileOffsets[(ytile * ifd.tile_count[0]) + xtile]
                    byte_count = ifd.TileByteCounts[(ytile * ifd.tile_count[0]) + xtile]
                    ranges.append((offset, byte_count, idx, idy))

            # Merge range requests, iterate through each merged request
            for (offsets, indices) in self.merge_range_requests(ranges):
                get_tile_task = asyncio.create_task(
                    self._file_reader.range_request(
                        offsets[0], offsets[1] - offsets[0] - 1
                    )
                )
                get_tile_task.add_done_callback(
                    partial(
                        self._stitch_merged_image_tile,
                        fused=fused,
                        offset=offsets[0],
                        range_indices=indices,
                        tile_width=tile_width,
                        tile_height=tile_height,
                        ovr_level=ovr_level,
                    )
                )
                tile_tasks.append(get_tile_task)
        else:
            for idx, xtile in enumerate(range(xmin, xmax + 1)):
                for idy, ytile in enumerate(range(ymin, ymax + 1)):
                    get_tile_task = asyncio.create_task(
                        self.get_tile(xtile, ytile, ovr_level)
                    )
                    get_tile_task.add_done_callback(
                        partial(
                            self._stitch_image_tile_callback,
                            fused_arr=fused,
                            idx=idx,
                            idy=idy,
                            tile_width=tile_width,
                            tile_height=tile_height,
                        )
                    )
                    tile_tasks.append(get_tile_task)

        # Request tiles
        await asyncio.gather(*tile_tasks)

        # Clip to request bounds
        clipped = fused[
            :,
            img_tiles["tly"] : img_tiles["tly"] + img_tiles["height"],
            img_tiles["tlx"] : img_tiles["tlx"] + img_tiles["width"],
        ]

        # Resample to match request size
        resized = resize(
            clipped,
            output_shape=(ifd.bands, shape[0], shape[1]),
            preserve_range=True,
            anti_aliasing=True,
        ).astype(ifd.dtype)

        return resized

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
            "supportedCRS": urljoin(
                f"http://www.opengis.net", f"/def/crs/EPSG/0/{self.epsg}"
            ),
            "tileMatrix": list(reversed(matrices)),
        }
        return tms
