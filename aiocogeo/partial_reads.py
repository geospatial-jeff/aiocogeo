"""COG mixins for partial reads"""
import asyncio
import abc
from dataclasses import dataclass
from functools import partial
import math
from typing import List, Tuple, Union

import affine
import numpy as np
from skimage.transform import resize

NpArrayType = Union[np.ndarray, np.ma.masked_array]


@dataclass
class ReadMetadata:
    # top left corner of the partial read
    tlx: float
    tly: float
    # width and height of the partial read (# of pixels)
    width: int
    height: int
    # width and height of each block (# of pixels)
    tile_width: int
    tile_height: int
    # range of internal x/y blocks which intersect the partial read
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    # expected number of bands
    bands: int
    # numpy data type
    dtype: np.dtype
    # overview level (where 0 is source)
    ovr_level: int


@dataclass
class PartialRead(abc.ABC):
    @property
    @abc.abstractmethod
    def is_masked(self) -> bool:
        """Check if the image has an internal mask"""
        ...

    @property
    @abc.abstractmethod
    def overviews(self) -> List[int]:
        """Return decimation factor for each overview (2**zoom)"""
        ...

    @abc.abstractmethod
    def geotransform(self, ovr_level: int = 0) -> affine.Affine:
        """Return the geotransform of the image at a specific overview level (defaults to native resolution)"""
        ...

    @abc.abstractmethod
    async def get_tile(self, x: int, y: int, z: int) -> NpArrayType:
        """Request an internal image tile at the specified row (x), column (y), and overview (z)"""
        ...

    @abc.abstractmethod
    async def read(self, bounds: Tuple[float, float, float, float], shape: Tuple[int, int]) -> Union[np.ndarray, np.ma.masked_array]:
        """Do a partial read"""
        ...

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

    def _calculate_image_tiles(
        self,
        bounds: Tuple[float, float, float, float],
        tile_width: int,
        tile_height: int,
        band_count: int,
        ovr_level: int,
        dtype: np.dtype,
    ) -> ReadMetadata:
        """
        Internal method to calculate which images tiles need to be requested for a partial read.  Also returns all of
        the required metadata about the image tiles to perform a partial read
        """
        geotransform = self.geotransform(ovr_level)
        invgt = ~geotransform

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

        return ReadMetadata(
            tlx=xorigin,
            tly=yorigin,
            width=round(brx - tlx),
            height=round(bry - tly),
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            tile_width=tile_width,
            tile_height=tile_height,
            bands=band_count,
            dtype=dtype,
            ovr_level=ovr_level,
        )

    def _init_array(self, img_tiles: ReadMetadata) -> NpArrayType:
        """
        Initialize an empty numpy array with the same shape of the partial read.  Individual blocks are mosaiced into
        this array as they are requested
        """
        fused = np.zeros(
            (
                img_tiles.bands,
                (img_tiles.ymax + 1 - img_tiles.ymin) * img_tiles.tile_height,
                (img_tiles.xmax + 1 - img_tiles.xmin) * img_tiles.tile_width,
            )
        ).astype(img_tiles.dtype)
        if self.is_masked:
            fused = np.ma.masked_array(fused)
        return fused

    @staticmethod
    def _stitch_image_tile(
        fut: asyncio.Future,
        fused_arr: NpArrayType,
        idx: int,
        idy: int,
        tile_width: int,
        tile_height: int,
    ) -> None:
        """Internal asyncio callback used to mosaic each image tile into a larger array (see ``_init_array``)"""
        img_arr = fut.result()
        fused_arr[
            :,
            idy * tile_height : (idy + 1) * tile_height,
            idx * tile_width : (idx + 1) * tile_width,
        ] = img_arr
        if np.ma.is_masked(img_arr):
            fused_arr.mask[
                :,
                idy * tile_height : (idy + 1) * tile_height,
                idx * tile_width : (idx + 1) * tile_width,
            ] = img_arr.mask

    async def _request_tiles(self, img_tiles: ReadMetadata) -> NpArrayType:
        """Concurrently request the image tiles and mosaic into a larger array"""
        img_arr = self._init_array(img_tiles)
        tile_tasks = []
        for idx, xtile in enumerate(range(img_tiles.xmin, img_tiles.xmax + 1)):
            for idy, ytile in enumerate(range(img_tiles.ymin, img_tiles.ymax + 1)):
                get_tile_task = asyncio.create_task(
                    self.get_tile(xtile, ytile, img_tiles.ovr_level)
                )
                get_tile_task.add_done_callback(
                    partial(
                        self._stitch_image_tile,
                        fused_arr=img_arr,
                        idx=idx,
                        idy=idy,
                        tile_width=img_tiles.tile_width,
                        tile_height=img_tiles.tile_height,
                    )
                )
                tile_tasks.append(get_tile_task)
        await asyncio.gather(*tile_tasks)
        return img_arr

    def _clip_array(self, arr: NpArrayType, img_tiles: ReadMetadata) -> NpArrayType:
        """Clip a numpy array to the extent of the parial read via slicing"""
        return arr[
            :,
            img_tiles.tly : img_tiles.tly + img_tiles.height,
            img_tiles.tlx : img_tiles.tlx + img_tiles.width,
        ]

    def _resample(
        self, clipped: NpArrayType, img_tiles: ReadMetadata, out_shape: Tuple[int, int]
    ) -> NpArrayType:
        """Resample a numpy array to the desired shape"""
        resized = resize(
            clipped,
            output_shape=(img_tiles.bands, out_shape[0], out_shape[1]),
            preserve_range=True,
            anti_aliasing=True,
        ).astype(img_tiles.dtype)
        if self.is_masked:
            resized_mask = resize(
                clipped.mask,
                output_shape=(img_tiles.bands, out_shape[0], out_shape[1]),
                preserve_range=True,
                anti_aliasing=True,
                order=0,
            )
            resized = np.ma.masked_array(resized, resized_mask)
        return resized

    def _postprocess(self, arr: NpArrayType, img_tiles: ReadMetadata, out_shape: Tuple[int, int]) -> NpArrayType:
        """Wrapper around ``_clip_array`` and ``_resample`` to postprocess the partial read"""
        return self._resample(
            self._clip_array(arr, img_tiles),
            img_tiles=img_tiles,
            out_shape=out_shape
        )