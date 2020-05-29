import abc
from dataclasses import dataclass
import struct
from typing import Optional

import imagecodecs
import numpy as np

from .filesystems import Filesystem
from .tag import Tag


@dataclass
class Compression(metaclass=abc.ABCMeta):
    _file_reader: Filesystem
    TileHeight: Tag
    TileWidth: Tag
    Predictor: Optional[Tag]
    JPEGTables: Optional[Tag]


    @property
    @abc.abstractmethod
    def bands(self) -> int:
        """Return the number of image bands"""
        ...

    @property
    @abc.abstractmethod
    def compression(self) -> str:
        """Return the compression of the IFD"""
        ...

    @property
    @abc.abstractmethod
    def dtype(self) -> np.dtype:
        """Return the data type of the IFD"""
        ...

    def _decompress(self, tile: bytes) -> np.ndarray:
        """Internal method to convert image bytes to numpy array with decompression applied"""
        try:
            return getattr(self, f"_{self.compression}")(tile)
        except AttributeError as e:
            raise NotImplementedError(
                f"{self.compression} is not currently supported"
            ) from e

    def _decompress_mask(self, tile: bytes) -> np.ndarray:
        """Internal method to decompress a binary mask and rescale to uint8"""
        decoded = np.frombuffer(imagecodecs.zlib_decode(tile), np.dtype('uint8'))
        mask = np.unpackbits(decoded).reshape(self.TileHeight.value, self.TileWidth.value) * 255
        return mask

    def _reshape(self, arr: np.ndarray) -> np.ndarray:
        """Internal method to reshape an array to the size expected by the IFD"""
        return arr.reshape(
            self.TileHeight.value,
            self.TileWidth.value,
            self.bands,
        )

    def _unpredict(self, arr: np.ndarray) -> None:
        """Internal method to unpredict if there is horizontal differencing"""
        if self.Predictor.value == 2:
            imagecodecs.delta_decode(arr, out=arr, axis=-1)

    def _jpeg(self, tile: bytes) -> np.ndarray:
        """Internal method to decompress JPEG image bytes and convert to numpy array"""
        jpeg_tables = self.JPEGTables
        jpeg_table_bytes = struct.pack(
            f"{self._file_reader._endian}{jpeg_tables.count}{jpeg_tables.tag_type.format}",
            *self.JPEGTables.value,
        )
        # # https://github.com/mapbox/COGDumper/tree/master/cogdumper
        if jpeg_table_bytes:
            if tile[0] == 0xFF and tile[1] == 0xD8:
                # insert tables, first removing the SOI and EOI
                tile = tile[0:2] + jpeg_table_bytes[2:-2] + tile[2:]
            else:
                raise Exception("Missing SOI marker for JPEG tile")
        decoded = imagecodecs.jpeg_decode(tile)
        return np.rollaxis(decoded, 2, 0)

    def _lzw(self, tile: bytes) -> np.ndarray:
        """Internal method to decompress LZW image bytes and convert to numpy array"""
        decoded = self._reshape(np.frombuffer(imagecodecs.lzw_decode(tile), self.dtype))
        self._unpredict(decoded)
        return np.rollaxis(decoded, 2, 0)

    def _webp(self, tile: bytes) -> np.ndarray:
        """Internal method to decompress WEBP image bytes and convert to numpy array"""
        decoded = np.rollaxis(imagecodecs.webp_decode(tile), 2, 0)
        return decoded

    def _deflate(self, tile: bytes) -> np.ndarray:
        """Internal method to decompress DEFLATE image bytes and convert to numpy array"""
        decoded = self._reshape(np.frombuffer(imagecodecs.zlib_decode(tile), self.dtype))
        self._unpredict(decoded)
        return np.rollaxis(decoded, 2, 0)