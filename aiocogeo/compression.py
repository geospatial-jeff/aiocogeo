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
        ...

    @property
    @abc.abstractmethod
    def compression(self) -> str:
        ...

    @property
    @abc.abstractmethod
    def dtype(self) -> np.dtype:
        ...

    def decompress(self, tile):
        try:
            return getattr(self, f"_{self.compression}")(tile)
        except AttributeError as e:
            raise NotImplementedError(
                f"{self.compression} is not currently supported"
            ) from e

    def decompress_mask(self, tile):
        decoded = np.frombuffer(imagecodecs.zlib_decode(tile), np.dtype('uint8'))
        mask = np.unpackbits(decoded).reshape(self.TileHeight.value, self.TileWidth.value) * 255
        return mask

    def _reshape(self, arr):
        return arr.reshape(
            self.TileHeight.value,
            self.TileWidth.value,
            self.bands,
        )

    def _unpredict(self, arr):
        # Unpredict if there is horizontal differencing
        if self.Predictor.value == 2:
            imagecodecs.delta_decode(arr, out=arr, axis=-1)

    def _jpeg(self, tile):
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

    def _lzw(self, tile):
        decoded = self._reshape(np.frombuffer(imagecodecs.lzw_decode(tile), self.dtype))
        self._unpredict(decoded)
        return np.rollaxis(decoded, 2, 0)

    def _webp(self, tile):
        decoded = np.rollaxis(imagecodecs.webp_decode(tile), 2, 0)
        return decoded

    def _deflate(self, tile):
        decoded = self._reshape(np.frombuffer(imagecodecs.zlib_decode(tile), self.dtype))
        self._unpredict(decoded)
        return np.rollaxis(decoded, 2, 0)