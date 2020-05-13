from dataclasses import dataclass
import struct

import imagecodecs
import numpy as np

from .constants import COMPRESSIONS
from .counter import BytesReader
from .ifd import IFD


@dataclass
class Compressions:
    ifd: IFD
    reader: BytesReader
    tile: bytes

    def decompress(self):
        try:
            return getattr(self, f"_{self.ifd.compression}")()
        except AttributeError as e:
            raise NotImplementedError(
                f"{self.ifd.compression} is not currently supported"
            ) from e

    def _reshape(self, arr):
        return arr.reshape(
            self.ifd.TileHeight.value,
            self.ifd.TileWidth.value,
            self.ifd.bands
        )

    def _unpredict(self, arr):
        # Unpredict if there is horizontal differencing
        if self.ifd.Predictor.value == 2:
            imagecodecs.delta_decode(arr, out=arr, axis=-1)

    def _jpeg(self):
        jpeg_tables = self.ifd.JPEGTables
        jpeg_table_bytes = struct.pack(
            f"{self.reader._endian}{jpeg_tables.count}{jpeg_tables.tag_type.format}",
            *self.ifd.JPEGTables.value,
        )
        # # https://github.com/mapbox/COGDumper/tree/master/cogdumper
        if jpeg_table_bytes:
            if self.tile[0] == 0xFF and self.tile[1] == 0xD8:
                # insert tables, first removing the SOI and EOI
                self.tile = self.tile[0:2] + jpeg_table_bytes[2:-2] + self.tile[2:]
            else:
                raise Exception("Missing SOI marker for JPEG tile")
        decoded = imagecodecs.jpeg_decode(self.tile)
        return decoded

    def _lzw(self):
        decoded = self._reshape(np.frombuffer(imagecodecs.lzw_decode(self.tile), self.ifd.dtype))
        self._unpredict(decoded)
        return decoded

    def _webp(self):
        decoded = imagecodecs.webp_decode(self.tile)
        return decoded

    def _deflate(self):
        decoded = self._reshape(np.frombuffer(imagecodecs.zlib_decode(self.tile)))
        self._unpredict(decoded)
        return decoded