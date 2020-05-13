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
        compression = COMPRESSIONS[self.ifd.Compression.value]
        try:
            return getattr(self, f"_{compression}")()
        except AttributeError as e:
            raise NotImplementedError(
                f"{compression} is not currently supported"
            ) from e

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
        decoded = imagecodecs.lzw_decode(self.tile)
        decoded = np.frombuffer(decoded, self.ifd.dtype).reshape(
            self.ifd.TileHeight.value,
            self.ifd.TileWidth.value,
            self.ifd.bands,
        )
        # Unpredict if there is horizontal differencing
        if self.ifd.Predictor.value == 2:
            imagecodecs.delta_decode(decoded, out=decoded, axis=-1)
        return decoded

    def _webp(self):
        decoded = imagecodecs.webp_decode(self.tile)
        return decoded
