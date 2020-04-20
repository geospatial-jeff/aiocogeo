from dataclasses import dataclass
import math
from typing import Dict

from .counter import BytesReader
from .tag import Tag


@dataclass
class IFD:
    next_ifd_offset: int
    tag_count: int
    tags: Dict[str, Tag] # Store tags as dict where key is tag name for easier access

    @property
    def tile_count(self):
        return (
            math.ceil(self.ImageWidth.value / float(self.TileWidth.value)),
            math.ceil(self.ImageHeight.value / float(self.TileHeight.value))
        )

    def __getattr__(self, item):
        return self.tags[item]

    def __iter__(self):
        for (_, tag) in self.tags.items():
            yield tag

    @classmethod
    async def read(cls, reader: BytesReader) -> "IFD":
        ifd_start = reader.tell()
        tag_count = await reader.read(2, cast_to_int=True)
        tiff_tags = {tag.name:tag for tag in list(filter(None, [(await Tag.read(reader)) for _ in range(tag_count)]))}
        reader.seek(ifd_start + (12 * tag_count) + 2)
        next_ifd_offset = await reader.read(4, cast_to_int=True)
        return cls(next_ifd_offset, tag_count, tiff_tags)
