from dataclasses import dataclass
import math
from typing import Dict, List

from .counter import BytesCounter
from .tag import Tag

from cogdumper.cog_tiles import COGTiff

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
        if item in ("next_ifd_offset", "tag_count"):
            return getattr(self, item)
        else:
            return self.tags[item]

    @classmethod
    def read(cls, header: BytesCounter) -> "IFD":
        tag_count = header.read(2, cast_to_int=True)
        tiff_tags = {tag.name:tag for tag in list(filter(None, [Tag.read(header) for _ in range(tag_count)]))}
        next_ifd_offset = header.read(4, cast_to_int=True)
        return cls(next_ifd_offset, tag_count, tiff_tags)
