from dataclasses import dataclass

@dataclass
class IFD:
    next_ifd_offset: int
    tag_count: int
    # tags: Dict[str, Tag]

    @classmethod
    def read(cls, header):
        tag_count = header.read(2, cast_to_int=True)
        header.incr(12 * tag_count)
        # TODO: Read tag data
        next_ifd_offset = header.read(4, cast_to_int=True)
        return cls(next_ifd_offset, tag_count)
