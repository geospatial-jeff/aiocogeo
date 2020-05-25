import abc
from dataclasses import dataclass
from urllib.parse import urlsplit

import aioboto3
import aiofiles
import aiohttp

from .constants import HEADER_OFFSET

@dataclass
class Filesystem(abc.ABC):
    filepath: str

    def __post_init__(self):
        self.data: bytes = b""
        self._offset: int = 0
        self._endian: str = "<"
        self._total_bytes_requested: int = 0
        self._total_requests: int = 0

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        ...

    @classmethod
    def create_from_filepath(cls, filepath: str) -> "Filesystem":
        splits = urlsplit(filepath)
        if splits.scheme in {"http", "https"}:
            return HttpFilesystem(filepath)
        elif splits.scheme == "s3":
            return S3Filesystem(filepath)
        elif (not splits.scheme and not splits.netloc):
            return LocalFilesystem(filepath)
        raise NotImplemented("Unsupported file system")

    @abc.abstractmethod
    async def range_request(self, start: int, offset: int) -> bytes:
        ...

    @abc.abstractmethod
    async def close(self) -> None:
        ...

    async def read(self, offset: int, cast_to_int: bool = False):
        if self._offset + offset > len(self.data):
            self.data += await self.range_request(len(self.data), HEADER_OFFSET)
        data = self.data[self._offset : self._offset + offset]
        self.incr(offset)
        order = "little" if self._endian == "<" else "big"
        return int.from_bytes(data, order) if cast_to_int else data

    def incr(self, offset: int) -> None:
        self._offset += offset

    def seek(self, offset: int) -> None:
        self._offset = offset

    def tell(self) -> int:
        return self._offset


@dataclass
class HttpFilesystem(Filesystem):

    async def range_request(self, start, offset):
        range_header = {"Range": f"bytes={start}-{start + offset}"}
        async with self.session.get(self.filepath, headers=range_header) as cog:
            data = await cog.content.read()
            self._total_bytes_requested += int(cog.headers["Content-Length"])
            self._total_requests += 1
        return data

    async def close(self):
        await self.session.close()

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self


@dataclass
class LocalFilesystem(Filesystem):

    async def range_request(self, start, offset):
        await self.file.seek(start)
        self._total_bytes_requested += (offset - start)
        self._total_requests += 1
        return await self.file.read(offset+1)

    async def close(self):
        await self.file.close()

    async def __aenter__(self):
        self.file = await aiofiles.open(self.filepath, 'rb')
        return self


@dataclass
class S3Filesystem(Filesystem):

    async def range_request(self, start: int, offset: int) -> bytes:
        req = await self.object.get(Range=f'bytes={start}-{start+offset}')
        self._total_bytes_requested += int(req['ResponseMetadata']['HTTPHeaders']['content-length'])
        self._total_requests += 1
        data = await req['Body'].read()
        return data

    async def close(self) -> None:
        await self.resource.__aexit__('', '', '')

    async def __aenter__(self):
        splits = urlsplit(self.filepath)
        self.resource = await aioboto3.resource('s3').__aenter__()
        self.object = await self.resource.Object(splits.netloc, splits.path[1:])
        return self