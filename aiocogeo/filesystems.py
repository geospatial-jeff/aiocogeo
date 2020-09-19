import abc
import asyncio
from dataclasses import dataclass, field
import json
import logging
import time
from typing import Any, Callable, Dict, Union
from urllib.parse import urlsplit

from aiocache import cached, Cache
import aiofiles
import aiohttp
import botocore.exceptions

from . import config

# https://github.com/developmentseed/rio-viz/blob/master/rio_viz/app.py#L33-L38
try:
    import aioboto3
    has_s3 = True
    if config.VERBOSE_LOGS:
        # Default to boto3 debug logs if verbose logging is enabled
        s3_log_level = config.LOG_LEVEL
    else:
        s3_log_level = logging.ERROR

    logging.getLogger("aiobotocore").setLevel(s3_log_level)
    logging.getLogger("botocore").setLevel(s3_log_level)
    logging.getLogger("aioboto3").setLevel(s3_log_level)
except ModuleNotFoundError:
    has_s3 = False


logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


def config_cache(fn: Callable) -> Callable:
    """
    Inject cache config params (https://aiocache.readthedocs.io/en/latest/decorators.html#aiocache.cached)
    """
    def wrap_function(*args, **kwargs):
        kwargs['cache_read'] = kwargs['cache_write'] = config.ENABLE_CACHE
        return fn(*args, **kwargs)
    return wrap_function


@dataclass
class Filesystem(abc.ABC):
    filepath: str
    kwargs: field(default_factory=dict)

    def __post_init__(self):
        self.data: bytes = b""
        self._offset: int = 0
        self._endian: str = "<"
        self._total_bytes_requested: int = 0
        self._total_requests: int = 0
        self._header_size: int = config.INGESTED_BYTES_AT_OPEN
        self._requested_ranges = []

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        ...

    @classmethod
    def create_from_filepath(cls, filepath: str, **kwargs) -> "Filesystem":
        """Instantiate the appropriate filesystem based on filepath scheme"""
        splits = urlsplit(filepath)
        if splits.scheme in {"http", "https"}:
            return HttpFilesystem(filepath, kwargs=kwargs)
        elif splits.scheme == "s3":
            if not has_s3:
                raise NotImplementedError("Package must be built with [s3] extra to read from S3")
            return S3Filesystem(filepath, kwargs=kwargs)
        elif (not splits.scheme and not splits.netloc):
            return LocalFilesystem(filepath, kwargs=kwargs)
        raise NotImplemented("Unsupported file system")

    @config_cache
    @cached(
        cache=Cache.MEMORY,
        key_builder=lambda fn,*args,**kwargs: f"{args[0].filepath}-{args[1]}-{args[2]}"
    )
    async def range_request(self, start: int, offset: int) -> bytes:
        """
        Perform and cache a range request.
        """
        return await self._range_request(start, offset)

    @abc.abstractmethod
    async def request_json(self):
        ...

    @abc.abstractmethod
    async def _range_request(self, start: int, offset: int) -> bytes:
        """Perform a range request"""
        ...

    @abc.abstractmethod
    async def _close(self) -> None:
        """
        Close any resources created in ``__aexit__``, allows extending ``Filesystem`` context managers past their scope
        """
        ...


    async def read(self, offset: int, cast_to_int: bool = False):
        """
        Read from the current offset (self._offset) to the specified offset and optionall cast the result to int
        """
        if self._offset + offset > len(self.data):
            self.data += await self.range_request(len(self.data), config.INGESTED_BYTES_AT_OPEN)
        data = self.data[self._offset : self._offset + offset]
        self.incr(offset)
        order = "little" if self._endian == "<" else "big"
        return int.from_bytes(data, order) if cast_to_int else data

    def incr(self, offset: int) -> None:
        """Increment offset"""
        self._offset += offset

    def seek(self, offset: int) -> None:
        """Seek to the specified offset (setter for ``self._offset``)"""
        self._offset = offset

    def tell(self) -> int:
        """Return the current offset (getter for ``self._offset``)"""
        return self._offset


@dataclass
class HttpFilesystem(Filesystem):

    async def get_session(self) -> aiohttp.ClientSession:
        trace_config = aiohttp.TraceConfig()
        trace_config.on_request_start.append((self._on_request_start))
        trace_config.on_request_end.append(self._on_request_end)
        if "session" in self.kwargs:
            session = self.kwargs['session']
            if not session._trace_configs:
                trace_config.freeze()
                session._trace_configs = [trace_config]
            return session
        return aiohttp.ClientSession(trace_configs=[trace_config])

    async def _range_request(self, start: int, offset: int) -> bytes:
        range_header = {"Range": f"bytes={start}-{start + offset}"}
        try:
            async with self.session.get(self.filepath, headers=range_header) as resp:
                resp.raise_for_status()
                data = await resp.content.read()
        except (aiohttp.ClientError, aiohttp.ClientResponseError) as e:
            await self._close()
            raise FileNotFoundError(f"File not found: {self.filepath}") from e
        return data

    async def request_json(self) -> Dict:
        try:
            async with self.session.get(self.filepath) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except (aiohttp.ClientError, aiohttp.ClientResponseError) as e:
            await self._close()
            raise FileNotFoundError(f"File not found: {self.filepath}") from e
        return data


    async def _close(self) -> None:
        if 'session' not in self.kwargs:
            await self.session.close()

    async def __aenter__(self):
        self.session = await self.get_session()
        return self


    async def _on_request_start(self, session, trace_config_ctx, params):
        trace_config_ctx.start = asyncio.get_event_loop().time()
        if config.VERBOSE_LOGS:
            debug_statement = (
                f"\n > {params.method} {params.url.path} HTTP/{session.version.major}.{session.version.minor}"
                f"\n   Host: {params.url.host}"
                f"\n   Range: {params.headers['Range']}"
            )
        else:
            debug_statement = f" STARTING REQUEST: {params.method} {params.url}"
        logger.debug(debug_statement)

    async def _on_request_end(self, session, trace_config_ctx, params):
        if params.response.status < 400:
            elapsed = round(asyncio.get_event_loop().time() - trace_config_ctx.start, 3)
            content_range = params.response.headers.get('Content-Range')
            self._total_bytes_requested += int(params.response.headers["Content-Length"])
            self._total_requests += 1
            if content_range:
                self._requested_ranges.append(tuple([int(v) for v in content_range.split(' ')[-1].split('/')[0].split('-')]))
            if config.VERBOSE_LOGS:
                debug_statement = [f"\n < HTTP/{session.version.major}.{session.version.minor}"]
                debug_statement += [f"\n < {k}: {v}" for (k, v) in params.response.headers.items()]
                debug_statement.append(f"\n < Duration: {elapsed}")
            else:
                debug_statement = f" FINISHED REQUEST in {elapsed} seconds: <STATUS {params.response.status}> ({content_range})"
            logger.debug("".join(debug_statement))


@dataclass
class LocalFilesystem(Filesystem):

    async def _range_request(self, start: int, offset: int) -> bytes:
        begin = time.time()
        await self.file.seek(start)
        data = await self.file.read(offset+1)
        elapsed = time.time() - begin
        self._total_bytes_requested += (offset - start + 1)
        self._total_requests += 1
        self._requested_ranges.append((start, start+offset))
        logger.debug(f" FINISHED REQUEST in {elapsed} seconds: <STATUS 206> ({start}-{start+offset})")
        return data

    async def request_json(self):
        return json.load(self.file)

    async def _close(self) -> None:
        await self.file.close()

    async def __aenter__(self):
        self.file = await aiofiles.open(self.filepath, 'rb')
        return self


@dataclass
class S3Filesystem(Filesystem):

    async def _range_request(self, start: int, offset: int) -> bytes:
        kwargs = {}
        if config.AWS_REQUEST_PAYER:
            kwargs['RequestPayer'] = config.AWS_REQUEST_PAYER
        begin = time.time()
        try:
            req = await self.object.get(Range=f'bytes={start}-{start+offset}', **kwargs)
        except botocore.exceptions.ClientError as e:
            await self._close()
            raise FileNotFoundError(f"File not found: {self.filepath}") from e
        elapsed = time.time() - begin
        content_range = req['ResponseMetadata']['HTTPHeaders']['content-range']
        if not config.VERBOSE_LOGS:
            status = req['ResponseMetadata']['HTTPStatusCode']
            logger.debug(f" FINISHED REQUEST in {elapsed} seconds: <STATUS {status}> ({content_range})")
        self._total_bytes_requested += int(req['ResponseMetadata']['HTTPHeaders']['content-length'])
        self._total_requests += 1
        self._requested_ranges.append(tuple([int(v) for v in content_range.split(' ')[-1].split('/')[0].split('-')]))
        data = await req['Body'].read()
        return data

    async def request_json(self):
        kwargs = {}
        if config.AWS_REQUEST_PAYER:
            kwargs['RequestPayer'] = config.AWS_REQUEST_PAYER
        begin = time.time()
        try:
            req = await self.object.get()
        except botocore.exceptions.ClientError as e:
            await self._close()
            raise FileNotFoundError(f"File not found: {self.filepath}") from e
        elapsed = time.time() - begin
        content_range = req['ResponseMetadata']['HTTPHeaders']['content-range']
        if not config.VERBOSE_LOGS:
            status = req['ResponseMetadata']['HTTPStatusCode']
            logger.debug(f" FINISHED REQUEST in {elapsed} seconds: <STATUS {status}> ({content_range})")
        self._total_bytes_requested += int(req['ResponseMetadata']['HTTPHeaders']['content-length'])
        self._total_requests += 1
        self._requested_ranges.append(tuple([int(v) for v in content_range.split(' ')[-1].split('/')[0].split('-')]))
        data = json.loads(await req['Body'].read().decode('utf-8'))
        return data

    async def _close(self) -> None:
        await self.resource.__aexit__('', '', '')

    async def __aenter__(self):
        splits = urlsplit(self.filepath)
        self.resource = await aioboto3.resource('s3').__aenter__()
        self.object = await self.resource.Object(splits.netloc, splits.path[1:])
        return self