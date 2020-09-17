import asyncio
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple, Union
from urllib.parse import urlsplit

import aiohttp
import numpy as np
from PIL import Image
from stac_pydantic.shared import MimeTypes

from .cog import COGReader, CompositeReader, ReaderMixin


@dataclass
class STACReader(ReaderMixin):
    filepath: str
    reader: CompositeReader = None
    include_types: Set[MimeTypes] = field(default_factory=lambda: {MimeTypes.cog})

    async def __aenter__(self):
        splits = urlsplit(self.filepath)
        if splits.scheme in {"http", "https"}:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.filepath) as resp:
                    resp.raise_for_status()
                    item = await resp.json()
        else:
            # TODO: support s3
            pass

        # Create a reader for each asset with a COG mime type
        reader_futs = []
        aliases = []
        for asset in item["assets"]:
            if item["assets"][asset]["type"] in self.include_types:
                reader = COGReader(item["assets"][asset]["href"]).__aenter__()
                reader_futs.append(reader)
                aliases.append(asset)
        self.reader = CompositeReader(await asyncio.gather(*reader_futs), aliases=aliases)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        for reader in self.reader.readers:
            await reader._file_reader._close()


    async def get_tile(self, x: int, y: int, z: int) -> Union[np.ndarray, List[np.ndarray]]:
        return await self.reader.get_tile(x, y, z)

    async def read(
        self,
        bounds: Tuple[float, float, float, float],
        shape: Tuple[int, int],
        resample_method: int = Image.NEAREST,
    ) -> Union[Union[np.ndarray, np.ma.masked_array], List[Union[np.ndarray, np.ma.masked_array]]]:
        return await self.reader.read(bounds, shape, resample_method)

    async def point(self, x: Union[float, int], y: Union[float, int]) -> Union[Union[np.ndarray, np.ma.masked_array], List[Union[np.ndarray, np.ma.masked_array]]]:
        return await self.reader.point(x, y)

    async def preview(
        self,
        max_size: int = 1024,
        height: Optional[int] = None,
        width: Optional[int] = None,
        resample_method: int = Image.NEAREST
    ) -> Union[Union[np.ndarray, np.ma.masked_array], List[Union[np.ndarray, np.ma.masked_array]]]:
        return await self.reader.preview(max_size, height, width, resample_method)