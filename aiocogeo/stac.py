import asyncio
from dataclasses import dataclass
from typing import List
from urllib.parse import urlsplit

import aiohttp

from .cog import COGReader


@dataclass
class STACReader:
    filepath: str
    readers: List[COGReader] = None

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

        reader_futs = []
        for asset in item["assets"]:
            if item["assets"][asset]["type"] == "image/x.geotiff":
                reader = COGReader(item["assets"][asset]["href"]).__aenter__()
                reader_futs.append(reader)
        self.readers = await asyncio.gather(*reader_futs)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        for reader in self.readers:
            await reader._file_reader._close()

    async def get_tile(self, x: int, y: int, z: int):
        futs = [reader.get_tile(x, y, z) for reader in self.readers]
        return await asyncio.gather(*futs)
