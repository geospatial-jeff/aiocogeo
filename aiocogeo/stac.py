import asyncio
from dataclasses import dataclass, field
from typing import Optional, Set
from urllib.parse import urlsplit

import aiohttp
from stac_pydantic.shared import Asset, MimeTypes

from .cog import COGReader, CompositeReader


@dataclass
class STACReader(CompositeReader):
    filepath: Optional[str] = None
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
                reader = COGReader(item["assets"][asset]["href"])
                reader.asset = Asset(name=asset, **item["assets"][asset])
                reader = reader.__aenter__()
                reader_futs.append(reader)
                aliases.append(asset)
        self.readers = await asyncio.gather(*reader_futs)
        self.aliases = aliases
        self.__post_init__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        for reader in self.readers:
            await reader._file_reader._close()