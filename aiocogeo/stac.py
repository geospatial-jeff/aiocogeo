import asyncio
from dataclasses import dataclass, field
from typing import List, Set, Tuple
from urllib.parse import urlsplit

import aiohttp
import numpy as np
from PIL import Image
from stac_pydantic.shared import MimeTypes

from .cog import COGReader, CompositeReader


@dataclass
class STACReader:
    filepath: str
    reader: CompositeReader = None
    include_types: Set[MimeTypes] = field(default_factory={MimeTypes.cog})

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
        for asset in item["assets"]:
            if item["assets"][asset]["type"] in self.include_types:
                reader = COGReader(item["assets"][asset]["href"]).__aenter__()
                reader_futs.append(reader)
        self.reader = CompositeReader(await asyncio.gather(*reader_futs))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        for reader in self.reader.readers:
            await reader._file_reader._close()