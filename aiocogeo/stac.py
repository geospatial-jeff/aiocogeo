import asyncio
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

from stac_pydantic.shared import Asset, MimeTypes

from .cog import COGReader, CompositeReader
from .filesystems import Filesystem
from .errors import MissingAssets


@dataclass
class AssetReader(COGReader):
    asset: Asset = Asset


@dataclass
class STACReader(CompositeReader):
    filepath: Optional[str] = None
    include_types: Set[MimeTypes] = field(default_factory=lambda: {MimeTypes.cog})

    kwargs: Optional[Dict] = field(default_factory=dict)


    async def __aenter__(self):
        async with Filesystem.create_from_filepath(self.filepath, **self.kwargs) as file_reader:
            self._file_reader = file_reader
            item = await file_reader.request_json()

        # Create a reader for each asset with a COG mime type
        reader_futs = []
        for asset in item["assets"]:
            if item["assets"][asset]["type"] in self.include_types:
                reader = AssetReader(
                    filepath=item["assets"][asset]["href"],
                    asset=Asset(name=asset, **item['assets'][asset])
                )
                reader_futs.append(reader)

        if not reader_futs:
            raise MissingAssets(f"No assets found of type {self.include_types}")

        reader_futs = map(lambda r: r.__aenter__(), filter(self.filter, reader_futs))
        self.readers = await asyncio.gather(*reader_futs)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._file_reader._close()
        for reader in self.readers:
            await reader._file_reader._close()