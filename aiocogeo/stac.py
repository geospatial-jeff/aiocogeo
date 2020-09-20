import asyncio
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

from stac_pydantic.shared import Asset, MimeTypes

from .cog import COGReader, CompositeReader
from .filesystems import Filesystem


try:
    import aioboto3
    has_s3 = True
except ModuleNotFoundError:
    has_s3 = False


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
        aliases = []
        for asset in item["assets"]:
            if item["assets"][asset]["type"] in self.include_types:
                reader = AssetReader(
                    filepath=item["assets"][asset]["href"],
                    asset=Asset(name=asset, **item['assets'][asset])
                )
                reader = reader.__aenter__()
                reader_futs.append(reader)
                aliases.append(asset)
        self.readers = await asyncio.gather(*reader_futs)
        self.aliases = aliases
        self.__post_init__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._file_reader._close()
        for reader in self.readers:
            await reader._file_reader._close()