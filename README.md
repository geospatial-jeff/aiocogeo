# [WIP] async-cog-reader


#### Usage
```python
import asyncio
from async_cog_reader import COGReader

infile = "http://cog.tif"

async def main():
    async with COGReader(infile) as cog:
        for ifd in cog.ifds:
            print(ifd.tags["ImageWidth"].value)

asyncio.run(main())
```