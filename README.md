# [WIP] async-cog-reader [![CircleCI](https://circleci.com/gh/geospatial-jeff/async-cog-reader/tree/master.svg?style=svg)](https://circleci.com/gh/geospatial-jeff/async-cog-reader/tree/master)


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