# [WIP] async-cog-reader [![CircleCI](https://circleci.com/gh/geospatial-jeff/async-cog-reader/tree/master.svg?style=svg)](https://circleci.com/gh/geospatial-jeff/async-cog-reader/tree/master)


### Usage
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


#### Partial Read
```python
import asyncio
from async_cog_reader import COGReader

infile = "https://async-cog-reader-test-data.s3.amazonaws.com/lzw_cog.tif"

async def main():
    async with COGReader(infile) as cog:
        assert cog.epsg == 26911
        # Projected bounds in native crs of image (in this case `EPSG:26911`)
        bounds = [367791.55780407554, 3769929.85023777, 368819.5343714542, 3770924.9263116163]
        tile = await cog.read(bounds=bounds, shape=(256, 256, 3))

asyncio.run(main())
```