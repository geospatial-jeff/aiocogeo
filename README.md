# aiocogeo [![CircleCI](https://circleci.com/gh/geospatial-jeff/aiocogeo/tree/master.svg?style=svg)](https://circleci.com/gh/geospatial-jeff/aiocogeo/tree/master)[![codecov](https://codecov.io/gh/geospatial-jeff/aiocogeo/branch/master/graph/badge.svg)](https://codecov.io/gh/geospatial-jeff/aiocogeo)

## Installation
```
pip install aiocogeo

# With S3 filesystem
pip install aiocogeo[s3]
```

## Usage
COGs are opened using the `COGReader` asynchronous context manager:

```python
from aiocogeo import COGReader

async with COGReader("http://cog.tif") as cog:
    ...
```

Several filesystems are supported:
- **HTTP/HTTPS** (`http://`, `https://`)
- **S3** (`s3://`)
- **File** (`/`)

### Metadata
Generating a [rasterio-style profile](https://rasterio.readthedocs.io/en/latest/topics/profiles.html) for the COG:

```python
async with COGReader("https://async-cog-reader-test-data.s3.amazonaws.com/lzw_cog.tif") as cog:
    print(cog.profile)

>>> {'driver': 'GTiff', 'width': 10280, 'height': 12190, 'count': 3, 'dtype': 'uint8', 'transform': Affine(0.6, 0.0, 367188.0,
       0.0, -0.6, 3777102.0), 'blockxsize': 512, 'blockysize': 512, 'compress': 'lzw', 'interleave': 'pixel', 'crs': 'EPSG:26911', 'tiled': True, 'photometric': 'rgb'}
```

#### Lower Level Metadata
A COG is composed of several IFDs, each with many TIFF tags:

```python
from aiocogeo.ifd import IFD
from aiocogeo.tag import Tag

async with COGReader("https://async-cog-reader-test-data.s3.amazonaws.com/lzw_cog.tif") as cog:
    for ifd in cog:
        assert isinstance(ifd, IFD)
        for tag in ifd:
            assert isinstance(tag, Tag)
```

Each IFD contains more granular metadata about the image than what is included in the profile.  For example, finding the
tilesize for each IFD:

```python
async with COGReader("https://async-cog-reader-test-data.s3.amazonaws.com/lzw_cog.tif") as cog:
    for ifd in cog:
        print(ifd.TileWidth.value, ifd.TileHeight.value)

>>> 512 512
    128 128
    128 128
    128 128
    128 128
    128 128
```

More advanced use cases may need access to tag-level metadata:
```python
async with COGReader("https://async-cog-reader-test-data.s3.amazonaws.com/lzw_cog.tif") as cog:
    first_ifd = cog.ifds[0]
    assert first_ifd.tag_count == 24

    for tag in first_ifd:
        print(tag)

>>> Tag(code=258, name='BitsPerSample', tag_type=TagType(format='H', size=2), count=3, length=6, value=(8, 8, 8))
    Tag(code=259, name='Compression', tag_type=TagType(format='H', size=2), count=1, length=2, value=5)
    Tag(code=257, name='ImageHeight', tag_type=TagType(format='H', size=2), count=1, length=2, value=12190)
    Tag(code=256, name='ImageWidth', tag_type=TagType(format='H', size=2), count=1, length=2, value=10280)
    ...
```

### Image Data
The reader also has methods for reading internal image tiles and performing partial reads.  Currently only jpeg, lzw, deflate, packbits, and webp compressions are supported.

#### Image Tiles
Reading the top left tile of an image at native resolution:

```python
async with COGReader("https://async-cog-reader-test-data.s3.amazonaws.com/webp_cog.tif") as cog:
    x = y = z = 0
    tile = await cog.get_tile(x, y, z)
    
    ifd = cog.ifds[z]
    assert tile.shape == (ifd.bands, ifd.TileHeight.value, ifd.TileWidth.value)
```

<p align="center">
  <img width="300" height="300" src="https://async-cog-reader-test-data.s3.amazonaws.com/readme/naip_top_left_tile.jpg">
</p>


#### Partial Read
You can read a portion of the image by specifying a bounding box in the native crs of the image and an output shape:

```python
async with COGReader("https://async-cog-reader-test-data.s3.amazonaws.com/webp_cog.tif") as cog:
    assert cog.epsg == 26911
    partial_data = await cog.read(bounds=(368461,3770591,368796,3770921), shape=(512,512))
```

<p align="center">
  <img width="300" height="300" src="https://async-cog-reader-test-data.s3.amazonaws.com/readme/partial_read.jpeg">
</p>

#### Internal Masks
If the COG has an internal mask, the returned array will be a masked array:

```python
import numpy as np

async with COGReader("https://async-cog-reader-test-data.s3.amazonaws.com/naip_image_masked.tif") as cog:
    assert cog.is_masked

    tile = await cog.get_tile(0,0,0)
    assert np.ma.is_masked(tile)
```

<p align="center">
  <img src="https://async-cog-reader-test-data.s3.amazonaws.com/readme/masked_tile.jpg" width="300" />
  <img src="https://async-cog-reader-test-data.s3.amazonaws.com/readme/mask.jpg" width="300" /> 
</p>

### Configuration
Configuration options are exposed through environment variables:
- **INGESTED_BYTES_AT_OPEN** - defines the number of bytes in the first GET request at file opening (defaults to 16KB)
- **HEADER_CHUNK_SIZE** - chunk size used to read header (defaults to 16KB)
- **ENABLE_BLOCK_CACHE** - determines if image blocks are cached in memory (defaults to TRUE)
- **ENABLE_HEADER_CACHE** - determines if COG headers are cached in memory (defaults to TRUE)
- **HTTP_MERGE_CONSECUTIVE_RANGES** - determines if consecutive ranges are merged into a single request (defaults to FALSE)
- **BOUNDLESS_READ** - determines if internal tiles outside the bounds of the IFD are read (defaults to TRUE)
- **BOUNDLESS_READ_FILL_VALUE** - determines the value used to fill boundless reads (defaults to 0)
- **LOG_LEVEL** - determines the log level used by the package (defaults to ERROR)
- **VERBOSE_LOGS** - enables verbose logging, designed for use when `LOG_LEVEL=DEBUG` (defaults to FALSE)
- **AWS_REQUEST_PAYER** - set to `requester` to enable reading from S3 RequesterPays buckets.

Refer to [`aiocogeo/config.py`](https://github.com/geospatial-jeff/aiocogeo/blob/master/aiocogeo/config.py) for more details about configuration options.

## CLI
```
$ aiocogeo --help
Usage: aiocogeo [OPTIONS] COMMAND [ARGS]...

Options:
  --install-completion [bash|zsh|fish|powershell|pwsh]
                                  Install completion for the specified shell.
  --show-completion [bash|zsh|fish|powershell|pwsh]
                                  Show completion for the specified shell, to
                                  copy it or customize the installation.

  --help                          Show this message and exit.

Commands:
  create-tms  Create OGC TileMatrixSet.
  info        Read COG metadata.

```