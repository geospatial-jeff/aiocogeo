"""Configurable values exposed to user as environment variables"""
import logging
import os

# Changes the log level
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "ERROR")

logging.basicConfig(level=LOG_LEVEL)

# Enables verbose logging.  It is recommended to also use ``LOG_LEVEL=DEBUG``.
VERBOSE_LOGS: bool = False if os.getenv("VERBOSE_LOGS", "FALSE") == "FALSE" else True

# https://gdal.org/user/virtual_file_systems.html#vsicurl-http-https-ftp-files-random-access
# Defines the number of bytes read in the first GET request at file opening
# Can help performance when reading images with a large header
INGESTED_BYTES_AT_OPEN: int = os.getenv("INGESTED_BYTES_AT_OPEN", 16384)

# Defines the chunk size used for additional GET requests required to read the header
HEADER_CHUNK_SIZE: int = os.getenv("HEADER_CHUNK_SIZE", 16384)

# https://trac.osgeo.org/gdal/wiki/ConfigOptions#VSI_CACHE
# Determines if in-memory block caching is enabled
ENABLE_BLOCK_CACHE: bool = True if os.getenv(
    "ENABLE_BLOCK_CACHE", "TRUE"
).upper() == "TRUE" else False


# enable caching of header requests, similar to GDAL's VSI CACHE
# https://trac.osgeo.org/gdal/wiki/ConfigOptions#VSI_CACHE
ENABLE_HEADER_CACHE: bool = True if os.getenv(
    "ENABLE_HEADER_CACHE", "TRUE"
).upper() == "TRUE" else False


# https://trac.osgeo.org/gdal/wiki/ConfigOptions#GDAL_HTTP_MERGE_CONSECUTIVE_RANGES
# Determines if consecutive range requests are merged into a single request, reducing the number of HTTP GET range
# requests required to read consecutive internal image tiles
HTTP_MERGE_CONSECUTIVE_RANGES: bool = True if os.getenv(
    "HTTP_MERGE_CONSECUTIVE_RANGES", "FALSE"
).upper() == "TRUE" else False


# Determines if internal tiles outside the bounds of the IFD are read. When set to ``TRUE`` (default), if a partial read
# isn't fully covered by internal tiles, missing tiles will be created using the fill value defined by the
# ``BOUNDLESS_READ_FILL_VALUE`` config option. When set to ``FALSE``, an exception will be raised instead
BOUNDLESS_READ: bool = False if os.getenv(
    "BOUNDLESS_READ", "TRUE"
) == "FALSE" else True


# Determines the fill value used for boundless reads
BOUNDLESS_READ_FILL_VALUE: int = int(os.getenv("BOUNDLESS_READ_FILL_VALUE", "0"))

# Enable reading from a S3 requester pays bucket
AWS_REQUEST_PAYER: str = os.getenv(
    "AWS_REQUEST_PAYER", None
)
