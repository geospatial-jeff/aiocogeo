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

# https://trac.osgeo.org/gdal/wiki/ConfigOptions#VSI_CACHE
# Determines if in-memory block caching is enabled
ENABLE_BLOCK_CACHE: bool = True if os.getenv(
    "ENABLE_BLOCK_CACHE", "TRUE"
).upper() == "TRUE" else False

# https://trac.osgeo.org/gdal/wiki/ConfigOptions#GDAL_HTTP_MERGE_CONSECUTIVE_RANGES
# Determines if consecutive range requests are merged into a single request, reducing the number of HTTP GET range
# requests required to read consecutive internal image tiles
HTTP_MERGE_CONSECUTIVE_RANGES: bool = True if os.getenv(
    "HTTP_MERGE_CONSECUTIVE_RANGES", "FALSE"
).upper() == "TRUE" else False
