"""Configurable values exposed to user as environment variables"""
import os
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)

# Changes the log level
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "ERROR")
logger.setLevel(logging.ERROR)

# https://gdal.org/user/virtual_file_systems.html#vsicurl-http-https-ftp-files-random-access
# Defines the number of bytes read in the first GET request at file opening
# Can help performance when reading images with a large header
INGESTED_BYTES_AT_OPEN: int = os.getenv("INGESTED_BYTES_AT_OPEN", 16384)