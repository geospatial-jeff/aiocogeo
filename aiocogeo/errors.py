"""aiocogeo.errors"""
from dataclasses import dataclass


class CogReadError(Exception):
    """exception base class"""

    ...


class InvalidTiffError(CogReadError):
    """file is not a tiff"""

    ...


class TileNotFoundError(CogReadError):
    """tile not found"""

    ...


class MissingAssets(CogReadError):
    """asset not found (stac)"""

    ...
