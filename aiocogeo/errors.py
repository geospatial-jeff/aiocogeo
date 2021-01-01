"""aiocogeo.errors"""
from dataclasses import dataclass


@dataclass
class CogReadError(Exception):
    """exception base class"""

    ...


@dataclass
class InvalidTiffError(CogReadError):
    """file is not a tiff"""

    ...


@dataclass
class TileNotFoundError(CogReadError):
    """tile not found"""

    ...


@dataclass
class MissingAssets(CogReadError):
    """asset not found (stac)"""

    ...
