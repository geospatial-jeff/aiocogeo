from dataclasses import dataclass


@dataclass
class CogReadError(Exception):
    message: str


@dataclass
class InvalidTiffError(CogReadError):
    ...


@dataclass
class TileNotFoundError(CogReadError):
    ...
