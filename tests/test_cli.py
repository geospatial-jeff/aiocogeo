import json

import pytest
from morecantile.models import TileMatrixSet

from aiocogeo import config
from aiocogeo.scripts.cli import app

from .conftest import TEST_DATA


@pytest.mark.parametrize("infile", TEST_DATA[:-1])
def test_info(cli_runner, infile, monkeypatch):
    monkeypatch.setattr(config, "ENABLE_BLOCK_CACHE", False)
    monkeypatch.setattr(config, "ENABLE_HEADER_CACHE", False)
    result = cli_runner.invoke(app, ["info", infile])
    assert result.exit_code == 0


@pytest.mark.parametrize("infile", TEST_DATA[:-1])
def test_info_json_formatted(cli_runner, infile, monkeypatch):
    monkeypatch.setattr(config, "ENABLE_BLOCK_CACHE", False)
    monkeypatch.setattr(config, "ENABLE_HEADER_CACHE", False)
    json_result = cli_runner.invoke(app, ["info", infile, "--json"])
    assert json_result.exit_code == 0

    json.loads(json_result.output)


@pytest.mark.parametrize("infile", TEST_DATA)
def test_create_tms(cli_runner, infile, monkeypatch):
    monkeypatch.setattr(config, "ENABLE_BLOCK_CACHE", False)
    monkeypatch.setattr(config, "ENABLE_HEADER_CACHE", False)
    result = cli_runner.invoke(app, ["create-tms", infile])
    assert result.exit_code == 0

    TileMatrixSet.parse_raw(result.output)
