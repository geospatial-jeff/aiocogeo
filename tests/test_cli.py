from morecantile.models import TileMatrixSet
import pytest

from .conftest import TEST_DATA

from aiocogeo.scripts.cli import app


@pytest.mark.parametrize("infile", TEST_DATA[:-1])
def test_info(cli_runner, infile):
    result = cli_runner.invoke(app, ["info", infile])
    assert result.exit_code == 0


@pytest.mark.parametrize("infile", TEST_DATA)
def test_create_tms(cli_runner, infile):
    result = cli_runner.invoke(app, ["create-tms", infile])
    assert result.exit_code == 0

    TileMatrixSet.parse_raw(result.output)