import asyncio
from functools import wraps
import json as _json

import typer

from aiocogeo import COGReader

app = typer.Typer()


def coro(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


def _make_bold(s, **kwargs):
    return typer.style(s, bold=True, **kwargs)


def _get_ifd_stats(ifds):
    ifd_stats = []
    for idx, ifd in enumerate(ifds):
        tile_sizes = [b / 1000 for b in ifd.TileByteCounts.value]
        mean_tile_size = round(sum(tile_sizes) / len(tile_sizes), 3)
        ifd_stats.append({
            'id': idx,
            'size': (ifd.ImageWidth.value, ifd.ImageHeight.value),
            'block_size': (ifd.TileWidth.value, ifd.TileHeight.value),
            'tile_sizes': {
                'min': min(tile_sizes),
                'max': max(tile_sizes),
                'mean': mean_tile_size
            }
        })
    return ifd_stats


def _create_ifd_table(ifds, start="\t"):
    table = (
        f"{start}{_make_bold('Id', underline=True):<20}{_make_bold('Size', underline=True):<27}"
        f"{_make_bold('BlockSize', underline=True):<26}{_make_bold('MinTileSize (KB)', underline=True):<33}"
        f"{_make_bold('MaxTileSize (KB)', underline=True):<33}{_make_bold('MeanTileSize (KB)', underline=True):<33}"
    )
    for stats in _get_ifd_stats(ifds):
        table += (
            f"\n\t\t{stats['id']:<8}"
            f"{'x'.join([str(val) for val in stats['size']]):<15}"
            f"{'x'.join([str(val) for val in stats['block_size']]):<14}"
            f"{stats['tile_sizes']['min']:<21}"
            f"{stats['tile_sizes']['max']:<21}"
            f"{stats['tile_sizes']['mean']:<30}"
        )
    return table

def _create_json_info(cog):
    profile = cog.profile

    info = {
        "file": cog.filepath,
        "profile": {
            "width": profile['width'],
            "height": profile['height'],
            "bands": profile['count'],
            "dtype": profile['dtype'],
            "crs": profile['crs'],
            "origin": (profile['transform'].c, profile['transform'].f),
            "resolution": (profile['transform'].a, profile['transform'].e),
            "bbox": cog.bounds,
            "compression": cog.ifds[0].compression,
            "internal_mask": cog.is_masked
        },
        "ifd": _get_ifd_stats(cog.ifds)
    }

    if cog.is_masked:
        info['mask_ifd'] = _get_ifd_stats(cog.mask_ifds)

    return info


@app.command(
    short_help="Read COG metadata.",
    help="Read COG profile, IFD, and mask IFD metadata.",
    no_args_is_help=True
)
@coro
async def info(
    filepath: str = typer.Argument(..., file_okay=True),
    json: bool = typer.Option(False, show_default=True, help="JSON-formatted response")
):
    sep = 25
    async with COGReader(filepath) as cog:
        if json:
            return typer.echo(_json.dumps(_create_json_info(cog), indent=1))
        profile = cog.profile
        typer.echo(
            f"""
        {_make_bold("FILE INFO:", underline=True)} {_make_bold(filepath)}

          {_make_bold("PROFILE")}
            {_make_bold("Width:"):<{sep}} {profile['width']}
            {_make_bold("Height:"):<{sep}} {profile['height']}
            {_make_bold("Bands:"):<{sep}} {profile['count']}
            {_make_bold("Dtype:"):<{sep}} {profile['dtype']}
            {_make_bold("Crs:"):<{sep}} {profile['crs']}
            {_make_bold("Origin:"):<{sep}} ({profile['transform'].c}, {profile['transform'].f})
            {_make_bold("Resolution:"):<{sep}} ({profile['transform'].a}, {profile['transform'].e})
            {_make_bold("BoundingBox:"):<{sep}} {cog.bounds}
            {_make_bold("Compression:"):<{sep}} {cog.ifds[0].compression}
            {_make_bold("Internal mask:"):<{sep}} {cog.is_masked}
        """
        )
        typer.echo(
            f"""\t  {_make_bold("IFD")}
            {_create_ifd_table(cog.ifds)}
        """
        )
        if cog.is_masked:
            typer.echo(
                f"""\t  {_make_bold("MASK IFD")}
                {_create_ifd_table(cog.mask_ifds, start="")}
            """
            )


@app.command(
    short_help="Create OGC TileMatrixSet.",
    help="Create OGC TileMatrixSet representation of the COG where each IFD is a unique tile matrix.",
    no_args_is_help=True
)
@coro
async def create_tms(filepath: str):
    async with COGReader(filepath) as cog:
        typer.echo(_json.dumps(cog.create_tile_matrix_set(), indent=1))
