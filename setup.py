from setuptools import setup

setup(
    name="aiocogeo",
    version="0.1",
    author=u"Jeff Albrecht",
    author_email="geospatialjeff@gmail.com",
    url="https://github.com/geospatial-jeff/async-cog-reader",
    license="mit",
    python_requires=">=3.7",
    install_requires=[
        "aioboto3",
        "aiofiles",
        "aiohttp<=3.6.2",
        "aiocache",
        "affine",
        "imagecodecs",
        "scikit-image",
        "typer",
    ],
    test_suite="tests",
    setup_requires=[
        'pytest-runner'
    ],
    entry_points={"console_scripts": ["aiocogeo=async_cog_reader.scripts.cli:app"]},
    tests_require=[
        "mercantile",
        "morecantile",
        "pytest",
        "pytest-asyncio<0.11.0",
        "pytest-cov",
        "rasterio",
        "rio-tiler==2.0a4",
        "shapely",
    ]
)