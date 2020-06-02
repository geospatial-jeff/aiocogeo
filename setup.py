from setuptools import setup, find_packages

with open("README.md") as f:
    desc = f.read()

setup(
    name="aiocogeo",
    description="Asynchronous cogeotiff reader",
    long_description=desc,
    long_description_content_type="text/markdown",
    version="0.0.3",
    author=u"Jeff Albrecht",
    author_email="geospatialjeff@gmail.com",
    url="https://github.com/geospatial-jeff/aiocogeo",
    license="mit",
    python_requires=">=3.7",
    classifiers=[
        'Intended Audience :: Developers',
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        'License :: OSI Approved :: MIT License',
    ],
    keywords="cogeo COG",
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
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
    entry_points={"console_scripts": ["aiocogeo=aiocogeo.scripts.cli:app"]},
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