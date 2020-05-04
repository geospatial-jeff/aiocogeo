from setuptools import setup

setup(
    name="async-cog-reader",
    version="0.1",
    author=u"Jeff Albrecht",
    author_email="geospatialjeff@gmail.com",
    url="https://github.com/geospatial-jeff/async-cog-reader",
    license="mit",
    python_requires=">=3.7",
    install_requires=[
        "aiohttp",
        "aiocache",
        "affine",
        "imagecodecs",
        "scikit-image"
    ],
    test_suite="tests",
    setup_requires=[
        'pytest-runner'
    ],
    tests_require=[
        "pytest",
        "pytest-asyncio<0.11.0",
        "pytest-cov",
        "rasterio",
        "rio-tiler==2.0a4"
    ]
)