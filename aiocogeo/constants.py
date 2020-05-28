# TODO: Add more compressions

WEB_MERCATOR_EPSG = 3857

COMPRESSIONS = {
    1: "uncompressed",
    5: "lzw",
    6: "jpeg",
    7: "jpeg",
    8: "deflate",
    50001: "webp",
}

INTERLEAVE = {1: "pixel", 2: "band"}

# https://www.awaresystems.be/imaging/tiff/tifftags/photometricinterpretation.html
PHOTOMETRIC = {
    0: "miniswhite",
    1: "minisblack",
    2: "rgb",
    3: "palette",
    4: "mask",
    5: "separated",
    6: "ycbcr",
    8: "cielab",
    9: "icclab",
    10: "itulab",
}

# https://github.com/cgohlke/tifffile/blob/master/tifffile/tifffile.py#L7901-L7986
# Map SampleFormat and BitsPerSample tags to numpy dtype
SAMPLE_DTYPES = {
    # UINT
    (1, 1): "?",  # bitmap
    (1, 2): "B",
    (1, 3): "B",
    (1, 4): "B",
    (1, 5): "B",
    (1, 6): "B",
    (1, 7): "B",
    (1, 8): "B",
    (1, 9): "H",
    (1, 10): "H",
    (1, 11): "H",
    (1, 12): "H",
    (1, 13): "H",
    (1, 14): "H",
    (1, 15): "H",
    (1, 16): "H",
    (1, 17): "I",
    (1, 18): "I",
    (1, 19): "I",
    (1, 20): "I",
    (1, 21): "I",
    (1, 22): "I",
    (1, 23): "I",
    (1, 24): "I",
    (1, 25): "I",
    (1, 26): "I",
    (1, 27): "I",
    (1, 28): "I",
    (1, 29): "I",
    (1, 30): "I",
    (1, 31): "I",
    (1, 32): "I",
    (1, 64): "Q",
    # VOID : treat as UINT
    (4, 1): "?",  # bitmap
    (4, 2): "B",
    (4, 3): "B",
    (4, 4): "B",
    (4, 5): "B",
    (4, 6): "B",
    (4, 7): "B",
    (4, 8): "B",
    (4, 9): "H",
    (4, 10): "H",
    (4, 11): "H",
    (4, 12): "H",
    (4, 13): "H",
    (4, 14): "H",
    (4, 15): "H",
    (4, 16): "H",
    (4, 17): "I",
    (4, 18): "I",
    (4, 19): "I",
    (4, 20): "I",
    (4, 21): "I",
    (4, 22): "I",
    (4, 23): "I",
    (4, 24): "I",
    (4, 25): "I",
    (4, 26): "I",
    (4, 27): "I",
    (4, 28): "I",
    (4, 29): "I",
    (4, 30): "I",
    (4, 31): "I",
    (4, 32): "I",
    (4, 64): "Q",
    # INT
    (2, 8): "b",
    (2, 16): "h",
    (2, 32): "i",
    (2, 64): "q",
    # IEEEFP : 24 bit not supported by numpy
    (3, 16): "e",
    # (3, 24): '',  #
    (3, 32): "f",
    (3, 64): "d",
    # COMPLEXIEEEFP
    (6, 64): "F",
    (6, 128): "D",
    # RGB565
    (1, (5, 6, 5)): "B",
    # COMPLEXINT : not supported by numpy
}

# https://github.com/python-pillow/Pillow/blob/master/src/PIL/TiffTags.py
TIFF_TAGS = {
    254: "NewSubfileType",
    256: "ImageWidth",
    257: "ImageHeight",
    258: "BitsPerSample",
    259: "Compression",
    262: "PhotometricInterpretation",
    277: "SamplesPerPixel",
    284: "PlanarConfiguration",
    317: "Predictor",
    322: "TileWidth",
    323: "TileHeight",
    324: "TileOffsets",
    325: "TileByteCounts",
    339: "SampleFormat",
    347: "JPEGTables",
    33550: "ModelPixelScaleTag",
    33922: "ModelTiepointTag",
    34735: "GeoKeyDirectoryTag",
}
