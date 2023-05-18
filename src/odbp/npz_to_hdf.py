#!/usr/bin/env python3

"""
ODBPlotter npz_to_hdf.py
ODBPlotter
https://www.github.com/Computational-Mechanics-Materials-Lab/ODBPlotter
MIT License (c) 2023

This file exposes the npz_to_hdf() method, used to translate a hierarchical
directory of .npz files into a .hdf5 file.

The inputs must be:
    A Pathlike for the directory of the source files
    A Pathlike of the resulting output .hdf5 file

Originally written by CMML Member CJ Nguyen
"""


import h5py
import numpy as np
from pathlib import Path
from os import PathLike, walk
from .util import NDArrayType, H5PYFileType


def convert_npz_to_hdf(
    hdf_path: PathLike,
    npz_dir: PathLike = Path("tmp_npz")
    ) -> None:

    # To my knowledge, h5py does not ship type hints
    hdf5_file: H5PYFileType
    with h5py.File(hdf_path, "w") as hdf5_file:
        root: PathLike
        files: list[PathLike]
        for root, _, files in walk(npz_dir, topdown=True):
            filename: PathLike
            for filename in files:
                item: PathLike = Path(root, filename)
                read_npz_to_hdf(item, npz_dir, hdf5_file)


def read_npz_to_hdf(item: PathLike, npz_dir: PathLike, hdf5_file: H5PYFileType) -> None:
    npz: NDArrayType = np.load(item)
    arr: NDArrayType = npz[npz.files[0]]
    item_name: PathLike = item.relative_to(npz_dir)
    item_name = Path(item_name.parent, item_name.stem)
    hdf5_file.create_dataset(
        str(item_name).replace("\\", "/"), data=arr, compression="gzip"
        )
