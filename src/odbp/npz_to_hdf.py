#!/usr/bin/env python3

"""
ODBPlotter npz_to_hdf.py
ODBPlotter
https://www.github.com/Computatinoal-Mechanics-Materials-Lab/ODBPlotter
MIT License (c) 2023

This file exposes the npz_to_hdf() method, used to translate a hierarchical
directory of .npz files into a .hdf5 file.

The inputs must be:
    A Pathlike for the directory of the source files
    A Pathlike of the resulting output .hdf5 file

Originally written by CMML Member CJ Nguyen
"""


import os
import h5py
import numpy as np
from pathlib import Path
from typing import TypeAlias, Union, Any


# Global Type Aliases for this file
PathLikeType: TypeAlias = Union[str, os.PathLike]
NDArrayType: TypeAlias = np.ndarray

def convert_npz_to_hdf(
    output_file: PathLikeType,
    npz_dir: PathLikeType = Path("tmp_npz")
    ) -> None:

    # To my knowledge, h5py does not ship type hints
    hdf5_file: Any
    with h5py.File(output_file, "w") as hdf5_file:
        root: PathLikeType
        files: list[PathLikeType]
        for root, _, files in os.walk(npz_dir, topdown=True):
            filename: PathLikeType
            for filename in files:
                item: PathLikeType = Path(root, filename)
                read_npz_to_hdf(item, npz_dir, hdf5_file)


def read_npz_to_hdf(item: PathLikeType, npz_dir: PathLikeType, hdf5_file: Any) -> None:
    npz: NDArrayType = np.load(item)
    arr: NDArrayType = npz[npz.files[0]]
    item_name: PathLikeType = item.relative_to(npz_dir)
    item_name = Path(item_name.parent, item_name.stem)
    hdf5_file.create_dataset(str(item_name).replace("\\", "/"), data=arr, compression="gzip")
