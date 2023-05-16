#!/usr/bin/env python

"""
ODBPlotter convert.py

ODBPlotter
https://www.github.com/Computational-Mechanics-Materials-Lab/ODBPlotter
MIT License (c) 2023

This file exposes the convert_odb_to_hdf() method, used to translate a
.odb file to a .hdf5 file.

The inputs must be:
    A Pathlike with a path to a .odb file.
    A string which referse to the abaqus executable
    A list of strs denoting which nodesets should be selected
    or None (default to the first nodeset)
    A list denoting which frames should be selected or None (default to all)

The output will be the path to a newly created .hdf5 file.
"""


import os
import pickle
from shutil import rmtree
from pathlib import Path
from subprocess import run
from typing import TypeAlias, Union, TextIO
from npz_to_hdf import convert_npz_to_hdf


# Global Type Aliases for this file
PathLikeType: TypeAlias = Union[str, os.PathLike]
NullableIntListUnion: TypeAlias = Union[None, list[int]]
NullableStrListUnion: TypeAlias = Union[None, list[str]]


def convert_odb_to_hdf(
    odb_path: PathLikeType,
    abaqus_executable: str = "abaqus",
    nodesets: NullableStrListUnion = None,
    frames: NullableIntListUnion = None
    ) -> PathLikeType:
    """
    convert_odb_to_hdf(
        odb_path: PathLikeType,
        abaqus_executable: str = "abaqus",
        nodesets: NullableStrListUnion = None,
        frames: NullableIntListUnion = None
        ) -> PathLikeType

    abaqus_executable will be the string name of the command-line executable
    version of abaqus being used. By default, "abaqus"

    In normal practice, one should separate file opening from file operations.
    However, due to the proprietary nature of the .odb file format, we must
    pass references to the .odb file by the PathLike filename

    The inputs must be:
        A Pathlike with a path to a .odb file.
        A string which referse to the abaqus executable
        A list of strs denoting which nodesets should be selected
        or None (default to the first nodeset)
        A list denoting which frames should be selected or None (default to all)

    The output will be the path to a newly created .hdf5 file.
    """

    # Normally we'd use relative imports, but that doesn't work with a
    # Python 2 file, so instead we use the absolute path to that Python 2 file
    odb_to_npz_script_path: PathLikeType = Path(
        Path(__file__).parent,
        "py2_scripts",
        "odb_to_npz.py"
        )

    # The nature of sending these values to a Python 2 subprocess necessitates
    # that non-string arguments (i.e. None or lists of ints) be pickled with
    # Pickle protocol 2 and sent to the subprocess that way.
    odb_to_npz_conversion_pickle_path: PathLikeType = Path(
        Path.cwd(),
        "odb_to_npz_conversion.pickle"
    )

    npz_result_path: PathLikeType = Path(
        Path.cwd(),
        "npz_path.pickle"
    )

    odb_to_npz_pickle_input_dict: dict[str, Union[list[str], list[int], None]]
    odb_to_npz_pickle_input_dict = {
        "nodesets": nodesets,
        "frames": frames
    }

    pickle_file: TextIO
    with open(odb_to_npz_conversion_pickle_path, "wb") as pickle_file:
        pickle.dump(odb_to_npz_pickle_input_dict, pickle_file, protocol=2)

    odb_convert_args: list[PathLikeType] = [
        abaqus_executable,
        "python",
        odb_to_npz_script_path,
        odb_path,
        odb_to_npz_conversion_pickle_path,
        npz_result_path
        ]

    run(odb_convert_args, shell=True)

    result_file: TextIO
    result_dir: PathLikeType
    with open(npz_result_path, "rb") as result_file:
        result_dir = Path(pickle.load(result_file))

    Path.unlink(npz_result_path)

    hdf_path = odb_path.parent.parent / "hdfs" / f"{odb_path.stem}.hdf5"

    convert_npz_to_hdf(hdf_path, result_dir)

    rmtree(result_dir)


if __name__ == "__main__":
    odb_file: str = "v3_05mm_i0_01_T_coord.odb"
    path: PathLikeType = Path("C:/", "users", "ch3136", "testing", "odbs", odb_file)
    if path.exists():
        convert_odb_to_hdf(path)