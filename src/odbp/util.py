#!/usr/bin/env python3

"""
Utility methods for odb_plotter
"""

import h5py
import multiprocessing
import pathlib
import numpy as np
import pandas as pd


# Python 3.10+ version
####### Type Aliases for this Project
#from typing import TypeAlias
#
## Lists that can be None, used for defaults of lists of frames or nodesets
#NullableIntList: TypeAlias = list[int] | None
#NullableStrList: TypeAlias = list[str] | None
#
## Types of large data-science types
#NDArrayType: TypeAlias = np.ndarray
#NPZFileType: TypeAlias = np.lib.npyio.NpzFile
#DataFrameType: TypeAlias = pd.core.frame.DataFrame
#H5PYGroupType: TypeAlias = h5py._hl.group.Group
#H5PYFileType: TypeAlias = h5py.File
#MultiprocessingPoolType: TypeAlias = multiprocessing.Pool
#
######

# Python 3.6+ version
from typing import Union, Tuple, List
NullableIntList = Union[List[int], None]
NullableStrList = Union[List[str], None]

NDArrayType = np.ndarray
NPZFileType = np.lib.npyio.NpzFile
DataFrameType = pd.core.frame.DataFrame
H5PYGroupType = h5py._hl.group.Group
H5PYFileType = h5py.File
MultiprocessingPoolType = multiprocessing.Pool


def confirm(message: str, confirmation: str, default: "Union[str, None]" = None) -> bool:
    yes_vals: Union[Tuple[str, str], Tuple[str, str, str]] = ("yes", "y")
    no_vals: Union[Tuple[str, str], Tuple[str, str, str]] = ("no", "n")
    if isinstance(default, str):
        if default.lower() in yes_vals:
            yes_vals = ("yes", "y", "")
            confirmation += " (Y/n)? "
        elif default.lower() in no_vals:
            no_vals = ("no", "n", "")
            confirmation += " (y/N)? "

    else:
        confirmation += " (y/n)? "

    while True:
        print(message)
        user_input: str = input(confirmation).lower()
        if user_input in yes_vals:
            return True
        elif user_input in no_vals:
            return False
        else:
            print("Error: invalid input")
