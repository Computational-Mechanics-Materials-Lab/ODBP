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
#from typing import TypeAlias, Optional
#
## Lists that can be None, used for defaults of lists of frames or nodesets
#NullableIntList: TypeAlias = Optional[list[int]]
#NullableStrList: TypeAlias = Optional[list[str]]
#NodeType: TypeAlais = dict[str, list[int]] | list[list[int]] | list[int]
#NullableNodeType: TypeAlias = Optional[NodeType]
## Types of large data-science types
#NDArrayType: TypeAlias = np.ndarray
#NPZFileType: TypeAlias = np.lib.npyio.NpzFile
#DataFrameType: TypeAlias = pd.DataFrame
#H5PYGroupType: TypeAlias = h5py._hl.group.Group
#H5PYFileType: TypeAlias = h5py.File
#MultiprocessingPoolType: TypeAlias = multiprocessing.Pool
#
######

# Python 3.6+ version
from typing import Union, Tuple, List, Optional, Dict
NullableIntList = Optional[List[int]]
NullableStrList = Optional[List[str]]
NodeType = Union[
    Dict[str, List[int]], List[List[int]], List[int]
]
NullableNodeType = Optional[NodeType]

NDArrayType = np.ndarray
NPZFileType = np.lib.npyio.NpzFile
DataFrameType = pd.DataFrame
H5PYGroupType = h5py.Group
H5PYFileType = h5py.File
MultiprocessingPoolType = multiprocessing.Pool

# Magic # Constants
ODB_MAGIC_NUM: bytes = b"HKSRD0"
HDF_MAGIC_NUM: bytes = b"\x89HDF\r\n"



def confirm(message: str, confirmation: str, default: "Optional[str]" = None) -> bool:
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
