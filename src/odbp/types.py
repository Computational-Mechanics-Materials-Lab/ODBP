#!/usr/bin/env python3

"""
Utility methods for odb_plotter
"""

import h5py
import multiprocessing
import numpy as np
import pandas as pd
from typing import Union, List, Optional, Dict
from itertools import chain

# Python 3.6+ version
NullableIntList = Optional[List[int]]
NullableStrList = Optional[List[str]]
NodeType = Union[Dict[str, chain], List[chain], chain]
NullableNodeType = Optional[NodeType]

NDArrayType = np.ndarray
NPZFileType = np.lib.npyio.NpzFile
DataFrameType = pd.DataFrame
H5PYGroupType = h5py.Group
H5PYFileType = h5py.File
MultiprocessingPoolType = multiprocessing.Pool
