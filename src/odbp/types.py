#!/usr/bin/env python3

"""
Utility methods for odb_plotter
"""

import h5py
import multiprocessing
import numpy as np
import pandas as pd
from itertools import chain

# Python 3.6+ version
NodeType = dict[str, chain] | list[chain] | chain

NDArrayType = np.ndarray
NPZFileType = np.lib.npyio.NpzFile
DataFrameType = pd.DataFrame
H5PYGroupType = h5py.Group
H5PYFileType = h5py.File
MultiprocessingPoolType = multiprocessing.Pool
