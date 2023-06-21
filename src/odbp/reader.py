"""
ODBPlotter npz_to_hdf.py
ODBPlotter
https://www.github.com/Computational-Mechanics-Materials-Lab/ODBPlotter
MIT License (c) 2023

This file provides tools to read .hdf5 files created by the ODBPlotter
converter into pandas dataframes

Originally written by CMML Member CJ Nguyen
"""

import h5py
import multiprocessing
import pathlib
import pandas as pd
from typing import Tuple, List
from .util import DataFrameType, H5PYFileType, MultiprocessingPoolType


def get_odb_data(
    hdf_path: pathlib.Path,
    cpus: int
    ) -> DataFrameType:
    """
    get_node_coords(hdf_path: pathlib.Path) -> DataFrameType
    return a data frame with nodes by integer index and floating point
    3-dimensional coordinates.
    """

    try:
        hdf_file: H5PYFileType
        with h5py.File(hdf_path) as hdf_file:
            step: str
            for step in hdf_file["nodes"].keys():
                frame_name: str
                # TODO dataclass
                args_list: List[Tuple[H5PYFileType, str, str]] = [
                        (hdf_path, step, frame_name)
                        for frame_name
                        in hdf_file["nodes"][step].keys() 
                        ]

                results: List[DataFrameType]
                pool: MultiprocessingPoolType
                with multiprocessing.Pool(processes=cpus) as pool:
                    results = pool.starmap(get_frame_data, args_list)

                return pd.concat(results)

    except (FileNotFoundError, OSError):
        raise Exception("Error accessing .hdf5 file")

    except KeyError:
        raise Exception(f"{hdf_path} file does not include node coordinates "
            'or they are not keyed by "nodes"')


def get_frame_data(
    hdf_path: pathlib.Path,
    step_name: str,
    frame_name: str,
    ) -> DataFrameType:
    """
    get_frame_data(
        hdf_file: open H5PY.File
        step_name: str,
        frame_name: str,
    ) -> DataFrameType
    Given a .hdf5 file, a step name, and a frame name, return
    a dataframe with data for that frame.
    """ 
    
    hdf_file: H5PYFileType
    with h5py.File(hdf_path) as hdf_file:
        return pd.DataFrame(
            data=hdf_file["nodes"][step_name][frame_name][:], 
            columns=["Node Label", "X", "Y", "Z", "Temp", "Time"]
            ).sort_values("Time")
