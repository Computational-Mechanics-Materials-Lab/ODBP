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
from typing import Tuple, List, Dict
from .types import DataFrameType, H5PYFileType, MultiprocessingPoolType


def get_odb_data(
    hdf_path: pathlib.Path,
    cpus: int
    ) -> "Tuple[Dict[str, str], DataFrameType]":
    """
    get_node_coords(hdf_path: pathlib.Path) -> DataFrameType
    return a data frame with nodes by integer index and floating point
    3-dimensional coordinates.
    """

    try:
        hdf_file: H5PYFileType
        with h5py.File(hdf_path) as hdf_file:
            dataset_name: str = list(hdf_file.keys())[0]
            final_result_attrs: Dict[str, str] = dict(hdf_file[dataset_name].attrs)
            steps_keys: List[str] = list(hdf_file[dataset_name].keys())
            step_key: str
            results_dfs: List[List[DataFrameType]] = list()
            for step_key in steps_keys:
                frame_keys: List[str] = list(hdf_file[dataset_name][step_key].keys())
                frame_key: str
                args_list: List[Tuple[pathlib.Path, str, str, str]] = [
                    (hdf_path, dataset_name, step_key, frame_key)
                    for frame_key
                    in frame_keys
                ]
                pool: MultiprocessingPoolType
                with multiprocessing.Pool(processes=cpus) as pool:
                    results: List[DataFrameType] = pool.starmap(get_frame_data, args_list)
                results_dfs.append(pd.concat(results))

        final_result: DataFrameType = pd.concat(results_dfs)

        return final_result_attrs, final_result

    except (FileNotFoundError, OSError):
        raise Exception("Error accessing .hdf5 file")


def get_frame_data(
    hdf_path: pathlib.Path,
    dataset_name: str,
    step_name: str,
    frame_name: str,
    ) ->  DataFrameType:
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
            hdf_file[dataset_name][step_name][frame_name][:], 
            ).sort_values("Time")
