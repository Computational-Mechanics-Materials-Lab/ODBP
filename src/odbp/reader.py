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
from .types import DataFrameType, H5PYFileType, MultiprocessingPoolType


def get_h5_data(
    h5_path: pathlib.Path, cpus: int
) -> tuple[dict[str, str], DataFrameType]:
    """
    get_node_coords(h5_path: pathlib.Path) -> DataFrameType
    return a data frame with nodes by integer index and floating point
    3-dimensional coordinates.
    """

    try:
        hdf5_file: H5PYFileType
        with h5py.File(h5_path, "r") as hdf5_file:
            dataset_name: str = list(hdf5_file.keys())[0]  # should only be 1 entry
            final_result_attrs: dict[str, str] = dict(hdf5_file[dataset_name].attrs)
            steps_keys: list[str] = list(hdf5_file[dataset_name].keys())
            result_dfs: list[DataFrameType] = []

            step_key: str
            for step_key in steps_keys:
                frame_keys: list[str] = list(hdf5_file[dataset_name][step_key].keys())
                frame_key: str
                args_list: list[tuple[pathlib.Path, str, str, str]] = [
                    (h5_path, dataset_name, step_key, frame_key)
                    for frame_key in frame_keys
                ]

                pool: MultiprocessingPoolType
                with multiprocessing.Pool(processes=cpus) as pool:
                    results: list[DataFrameType] = pool.starmap(
                        get_frame_data, args_list
                    )
                result_dfs.append(pd.concat(results))

        final_result: DataFrameType = pd.concat(result_dfs).sort_values(
            ["Time", "Node Label"], ascending=True
        )

        return final_result_attrs, final_result

    except (FileNotFoundError, OSError) as err:
        raise Exception("Error accessing .hdf5 file") from err


def get_frame_data(
    h5_path: pathlib.Path,
    dataset_name: str,
    step_name: str,
    frame_name: str,
) -> DataFrameType:

    hdf5_file: H5PYFileType
    with h5py.File(h5_path, "r") as hdf5_file:
        return pd.DataFrame(hdf5_file[dataset_name][step_name][frame_name][:])
