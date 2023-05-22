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
import numpy as np
import pandas as pd
from os import PathLike
from .util import NDArrayType, DataFrameType, H5PYGroupType, H5PYFileType


def get_node_coords(hdf_path: PathLike) -> DataFrameType:
    """
    get_node_coords(hdf_path: PathLike) -> DataFrameType
    return a data frame with nodes by integer index and floating point
    3-dimensional coordinates.
    """

    try:
        hdf_file: H5PYFileType
        with h5py.File(hdf_path) as hdf_file:
            coords: DataFrameType = hdf_file["node_coords"][:]
    
    except (FileNotFoundError, OSError):
        raise "Error accessing .hdf5 file"

    except KeyError:
        raise f"{hdf_path} file does not include node coordinates or they" \
            "are not keyed by 'node_coords'"
    
    else:
        return pd.DataFrame(
            data=coords,
            columns=["Node Labels", "X", "Y", "Z"]
        ).astype({"Node Labels": int})


def get_node_times_temps(
    hdf_path: PathLike,
    node: int,
    frame_sample: int, # TODO frame_ind
    x: float,
    y: float,
    z: float
    ) -> DataFrameType:
    """
    get_node_times_temps(
        hdf_path: PathLike,
        node: int,
        frame_sample: int
        x: float,
        y: float,
        z: float
    ) -> DataFrameType
    Given a path to a .hdf5 file, the label and coodinates of a node, return
    a dataframe associating the node with its temperatures per frame.

    0.5.0 Currently this still uses the frame_sample value for stepped frames,
    but this will be updated.
    """

    hdf_file: H5PYFileType
    with h5py.File(hdf_path) as hdf_file:
        temp_steps: H5PYGroupType = hdf_file["temps"]
        time_steps: H5PYGroupType = hdf_file["step_frame_times"]
        target_len: int = len(temp_steps[list(temp_steps.keys())[0]])
        temps: NDArrayType = np.zeros(target_len)
        times: NDArrayType = np.zeros(target_len)
        xs: NDArrayType = np.full(target_len, x)
        ys: NDArrayType = np.full(target_len, y)
        zs: NDArrayType = np.full(target_len, z)

        step: str
        for step in temp_steps:
            ind: int
            frame: str
            for ind, frame in enumerate(temp_steps[step]):
                temps[ind] = temp_steps[step][frame][node]
                times[ind] = time_steps[step][int(
                    frame.replace("frame_", "")
                    ) // frame_sample
                    ] # TODO frame_ind
    
    return pd.DataFrame(
        data=np.vstack(
            (temps, times, xs, ys, zs), casting="no"
            ).T, 
        columns=["Temp", "Time", "X", "Y", "Z"]
        ).sort_values("Time")