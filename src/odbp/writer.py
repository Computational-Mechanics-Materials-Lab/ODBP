#!/usr/bin/env python3

"""
ODBPlotter npz_to_hdf.py
ODBPlotter
https://www.github.com/Computational-Mechanics-Materials-Lab/ODBPlotter
MIT License (c) 2023

This file exposes the npz_to_hdf() method, used to translate a hierarchical
directory of .npz files into a .hdf5 file.

The inputs must be:
    A Pathlike for the directory of the source files
    A Pathlike of the resulting output .hdf5 file

Originally written by CMML Member CJ Nguyen
"""


import h5py
import pathlib
import os
import warnings
import numpy as np
from typing import Dict, List, Optional
from .types import (
    NDArrayType,
    NPZFileType,
    H5PYFileType,
    NullableNodeType,
    NullableStrList,
)


def convert_npz_to_hdf(
    hdf_path: pathlib.Path,
    #data_model: str,
    npz_dir: pathlib.Path = pathlib.Path("tmp_npz"),
    temp_low: "Optional[float]" = None,
    temp_high: "Optional[float]" = None,
    time_step: int = 1,
    nodesets: NullableStrList = None,
    nodes: NullableNodeType = None,
    parts: NullableStrList = None,
    steps: NullableStrList = None,
    coord_key: str = "COORD",
    target_outputs: NullableStrList = None,
    odb_path: "Optional[str]" = None,
) -> None:
    # Format of the npz_dir:
    # node_coords.npz (locations per node)
    # step_frame_times/<step>.npz (times per step)
    # temps/<step>/<frame>.npz (temperatures per node per frame)
    # All of these must exist (error if they do not)
    # They're the only things we care about

    hdf_path = pathlib.Path(hdf_path)
    npz_dir = pathlib.Path(npz_dir)

    coordinate_data: Optional[NDArrayType] = None
    #if data_model != "mechanical":
    #step_frame_times_dir: pathlib.Path = npz_dir / "step_frame_times"
    #step_frame_times: Dict[str, NDArrayType] = dict()
    #for file in step_frame_times_dir.iterdir():
    #    key: str = str(file.stem)
    #    with warnings.catch_warnings():
    #        warnings.filterwarnings("ignore", category=UserWarning, append=True)
    #        step_frame_times_file: NPZFileType
    #        with np.load(file) as step_frame_times_file:
    #            time_data: NDArrayType = step_frame_times_file[
    #                step_frame_times_file.files[0]
    #            ]

    #    step_frame_times[key] = time_data
    node_coords_path: pathlib.Path = npz_dir / pathlib.Path("node_coords.npz")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, append=True)
        node_coords_file: NPZFileType
        with np.load(node_coords_path) as node_coords_file:
            coordinate_data = node_coords_file[node_coords_file.files[0]]

    data_dir: pathlib.Path = npz_dir / pathlib.Path("data")

    data: Dict[str, Dict[int, Dict[str, NDArrayType]]] = dict()
    step: pathlib.Path
    for step in data_dir.iterdir():
        step_key = str(step.stem)
        data[step_key] = dict()
        file: pathlib.Path
        for file in step.iterdir():
            data_type: str
            frame_str: str
            data_type, frame_str = file.stem.split("_")
            frame: int = int(frame_str)
            if frame not in data[step_key]:
                data[step_key][frame] = dict()

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, append=1)
                data_file: NPZFileType
                with np.load(file) as data_file:
                    data[step_key][frame][data_type] = np.vstack(
                        data_file[data_file.files[0]]
                    )

    hdf5_file: H5PYFileType
    with h5py.File(hdf_path, "w") as hdf5_file:
        total_name: str = str(hdf_path.stem)
        step: str
        for step in data:
            i: int
            frame_dict: Dict[int, Dict[str, NDArrayType]]
            for i, frame_dict in enumerate(data[step].items()):
                frame: int
                data_type_dict: Dict[str, NDArrayType]
                frame, data_type_dict = frame_dict

                frame_time = data_type_dict.pop("Time")

                #if data_model == "thermal":
                target_len: int = len(list(data_type_dict.values())[0])
                column_headers: List[str] = [
                    "Time",
                ]
                column_headers += list(data_type_dict.keys())
                # Preserve order
                column_headers = list(dict.fromkeys(column_headers))
                column_dtypes: np.dtype = np.dtype(
                    {
                        "names": column_headers,
                        "formats": [np.float64 for _ in column_headers],
                    }
                )

                total_data: NDArrayType = np.hstack(
                    (
                        np.vstack(np.full((target_len), frame_time)),
                        *list(data_type_dict.values()),
                    )
                )
                total_rec: np.record = np.rec.fromarrays(
                    total_data.T, dtype=column_dtypes
                )
                hdf5_file.create_dataset(
                    f"{total_name}/{step}/{frame}",
                    data=total_rec,
                    compression="gzip",
                )

                #elif data_model == "mechanical":
                #    target_len: int = len(list(data_type_dict.values())[0])
                #    column_headers: List[str] = ["Time"]
                #    column_headers += list(data_type_dict.keys())
                #    column_headers += ["Node Label", "X", "Y", "Z"]
                #    # Preserve order
                #    column_headers = list(dict.fromkeys(column_headers))
                #    column_dtypes: np.dtype = np.dtype(
                #        {
                #            "names": column_headers,
                #            "formats": [np.float64 for _ in column_headers],
                #        }
                #    )

                #    total_data: NDArrayType = np.hstack(
                #        (
                #            np.vstack(np.full((target_len), step_frame_times[step][i])),
                #            *list(data_type_dict.values()),
                #        )
                #    )
                #    total_rec: np.record = np.rec.fromarrays(
                #        total_data.T, dtype=column_dtypes
                #    )
                #    hdf5_file.create_dataset(
                #        f"{total_name}/{step}/{frame}",
                #        data=total_rec,
                #        compression="gzip",
                #    )

        #if data_model == "thermal" and coordinate_data is not None:
        if coordinate_data is not None:
            coordinate_dtypes: np.dtype = np.dtype(
                {
                    "names": ["Node Label", "X", "Y", "Z"],
                    "formats": [np.float64 for _ in range(coordinate_data.shape[1])],
                }
            )
            coord_rec: np.record = np.rec.fromarrays(
                coordinate_data.T, dtype=coordinate_dtypes
            )
            hdf5_file.create_dataset(
                f"{total_name}/coordinates", data=coord_rec, compression="gzip"
            )

        #hdf5_file[total_name].attrs["data_model"] = data_model
        if temp_low is not None:
            hdf5_file[total_name].attrs["temp_low"] = temp_low
        if temp_high is not None:
            hdf5_file[total_name].attrs["temp_high"] = temp_high
        if time_step is not None:
            hdf5_file[total_name].attrs["time_step"] = time_step
        if nodesets is not None:
            hdf5_file[total_name].attrs["nodesets"] = nodesets
        else:
            hdf5_file[total_name].attrs["nodesets"] = "All Nodesets"
        if nodes is not None:
            hdf5_file[total_name].attrs["nodes"] = nodes
        else:
            hdf5_file[total_name].attrs["nodes"] = "All Nodes"
        if parts is not None:
            hdf5_file[total_name].attrs["parts"] = parts
        else:
            hdf5_file[total_name].attrs["parts"] = "All Parts"
        if steps is not None:
            hdf5_file[total_name].attrs["steps"] = steps
        else:
            hdf5_file[total_name].attrs["steps"] = "All Steps"
        hdf5_file[total_name].attrs["coord_key"] = coord_key
        if target_outputs is not None:
            hdf5_file[total_name].attrs["target_outputs"] = target_outputs
        else:
            hdf5_file[total_name].attrs["target_outputs"] = "All Fields"
        if odb_path is not None:
            hdf5_file[total_name].attrs["odb_path"] = str(odb_path)
        else:
            hdf5_file[total_name].attrs["odb_path"] = "Unknown"
