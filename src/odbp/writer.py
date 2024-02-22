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
import warnings
import numpy as np
from .types import (
    NodeType,
    NDArrayType,
    NPZFileType,
    H5PYFileType,
)


def convert_npz_to_h5(
    h5_path: pathlib.Path,
    npz_dir: pathlib.Path = pathlib.Path("tmp_npz"),
    temp_low: float | None = None,
    temp_high: float | None = None,
    time_step: int = 1,
    nodesets: list[str] | None = None,
    nodes: NodeType | None = None,
    parts: list[str] | None = None,
    steps: list[str] | None = None,
    coord_key: str = "COORD",
    target_outputs: list[str] | None = None,
    output_mapping: dict | None = None,
    odb_path: pathlib.Path | None = None,
) -> None:
    # Format of the npz_dir:
    # node_coords.npz (locations per node)
    # step_frame_times/<step>.npz (times per step)
    # temps/<step>/<frame>.npz (temperatures per node per frame)
    # All of these must exist (error if they do not)
    # They're the only things we care about

    h5_path = pathlib.Path(h5_path)
    npz_dir = pathlib.Path(npz_dir)

    data_dir: pathlib.Path = npz_dir / pathlib.Path("data")

    data: dict[str, dict[int, dict[str, NDArrayType | dict[str, NDArrayType]]]] = {}
    step_dir: pathlib.Path
    for step_dir in data_dir.iterdir():
        step_key: str = step_dir.stem
        data[step_key] = {}
        file: pathlib.Path
        for file in step_dir.iterdir():
            data_parts: list[str] = file.stem.split("_")
            data_type: str = data_parts.pop(0)
            frame_str: str = data_parts.pop(-1)
            component_label: str | None = None
            frame_val: int = int(frame_str)
            if frame_val not in data[step_key]:
                data[step_key][frame_val] = {}
            if len(data_parts) > 0:
                # TODO What if this length is > 1
                if len(data_parts) != 1:
                    raise RuntimeError("NOT SURE HOW TO FIX THIS YET!!!")
                else:
                    component_label = data_parts[0]

            data_file: NPZFileType
            if component_label is not None:
                if data_type not in data[step_key][frame_val].keys():
                    data[step_key][frame_val][data_type] = {}

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning, append=True)
                    with np.load(file) as data_file:
                        data[step_key][frame_val][data_type][component_label] = (
                            np.vstack(data_file[data_file.files[0]])
                        )

            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning, append=True)
                    with np.load(file) as data_file:
                        data[step_key][frame_val][data_type] = np.vstack(
                            data_file[data_file.files[0]]
                        )

    step: str
    step_dict: dict[int, dict[str, NDArrayType | dict[str, NDArrayType]]]
    for step, step_dict in data.items():
        frame: int
        frame_dict: dict[str, NDArrayType | dict[str, NDArrayType]]
        for frame, frame_dict in step_dict.items():
            to_remove: list[str] = []
            to_add: list[dict[str, NDArrayType]] = []
            output: str
            output_obj: NDArrayType | dict[str, NDArrayType]
            for output, output_obj in frame_dict.items():
                if isinstance(output_obj, dict):
                    to_remove.append(output)
                    spec_output: str
                    spec_output_obj: NDArrayType
                    for spec_output, spec_output_obj in output_obj.items():
                        if not isinstance(spec_output_obj, np.ndarray):
                            raise RuntimeError("NOT SURE HOW TO FIX THIS YET!!!")
                        to_add.append({spec_output: spec_output_obj})

            add_dict: dict[str, NDArrayType]
            for add_dict in to_add:
                k: str
                v: NDArrayType
                for k, v in add_dict.items():
                    frame_dict[k] = v

            remove_key: str
            for remove_key in to_remove:
                del frame_dict[remove_key]

    if output_mapping is None:
        output_mapping = {}

    hdf5_file: H5PYFileType
    with h5py.File(h5_path, "w") as hdf5_file:
        total_name: str = h5_path.stem
        data_step: str
        for data_step in data:
            data_frame_dict: dict[int, dict[str, NDArrayType]]
            for data_frame_dict in data[data_step].items():
                data_frame: int
                data_type_dict: dict[str, NDArrayType]
                data_frame, data_type_dict = data_frame_dict

                frame_time: NDArrayType = data_type_dict.pop("Time")
                target_len: int = len(list(data_type_dict.values())[0])
                column_headers: list[str] = [
                    "Node Label",
                    "Time",
                ]
                column_headers += list(data_type_dict.keys())
                column_headers = [output_mapping.get(c, c) for c in column_headers]
                column_dtypes: np.dtype = np.dtype(
                    {
                        "names": column_headers,
                        "formats": [np.float64 for _ in column_headers],
                    }
                )
                total_data: NDArrayType = np.hstack(
                    (
                        np.vstack(np.arange(1, target_len + 1, 1)),
                        np.vstack(np.full((target_len), frame_time)),
                        *list(data_type_dict.values())
                    )
                )
                total_rec: np.rec = np.rec.fromarrays(total_data.T, dtype=column_dtypes)
                hdf5_file.create_dataset(
                    f"{total_name}/{data_step}/{data_frame}",
                    data=total_rec,
                    compression="gzip",
                    compression_opts=9,
                )

        # TODO!!!
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