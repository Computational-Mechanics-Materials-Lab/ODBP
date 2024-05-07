#!/usr/bin/env python3

"""
ODBP npz_to_hdf.py
ODBP
https://www.github.com/Computational-Mechanics-Materials-Lab/ODBP
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
import pickle
import numpy as np
import numpy.typing as npt
from typing import Any, Literal, assert_never
from io import BufferedReader

from .data_model import (
    element_connectivity_name,
    element_sets_to_elements_name,
    node_sets_to_nodes_name,
    part_to_element_set_name,
    part_to_node_set_name,
)


def convert_npz_to_h5(
    h5_path: pathlib.Path,
    npz_dir: pathlib.Path,
    output_mapping: dict[str, str],
) -> None:
    h5_path = pathlib.Path(h5_path)
    npz_dir = pathlib.Path(npz_dir)

    data_dir: pathlib.Path = npz_dir / pathlib.Path("data")

    # ew
    data: dict[
        str,
        dict[
            Literal["Nodal", "Elemental"],
            dict[int, dict[str, npt.NDArray | dict[str, npt.NDArray]]],
        ],
    ] = {}

    element_connectivity: npt.NDArray
    node_coords: dict[str, npt.NDArray]
    elementsets_to_elements_mapping: dict[str, npt.NDArray] = {}
    nodesets_to_nodes_mapping: dict[str, npt.NDArray] = {}
    part_to_elementsets_mapping: dict[str, npt.NDArray] = {}
    part_to_nodesets_mapping: dict[str, npt.NDArray] = {}

    step_dir_or_input_pickle_or_npz: pathlib.Path
    for step_dir_or_input_pickle_or_npz in data_dir.iterdir():
        if step_dir_or_input_pickle_or_npz.is_dir():
            step_dir: pathlib.Path = step_dir_or_input_pickle_or_npz
            step_key: str = step_dir.stem
            data[step_key] = {}
            file: pathlib.Path
            for file in step_dir.iterdir():
                # location_type_..._frame, ... may be nothing or may be a component label
                data_parts: list[str] = file.stem.split("_")
                data_location: str = data_parts.pop(0)
                if data_location not in data[step_key].keys():
                    data[step_key][data_location] = {}
                data_type: str = data_parts.pop(0)
                frame_str: str = data_parts.pop(-1)
                component_label: str | None = None
                frame_val: int = int(frame_str)
                if frame_val not in data[step_key][data_location].keys():
                    data[step_key][data_location][frame_val] = {}
                if len(data_parts) > 0:
                    # This should always be a 1-element list if it's not empty
                    component_label = data_parts[0]

                data_file: Any
                if component_label is not None:
                    if data_type not in data[step_key][data_location][frame_val].keys():
                        data[step_key][data_location][frame_val][data_type] = {}

                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore", category=UserWarning, append=True
                        )
                        with np.load(file) as data_file:
                            data[step_key][data_location][frame_val][data_type][
                                component_label
                            ] = np.vstack(data_file[data_file.files[0]])

                else:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore", category=UserWarning, append=True
                        )
                        with np.load(file) as data_file:
                            data[step_key][data_location][frame_val][data_type] = (
                                np.vstack(data_file[data_file.files[0]])
                            )

        else:
            input_pickle_or_npz: pathlib.Path = step_dir_or_input_pickle_or_npz
            if input_pickle_or_npz.suffix == ".pickle":
                input_pickle: pathlib.Path = input_pickle_or_npz
                pf: BufferedReader
                match input_pickle.stem:
                    case "elementsets_to_elements_mapping":
                        with open(input_pickle, "rb") as pf:
                            elementsets_to_elements_mapping = pickle.load(pf)

                    case "nodesets_to_nodes_mapping":
                        with open(input_pickle, "rb") as pf:
                            nodesets_to_nodes_mapping = pickle.load(pf)

                    case "part_to_elementsets_mapping":
                        with open(input_pickle, "rb") as pf:
                            part_to_elementsets_mapping = pickle.load(pf)

                    case "part_to_nodesets_mapping":
                        with open(input_pickle, "rb") as pf:
                            part_to_nodesets_mapping = pickle.load(pf)

                    case _:
                        raise Exception(f"Unknown pickle file: {input_pickle}")

            elif input_pickle_or_npz.suffix == ".npz":
                npz_file: pathlib.Path = input_pickle_or_npz
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning, append=True)
                    match npz_file.stem:
                        case "node_coords":
                            with np.load(npz_file) as data_file:
                                node_coords_data: npt.NDArray = data_file[
                                    data_file.files[0]
                                ]
                                data_len: int = node_coords_data.shape[0]
                                node_coords = {
                                    "X": node_coords_data[:, 0].reshape((data_len, 1)),
                                    "Y": node_coords_data[:, 1].reshape((data_len, 1)),
                                    "Z": node_coords_data[:, 2].reshape((data_len, 1)),
                                }

                        case "element_connectivity":
                            with np.load(npz_file) as data_file:
                                element_connectivity: npt.NDArray = data_file[data_file.files[0]]

                        case _:
                            raise Exception(f"Unknown npz file: {npz_file}")

            else:
                raise ValueError(f"Unexpected file: {input_pickle_or_npz}")

    step_dict: dict[str, dict[int, dict[str, npt.NDArray | dict[str, npt.NDArray]]]]
    for step_dict in data.values():
        location_dict: dict[int, dict[str, npt.NDArray | dict[str, npt.NDArray]]]
        for location_dict in step_dict.values():
            frame_dict: dict[str, npt.NDArray | dict[str, npt.NDArray]]
            for frame_dict in location_dict.values():
                to_remove: list[str] = []
                to_add: list[dict[str, npt.NDArray]] = []
                output: str
                output_obj: npt.NDArray | dict[str, npt.NDArray]
                for output, output_obj in frame_dict.items():
                    if isinstance(output_obj, dict):
                        to_remove.append(output)
                        spec_output: str
                        spec_output_obj: npt.NDArray
                        for spec_output, spec_output_obj in output_obj.items():
                            to_add.append({spec_output: spec_output_obj})

                add_dict: dict[str, npt.NDArray]
                for add_dict in to_add:
                    add_k: str
                    add_v: npt.NDArray
                    for add_k, add_v in add_dict.items():
                        frame_dict[add_k] = add_v

                remove_key: str
                for remove_key in to_remove:
                    del frame_dict[remove_key]

    if output_mapping is None:
        output_mapping = {}

    hdf5_file: h5py.File
    with h5py.File(h5_path, "w") as hdf5_file:
        total_name: str = h5_path.stem
        hdf5_file.create_dataset(
            f"{total_name}/{element_connectivity_name}",
            data=element_connectivity,
            compression="gzip",
            compression_opts=9,
        )

        name: str
        mapping: dict[str, npt.NDArray]
        for name, mapping in (
            (
                element_sets_to_elements_name,
                elementsets_to_elements_mapping,
            ),
            (
                node_sets_to_nodes_name,
                nodesets_to_nodes_mapping,
            ),
            (
                part_to_element_set_name,
                part_to_elementsets_mapping,
            ),
            (
                part_to_node_set_name,
                part_to_nodesets_mapping,
            ),
        ):
            k: str
            v: npt.NDArray
            for k, v in mapping.items():
                hdf5_file.create_dataset(
                    f"{total_name}/{name}/{k}",
                    data=v,
                    compression="gzip",
                    compression_opts=9,
                )

        data_step: str
        for data_step in data:
            data_component: Literal["Nodal", "Elemental"]
            data_frame_dict: dict[int, dict[str, npt.NDArray | dict[str, npt.NDArray]]]
            for data_component, data_frame_dict in data[data_step].items():
                data_frame: int
                data_type_dict: dict[str, npt.NDArray | dict[str, npt.NDArray]]
                for data_frame, data_type_dict in data_frame_dict.items():

                    frame_time: npt.NDArray | dict[str, npt.NDArray] = (
                        data_type_dict.pop("Time")
                    )
                    if not isinstance(frame_time, np.ndarray):
                        raise ValueError("Time should map to only one array!")
                    target_len: int
                    if data_component == "Nodal":
                        target_len = len(node_coords["X"])
                    elif data_component == "Elemental":
                        target_len = len(element_connectivity)

                    column_headers: list[str]
                    total_data: npt.NDArray
                    total_rec: np.recarray
                    column_dtypes: np.dtype
                    if data_component == "Elemental":
                        column_headers = [output_mapping.get("Element Label", "Element Label")]
                        column_headers.append("Time")
                        column_headers += list(data_type_dict.keys())
                        column_headers = [
                            output_mapping.get(c, c) for c in column_headers
                        ]
                        column_dtypes = np.dtype(
                            {
                                "names": column_headers,
                                "formats": [np.float64 for _ in column_headers],
                            }
                        )

                        total_data = np.hstack(
                            (
                                np.vstack(np.arange(1, target_len + 1, 1)),
                                np.vstack(np.full((target_len), frame_time)),
                                *list(data_type_dict.values()),
                            )
                        )

                        total_rec = np.rec.fromarrays(total_data.T, dtype=column_dtypes)
                        hdf5_file.create_dataset(
                            f"{total_name}/{data_step}/{data_component}/{data_frame}",
                            data=total_rec,
                            compression="gzip",
                            compression_opts=9,
                        )
                    elif data_component == "Nodal":
                        data_type_dict |= node_coords
                        column_headers = [output_mapping.get("Node Label", "Node Label")]
                        column_headers.append("Time")
                        column_headers += list(data_type_dict.keys())
                        column_headers = [
                            output_mapping.get(c, c) for c in column_headers
                        ]
                        column_dtypes = np.dtype(
                            {
                                "names": column_headers,
                                "formats": [np.float64 for _ in column_headers],
                            }
                        )
                        total_data = np.hstack(
                            (
                                np.vstack(np.arange(1, target_len + 1, 1)),
                                np.vstack(np.full((target_len), frame_time)),
                                *list(data_type_dict.values()),
                            )
                        )
                        total_rec = np.rec.fromarrays(total_data.T, dtype=column_dtypes)
                        hdf5_file.create_dataset(
                            f"{total_name}/{data_step}/{data_component}/{data_frame}",
                            data=total_rec,
                            compression="gzip",
                            compression_opts=9,
                        )
                    else:
                        assert_never(data_component)

        # TODO!!!
        # if temp_low is not None:
        #    hdf5_file[total_name].attrs["temp_low"] = temp_low
        # if temp_high is not None:
        #    hdf5_file[total_name].attrs["temp_high"] = temp_high
        # if nodesets is not None:
        #    hdf5_file[total_name].attrs["nodesets"] = nodesets
        # else:
        #    hdf5_file[total_name].attrs["nodesets"] = "All Nodesets"
        # if nodes is not None:
        #    hdf5_file[total_name].attrs["nodes"] = nodes
        # else:
        #    hdf5_file[total_name].attrs["nodes"] = "All Nodes"
        # if parts is not None:
        #    hdf5_file[total_name].attrs["parts"] = parts
        # else:
        #    hdf5_file[total_name].attrs["parts"] = "All Parts"
        # if steps is not None:
        #    hdf5_file[total_name].attrs["steps"] = steps
        # else:
        #    hdf5_file[total_name].attrs["steps"] = "All Steps"
        # if odb_path is not None:
        #    hdf5_file[total_name].attrs["odb_path"] = str(odb_path)
        # else:
        #    hdf5_file[total_name].attrs["odb_path"] = "Unknown"
