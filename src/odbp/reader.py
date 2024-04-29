"""
ODBP npz_to_hdf.py
ODBP
https://www.github.com/Computational-Mechanics-Materials-Lab/ODBP
MIT License (c) 2023

This file provides tools to read .hdf5 files created by the ODBP
converter into pandas dataframes

Originally written by CMML Member CJ Nguyen
"""

import h5py
import multiprocessing
import pathlib
import pandas as pd
import numpy.typing as npt
from dataclasses import dataclass

from .data_model import (
    element_connectivity_name,
    element_sets_to_elements_name,
    node_sets_to_nodes_name,
    part_to_element_set_name,
    part_to_node_set_name,
)


@dataclass
class OdbpData:
    element_data: pd.DataFrame
    node_data: pd.DataFrame
    element_connectivity: npt.NDArray
    element_sets_to_elements: dict[str, npt.NDArray]
    node_sets_to_nodes: dict[str, npt.NDArray]
    part_to_element_set: dict[str, npt.NDArray]
    part_to_node_set: dict[str, npt.NDArray]
    remaining_nodes_not_in_elements: pd.DataFrame | None = None


def get_odb_data(
    h5_path: pathlib.Path,
    dataset_name: str,
    target_dataset: str,
) -> OdbpData:
    hdf5_file: h5py.File
    with h5py.File(h5_path, "r") as hdf5_file:
        attr_data: dict[str, npt.NDArray] = {}
        key: str
        for key in hdf5_file[dataset_name][target_dataset].keys():
            attr_data[key] = hdf5_file[dataset_name][target_dataset][
                key
            ][:]

    return attr_data


def get_h5_data(h5_path: pathlib.Path, cpus: int, output_mapping: dict[str, str]) -> tuple[dict[str, str], OdbpData]:
    """
    get_node_coords(h5_path: pathlib.Path, cpus: int) -> pd.DataFrame
    return a data frame with nodes by integer index and floating point
    3-dimensional coordinates.
    """

    #final_odb_data: OdbpData = OdbpData()
    try:
        hdf5_file: h5py.File
        with h5py.File(h5_path, "r") as hdf5_file:
            dataset_name: str = list(hdf5_file.keys())[0]  # should only be 1 entry
            final_result_attrs: dict[str, str] = dict(hdf5_file[dataset_name].attrs)

            data_keys: list[str] = [
                element_connectivity_name,
                element_sets_to_elements_name,
                node_sets_to_nodes_name,
                part_to_element_set_name,
                part_to_node_set_name,
            ]

            element_connectivity: npt.NDArray = hdf5_file[dataset_name][
                element_connectivity_name
            ][:]

            target_data_key: str
            target_data_keys: list[str] = data_keys[:]
            target_data_keys.remove(element_connectivity_name)
            target_data: list[dict[str, npt.NDArray]] = []
            for target_data_key in target_data_keys:
                target_data.append(get_odb_data(
                    h5_path,
                    dataset_name,
                    target_data_key,
                ))

            element_sets_to_elements: dict[str, npt.NDArray]
            node_sets_to_nodes: dict[str, npt.NDArray]
            part_to_element_set: dict[str, npt.NDArray]
            part_to_node_set: dict[str, npt.NDArray]

            element_sets_to_elements, node_sets_to_nodes, part_to_element_set, part_to_node_set = target_data

            k: str
            steps_keys: list[str] = [
                k for k in hdf5_file[dataset_name].keys() if k not in data_keys
            ]
            result_dfs: dict[str, list[pd.DataFrame]] = {}

            step_key: str
            for step_key in steps_keys:
                data_type_keys: list[str] = list(
                    hdf5_file[dataset_name][step_key].keys()
                )
                data_type_key: str
                for data_type_key in data_type_keys:
                    if data_type_key not in result_dfs:
                        result_dfs[data_type_key] = []
                    frame_keys: list[str] = list(
                        hdf5_file[dataset_name][step_key][data_type_key].keys()
                    )
                    frame_key: str
                    args_list: list[tuple[pathlib.Path, str, str, str, str]] = [
                        (h5_path, dataset_name, step_key, data_type_key, frame_key)
                        for frame_key in frame_keys
                    ]

                    pool: multiprocessing.pool.Pool
                    with multiprocessing.Pool(processes=cpus) as pool:
                        results: list[pd.DataFrame] = pool.starmap(
                            get_frame_data, args_list
                        )
                    result_dfs[data_type_key].append(pd.concat(results))

        node_data: pd.DataFrame
        element_data = pd.DataFrame
        quantity_key: str
        time_key: str = output_mapping.get("Time", "Time")
        node_label_key: str = output_mapping.get("Node Label", "Node Label")
        elem_label_key: str = output_mapping.get("Element Label", "Element Label")
        for quantity_key in result_dfs.keys():
            match quantity_key:
                case "Nodal":
                    node_data = pd.concat(
                        result_dfs[quantity_key]
                    ).sort_values([time_key, node_label_key], ascending=True)

                case "Elemental":
                    element_data = pd.concat(
                        result_dfs[quantity_key]
                    ).sort_values([time_key, elem_label_key], ascending=True)

                case _:
                    raise Exception(f"ODB Quantity {quantity_key} was not found!")

        final_odb_data: OdbpData = OdbpData(
            element_data=element_data,
            node_data=node_data,
            element_connectivity=element_connectivity,
            element_sets_to_elements=element_sets_to_elements,
            node_sets_to_nodes=node_sets_to_nodes,
            part_to_element_set=part_to_element_set,
            part_to_node_set=part_to_node_set,
        )

        return final_result_attrs, final_odb_data

    except (FileNotFoundError, OSError) as err:
        print(err)
        raise Exception("Error accessing .hdf5 file") from err


def get_frame_data(
    h5_path: pathlib.Path,
    dataset_name: str,
    step_name: str,
    data_type_name: str,
    frame_name: str,
) -> pd.DataFrame:

    hdf5_file: h5py.File
    with h5py.File(h5_path, "r") as hdf5_file:
        return pd.DataFrame(
            hdf5_file[dataset_name][step_name][data_type_name][frame_name][:]
        )
