#!/usr/bin/env python3

"""
ODBP odb.py
ODBP
https://www.github.com/Computational-Mechanics-Materials-Lab/ODBP
MIT License (c) 2023

Implement the main ODB class, which is used to process the .hdf5 data and
filter it by desired values.
"""

import subprocess
import shutil
import pathlib
import pickle
import operator

# BOTH TODO
#import h5py
#import multiprocessing

import numpy as np
import numpy.typing as npt
import pandas as pd
import polyscope as ps

from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from typing import Any, Iterator, BinaryIO, Self, Callable, assert_never
from io import BufferedReader

from .odbp_settings import OdbpSettings, OdbpOutputs, OdbpPlotType, OdbpOutput
from .writer import convert_npz_to_h5
from .reader import get_h5_data, OdbpData


class Odbp(OdbpSettings):
    """
    Stores Data from a .hdf5, implements extractor methods
    to transfer from .odb to .hdf5
    Implements abilities to resize the dimenisons or timeframe of the data
    """

    __slots__ = (
        # file-defined components
        "_data_handler",
        "_data",
        "_py2_scripts_path",
        "_convert_script_path",
        "_convert_pickle_path",
        "_convert_result_path",
        "_extract_script_path",
        "_extract_pickle_path",
        "_extract_result_path",
        "_get_odb_info_script_path",
        "_get_odb_info_result_path",
        "_iterator_ind",
        "_iterable_cols_and_vals",
        "_iterator_key",
        "_data_for_iteration"
        "_frame_keys",
        "_frame_keys_per_step",
        "_frame_range",
        "_step_names",
        "_step_lens",
        "_nodeset_names",
        "_elementset_names",
        "_part_names",
        "_node_range",
        "_element_range",
        "_nodes_per_part",
        "_nodes_per_nodeset",
        "_nodeset_per_part",
        "_elements_per_part",
        "_elements_per_elementset",
        "_elementset_per_part",
        "_h5_status",
    )

    def __init__(self: Self) -> None:
        """
        Type Hints and hard-coded parameters. See the @staticmethod
        "constructors" of this class in order to learn about initialization
        """

        super().__init__()

        self._data_handler: DataLoader | DataUnloader = DataLoader()
        self._data: OdbpData

        self._py2_scripts_path: pathlib.Path = pathlib.Path(
            pathlib.Path(__file__).parent, "py2_scripts"
        ).absolute()

        # TODO can be simpler
        # Hardcoded paths for Python 3 - 2 communication
        self._convert_script_path: pathlib.Path = (
            self._py2_scripts_path / "converter.py"
        )

        self._extract_script_path: pathlib.Path = (
            self._py2_scripts_path / "extractor.py"
        )

        self._get_odb_info_script_path: pathlib.Path = (
            self._py2_scripts_path / "odb_info_getter.py"
        )

        self._convert_pickle_path: pathlib.Path = pathlib.Path(
            pathlib.Path.cwd().absolute(), "convert.pickle"
        )

        self._extract_pickle_path: pathlib.Path = pathlib.Path(
            pathlib.Path.cwd().absolute(), "extract_from_odb.pickle"
        )

        self._convert_result_path: pathlib.Path = pathlib.Path(
            pathlib.Path.cwd().absolute(), "convert_result.pickle"
        )

        self._extract_result_path: pathlib.Path = pathlib.Path(
            pathlib.Path.cwd().absolute(), "extract_results.pickle"
        )

        self._get_odb_info_result_path: pathlib.Path = pathlib.Path(
            pathlib.Path.cwd(), "odb_info_result.pickle"
        ).absolute()

        # Type Hinting these was futile. TODO,
        #self._frame_keys: list[str]
        #self._frame_keys_per_step: dict[str, list[int]]
        #self._frame_range: int
        #self._step_names: list[str]
        #self._step_lens: dict[str, int]
        #self._nodeset_names: list[str]
        #self._elementset_names: list[str]
        #self._part_names: list[str]
        #self._node_range: int
        #self._element_range: int
        #self._nodes_per_part: dict[str, list[int]]
        #self._nodes_per_nodeset: dict[str, list[int]]
        #self._nodeset_per_part: dict[str, list[str]]
        #self._elements_per_part: dict[str, list[int]]
        #self._elements_per_elementset: dict[str, list[int]]
        #self._elementset_per_part: dict[str, list[str]]
        self._frame_keys: Any
        self._frame_keys_per_step: Any
        self._frame_range: Any
        self._step_names: Any
        self._step_lens: Any
        self._nodeset_names: Any
        self._elementset_names: Any
        self._part_names: Any
        self._node_range: Any
        self._element_range: Any
        self._nodes_per_part: Any
        self._nodes_per_nodeset: Any
        self._nodeset_per_part: Any
        self._elements_per_part: Any
        self._elements_per_elementset: Any
        self._elementset_per_part: Any

        self._iterable_cols_and_vals: dict[str, list[float]]
        self._iterator_key: str
        self._iterator_ind: int = 0
        self._data_for_iteration: npt.NDArray | None = None

        self._h5_status: dict[str, str]


    def __iter__(self: Self) -> Self:
        return self

    def __next__(self: Self) -> Iterator[OdbpData]:
        if not hasattr(self, "_data"):
            raise AttributeError(
                "Odbp() object only functions as an iterator"
                "After load_h5() has been called."
            )

        if self._data_for_iteration is None:
            data_for_iteration: npt.NDArray = np.copy(self._iterable_cols_and_vals[self._iterator_key])
            iterator_output: OdbpOutput = self.outputs.outputs_by_names[self._iterator_key]
            iterator_min_op: Callable
            iterator_max_op: Callable
            iterator_min_op, iterator_max_op = self._min_max_op_from_output_for_iter(iterator_output)

            # TODO Hacky
            data_for_iteration = data_for_iteration[iterator_min_op(data_for_iteration, iterator_output.bound_max)]
            data_for_iteration = data_for_iteration[iterator_max_op(data_for_iteration, iterator_output.bound_min)]
            self._data_for_iteration = data_for_iteration

        if self._iterator_ind >= len(self._data_for_iteration):
            self._iterator_ind = 0
            del self._data_for_iteration
            self._data_for_iteration = None
            raise StopIteration

        ind: int = self._iterator_ind
        self._iterator_ind += 1

        data_to_filter: OdbpData = OdbpData(
            element_data=self.data.element_data,
            node_data = self.data.node_data,
            element_connectivity=self.data.element_connectivity,
            element_sets_to_elements=self.data.element_sets_to_elements,
            node_sets_to_nodes=self.data.node_sets_to_nodes,
            part_to_element_set=self.data.part_to_element_set,
            part_to_node_set=self.data.part_to_node_set,
            remaining_nodes_not_in_elements=self.data.remaining_nodes_not_in_elements
        )

        data_to_filter.node_data = data_to_filter.node_data[data_to_filter.node_data[self._iterator_key] == self._data_for_iteration[ind]]
        data_to_filter.element_data = data_to_filter.element_data[data_to_filter.element_data[self._iterator_key] == self._data_for_iteration[ind]]

        return_data: OdbpData = self.filter_by_output_bounds(data_to_filter)

        return return_data


    def filter_by_output_bounds(self: Self, data_to_modify: OdbpData | None = None) -> OdbpData:
        if data_to_modify is None:
            # TODO this doesn't really work
            data_to_modify = self.data

        return_data: OdbpData = OdbpData(
            element_data=data_to_modify.element_data,
            node_data = data_to_modify.node_data,
            element_connectivity=data_to_modify.element_connectivity,
            element_sets_to_elements=data_to_modify.element_sets_to_elements,
            node_sets_to_nodes=data_to_modify.node_sets_to_nodes,
            part_to_element_set=data_to_modify.part_to_element_set,
            part_to_node_set=data_to_modify.part_to_node_set
        )

        nodes_to_remove: list[int] = []
        elems_to_remove: list[int] = []
        output: OdbpOutput
        node_label_key: list[str] = [self.output_mapping.get("Node Label", "Node Label")]
        element_label_key: list[str] = [self.output_mapping.get("Element Label", "Element Label")]
        for output in self.outputs.outputs:

            min_op: Callable
            max_op: Callable
            min_op, max_op = self._min_max_op_from_output(output)

            if output.name in return_data.node_data:
                nodes_to_remove.extend(return_data.node_data[min_op(return_data.node_data[output.name], output.bound_min)][node_label_key].astype(int).to_numpy() - 1)
                nodes_to_remove.extend(return_data.node_data[max_op(return_data.node_data[output.name], output.bound_max)][node_label_key].astype(int).to_numpy() - 1)

            if output.name in return_data.element_data:
                elems_to_remove.extend(return_data.element_data[min_op(return_data.element_data[output.name], output.bound_min)][element_label_key].astype(int).to_numpy() - 1)
                elems_to_remove.extend(return_data.element_data[max_op(return_data.element_data[output.name], output.bound_max)][element_label_key].astype(int).to_numpy() - 1)
  
        final_nodes_to_remove: npt.NDArray = np.unique(np.array(nodes_to_remove, dtype=np.int64))
        final_elems_to_remove: npt.NDArray = np.unique(np.array(elems_to_remove, dtype=np.int64))

        node_indices: npt.NDArray = np.copy(return_data.node_data[node_label_key].astype(int).to_numpy() - 1)
        updated_node_indices: npt.NDArray = np.delete(node_indices, final_nodes_to_remove, axis=0)

        elem_indices: npt.NDArray = np.copy(return_data.element_data[element_label_key].astype(int).to_numpy() - 1)
        updated_elem_indices: npt.NDArray = np.delete(elem_indices, final_elems_to_remove, axis=0)

        updated_elem_connectivity: npt.NDArray = return_data.element_connectivity[updated_elem_indices].reshape(-1, 8)
        elems_with_remaining_nodes_indices: npt.NDArray = np.isin(updated_elem_connectivity - 1, updated_node_indices).all(axis=1)
        updated_elem_indices = updated_elem_indices[elems_with_remaining_nodes_indices]
        updated_elem_connectivity = return_data.element_connectivity[updated_elem_indices].reshape(-1, 8)
        return_data.element_connectivity = updated_elem_connectivity

        if updated_node_indices.shape[0] > 0 and not node_indices.shape == updated_node_indices.shape:
            remaining_nodes_not_in_elems: pd.DataFrame = return_data.node_data.iloc[np.setdiff1d(updated_node_indices, np.unique(updated_elem_connectivity) - 1)]# - 1]
            return_data.remaining_nodes_not_in_elements = remaining_nodes_not_in_elems

        return return_data

    # TODO This is gross
    def _min_max_op_from_output_for_iter(self: Self, output: OdbpOutput) -> tuple[Callable, Callable]:
        min_op: Callable
        if output.bound_min_equal:
            if output.bounds_between_or_outside == output.ODBPOUTPUTBOUNDSRANGE.BETWEEN:
                min_op = operator.lt
            elif output.bounds_between_or_outside == output.ODBPOUTPUTBOUNDSRANGE.OUTSIDE:
                min_op = operator.gt
            else:
                assert_never(output.bounds_between_or_outside)

        else:
            if output.bounds_between_or_outside == output.ODBPOUTPUTBOUNDSRANGE.BETWEEN:
                min_op = operator.le
            elif output.bounds_between_or_outside == output.ODBPOUTPUTBOUNDSRANGE.OUTSIDE:
                min_op = operator.ge
            else:
                assert_never(output.bounds_between_or_outside)


        max_op: Callable
        if output.bound_max_equal:
            if output.bounds_between_or_outside == output.ODBPOUTPUTBOUNDSRANGE.BETWEEN:
                max_op = operator.gt
            elif output.bounds_between_or_outside == output.ODBPOUTPUTBOUNDSRANGE.OUTSIDE:
                max_op = operator.lt
            else:
                assert_never(output.bounds_between_or_outside)

        else:
            if output.bounds_between_or_outside == output.ODBPOUTPUTBOUNDSRANGE.BETWEEN:
                max_op = operator.ge
            elif output.bounds_between_or_outside == output.ODBPOUTPUTBOUNDSRANGE.OUTSIDE:
                max_op = operator.le
            else:
                assert_never(output.bounds_between_or_outside)

        return min_op, max_op
        
    def _min_max_op_from_output(self: Self, output: OdbpOutput) -> tuple[Callable, Callable]:
        min_op: Callable
        if output.bound_min_equal:
            if output.bounds_between_or_outside == output.ODBPOUTPUTBOUNDSRANGE.BETWEEN:
                # lower <= Foo
                min_op = operator.le
            elif output.bounds_between_or_outside == output.ODBPOUTPUTBOUNDSRANGE.OUTSIDE:
                # Foo <= lower --> lower >= Foo
                min_op = operator.ge
            else:
                assert_never(output.bounds_between_or_outside)

        else:
            if output.bounds_between_or_outside == output.ODBPOUTPUTBOUNDSRANGE.BETWEEN:
                # lower < Foo
                min_op = operator.lt
            elif output.bounds_between_or_outside == output.ODBPOUTPUTBOUNDSRANGE.OUTSIDE:
                # Foo < lower --> lower > Foo
                min_op = operator.gt
            else:
                assert_never(output.bounds_between_or_outside)


        max_op: Callable
        if output.bound_max_equal:
            if output.bounds_between_or_outside == output.ODBPOUTPUTBOUNDSRANGE.BETWEEN:
                # Foo <= upper --> upper >= Foo
                max_op = operator.ge
            elif output.bounds_between_or_outside == output.ODBPOUTPUTBOUNDSRANGE.OUTSIDE:
                # upper <= Foo
                max_op = operator.le
            else:
                assert_never(output.bounds_between_or_outside)

        else:
            if output.bounds_between_or_outside == output.ODBPOUTPUTBOUNDSRANGE.BETWEEN:
                # Foo < upper -> upper > Foo
                max_op = operator.gt
            elif output.bounds_between_or_outside == output.ODBPOUTPUTBOUNDSRANGE.OUTSIDE:
                # upper < Foo
                max_op = operator.lt
            else:
                assert_never(output.bounds_between_or_outside)

        return min_op, max_op

    @property
    def data(self: Self) -> OdbpData:
        return self._data

    @data.deleter
    def data(self: Self) -> None:
        del self._data

    # TODO!!!
    # @property
    # def frame_range(self) -> tuple[int, int]:
    #    return self._frame_range

    # @property
    # def frame_keys(self) -> list[str]:
    #    return self._frame_keys

    # @property
    # def step_names(self) -> list[str]:
    #    return self._step_names

    # @property
    # def step_lens(self) -> dict[str, int]:
    #    return self._step_lens

    # @property
    # def frame_keys_per_step(self) -> dict[str, list[str]]:
    #    return self._frame_keys_per_step

    # @property
    # def nodeset_names(self) -> list[str]:
    #    return self._nodeset_names

    # @property
    # def part_names(self) -> list[str]:
    #    return self._part_names

    # @property
    # def node_range(self) -> tuple[int, int]:
    #    return self._node_range

    # @property
    # def node_ranges_per_part(self) -> dict[str, tuple[int, int]]:
    #    return self._node_ranges_per_part

    @property
    def h5_status(self: Self) -> dict[str, str]:
        return self._h5_status

    def convert(
        self: Self,
        h5_path: pathlib.Path | None = None,
        *,
        odb_path: pathlib.Path | None = None,
        set_odb: bool = False,
        set_h5: bool = False,
    ) -> None:
        if odb_path is not None:
            odb_path = pathlib.Path(odb_path)
            if set_odb:
                self.odb_path = odb_path

        else:
            if not hasattr(self, "odb_path"):
                raise AttributeError("Path to target .odb file " "is not set or given.")

            else:
                odb_path = self.odb_path

        if h5_path is not None:
            h5_path = pathlib.Path(h5_path)
            if set_h5:
                self.h5_path = h5_path

        else:
            if not hasattr(self, "h5_path"):
                raise AttributeError("Path to target .hdf5 file " "is not set or given")

            else:
                h5_path = self.h5_path

        self._convert(h5_path, odb_path)

    @classmethod
    def convert_by_path(cls, h5_path: pathlib.Path, odb_path: pathlib.Path) -> None:
        h5_path = pathlib.Path(h5_path)
        odb_path = pathlib.Path(odb_path)
        cls()._convert(h5_path, odb_path)

    def _convert(self: Self, h5_path: pathlib.Path, odb_path: pathlib.Path | None) -> None:
        if odb_path is None:
            raise ValueError("odb_path attribute is not set!")
        # TODO
        # convert_pickle_input_dict: dict[str, int | str | list[str] | list[int] | list[chain[Any]] None] = {
        convert_pickle_input_dict: dict[str, Any] = {
            "cpus": self.cpus,
            #"nodes": self.nodes,
            #"nodesets": self.nodesets,
            #"parts": self.parts,
            #"steps": self.steps,
            "defaults_for_outputs": self.defaults_for_outputs,
        }

        pickle_file: BinaryIO
        with open(self._convert_pickle_path, "wb") as pickle_file:
            pickle.dump(convert_pickle_input_dict, pickle_file, protocol=2)

        odb_convert_args: list[pathlib.Path | str] = [
            self.abaqus_executable,
            "python",
            self._convert_script_path,
            odb_path,
            self._convert_pickle_path,
            self._convert_result_path,
        ]

        # TODO
        # shell=True is BAD PRACTICE, but abaqus python won't run without it
        subprocess.run(odb_convert_args, shell=True)

        result_file: BinaryIO
        result_dir: pathlib.Path

        try:
            with open(self._convert_result_path, "rb") as result_file:
                result_dir = pathlib.Path(pickle.load(result_file))

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"File {self._convert_result_path} was not found. See previous Python 2 errors"
            ) from e

        pathlib.Path.unlink(self._convert_result_path)

        #temp_low = self.temp_low if hasattr(self, "temp_low") else None
        #temp_high = self.temp_high if hasattr(self, "temp_high") else None

        convert_npz_to_h5(
            h5_path,
            result_dir,
            #temp_low,
            #temp_high,
            #self.nodesets,
            #self.nodes,
            #self.parts,
            #self.steps,
            self.output_mapping,
            #odb_path,
        )

        if result_dir.exists():
            shutil.rmtree(result_dir)

    def get_odb_info(self: Self) -> None:
        # Ideally this would not work this way, but
        # the python2 makes transferring a raw dict the easiest option
        result: dict[
            str,
            int
            | list[str]
            | dict[str, int]
            | dict[str, list[int]]
            | dict[str, list[str]],
        ] = self._get_odb_info()
        # No setters for these, just this method
        self._frame_range = result["frame_range"]
        self._frame_keys = result["frame_keys"]
        self._frame_keys_per_step = result["frame_keys_per_step"]
        self._step_names = result["step_names"]
        self._step_lens = result["step_lens"]
        self._nodeset_names = result["nodeset_names"]
        self._elementset_names = result["elementset_names"]
        self._part_names = result["part_names"]
        self._node_range = result["node_range"]
        self._element_range = result["element_range"]
        self._nodes_per_part = result["nodes_per_part"]
        self._nodes_per_nodeset = result["nodes_per_nodeset"]
        self._nodeset_per_part = result["nodeset_per_part"]
        self._elements_per_part = result["elements_per_part"]
        self._elements_per_elementset = result["elements_per_elementset"]
        self._elementset_per_part = result["elementset_per_part"]

        # TODO
        # if hasattr(self, "h5_path"):
        #    hdf5_file: h5py.File
        #    with h5py.File(self.h5_path, "r+") as hdf5_file:
        #        total_name: str = str(self.h5_path.stem)
        #        hdf5_file[total_name].attrs["frame_range"] = self._frame_range
        #        hdf5_file[total_name].attrs["frame_keys"] = self._frame_keys
        #        for step, frame_keys in self._frame_keys_per_step.items():
        #            hdf5_file[total_name].attrs[f"frame_keys_per_{step}"] = frame_keys
        #        hdf5_file[total_name].attrs["step_names"] = self._step_names
        #        for step, length in self._step_lens.items():
        #            hdf5_file[total_name].attrs[f"step_{step}_length"] = length
        #        hdf5_file[total_name].attrs["nodeset_names"] = self._nodeset_names
        #        hdf5_file[total_name].attrs["part_names"] = self._part_names
        #        hdf5_file[total_name].attrs["node_range"] = self._node_range
        #        for part, node_range in self._node_ranges_per_part.items():
        #            hdf5_file[total_name].attrs[f"node_ranges_per_{part}"] = node_range

    @classmethod
    def get_odb_info_from_file(
        cls, path: pathlib.Path
    ) -> dict[str, tuple[int, int] | list[str] | dict[str, tuple[int, int]]]:
        return cls()._get_odb_info(path)

    def _get_odb_info(
        self: Self, path: pathlib.Path | None = None
    ) -> dict[str, tuple[int, int] | list[str] | dict[str, tuple[int, int]]]:
        if path is None:
            if hasattr(self, "odb_path"):
                path = self.odb_path

            else:
                raise AttributeError("Either pass in or set odb_path")

        # TODO
        # shell=True is bad practice, but abaqus python won't run without it
        subprocess.run(
            [
                self.abaqus_executable,
                "python",
                str(self._get_odb_info_script_path),
                str(path),
                str(self._get_odb_info_result_path),
            ],
            shell=True,
        )

        result_file: BufferedReader
        try:
            with open(self._get_odb_info_result_path, "rb") as result_file:
                return pickle.load(result_file)

        except FileNotFoundError:
            raise FileNotFoundError(
                f"File {self._get_odb_info_result_path} was not found. See previous Python 2 errors"
            )

    def load_h5(self: Self) -> None:
        if not hasattr(self, "h5_path"):
            raise AttributeError(
                "h5_path attribute must be set before " "calling load_h5 method."
            )

        if isinstance(self._data_handler, DataLoader):
            # Only case where this should be set, bypass the setter
            self._h5_status, self._data = self._data_handler.load_h5(
                self.h5_path, self.cpus, self.output_mapping
            )
            self._data_handler = DataUnloader()
            self._iterable_cols_and_vals = {}
            self._find_iterable_keys()
            
            if len(self._iterable_cols_and_vals) == 0:
                print(f"WARNING: No iterable columns found in {self.h5_path}")
            elif len(self._iterable_cols_and_vals) == 1:
                self._iterator_key = list(self._iterable_cols_and_vals.keys())[0]
            else:
                self._iterator_key = list(self._iterable_cols_and_vals.keys())[0]
                remaining_iterator_vals: list[str] = list(self._iterable_cols_and_vals.keys()).remove(self._iterator_key)
                print(f"INFO: {self._iterator_key} selection as default iterator value. Other valid options are: {remaining_iterator_vals}")

            # use a _outputs member because we don't want a normal setter
            self._outputs = OdbpOutputs(self.data, self.output_mapping, self.defaults_for_outputs)

        else:
            raise AttributeError(
                "load_h5 can only be used once in a row, "
                "before a .hdf5 file is loaded. Call unload_h5 on this "
                "Odbp before calling load_h5."
            )

    def unload_h5(self: Self) -> None:
        if isinstance(self._data_handler, DataUnloader):
            self._data_handler.unload_h5()
            # Unified deleter
            del self.data
            self._data_handler = DataLoader()
            del self._iterable_cols_and_vals

        else:
            raise AttributeError("unload_h5 can only be called after " "load_h5.")

    def _find_iterable_keys(self: Self) -> None:
        node_cols: set[str] = set(self.data.node_data.columns.values)
        elem_cols: set[str] = set(self.data.element_data.columns.values)
        iterable_cols: list[str] = list(node_cols & elem_cols)
        col: str
        for col in iterable_cols:
            nodal_unique_vals: npt.NDArray = np.sort(np.unique(self.data.node_data[col]))
            elemental_unique_vals: npt.NDArray = np.sort(np.unique(self.data.element_data[col]))
            if np.array_equal(nodal_unique_vals, elemental_unique_vals):
                self._iterable_cols_and_vals[col] = nodal_unique_vals

    def plot(self: Self, target_key: str | None = None, *, target_key_lower_bound: float | None = None, target_key_upper_bound: float | None = None) -> None:
        use_cmap: bool = False
        if target_key is not None:
            if target_key not in self.outputs.outputs_by_names:
                raise ValueError(f"Key {target_key} is not present in the data!")

            use_cmap = True

            mesh_key_quantity_def: str
            surf_key_quantity_def: str
            if target_key in self.data.node_data.columns:
                mesh_key_quantity_def = surf_key_quantity_def = "vertices"
            elif target_key in self.data.element_data.columns:
                mesh_key_quantity_def = "cells"
                surf_key_quantity_def = "faces"

            epsilon: float = np.finfo(float).eps
            if target_key_lower_bound is None:
                target_key_lower_bound = self.outputs.outputs_by_names[target_key].cmap_min

            if not self.outputs.outputs_by_names[target_key].cmap_min_equal:
                target_key_lower_bound += epsilon

            if target_key_upper_bound is None:
                target_key_upper_bound = self.outputs.outputs_by_names[target_key].cmap_max

            if not self.outputs.outputs_by_names[target_key].cmap_max_equal:
                target_key_upper_bound -= epsilon


        ps.init()
        ps.set_ground_plane_mode("none")
        ps.set_up_dir("z_up")
        ps.set_front_dir("x_front")
        ps.set_background_color(self.background_color)
        ps.set_view_projection_mode(self.view_projection_mode)
        ps.set_program_name("ODBP")
        frame: OdbpData
        for frame in self:
            plot_type: OdbpPlotType
            hull: ConvexHull
            mesh: ps.VolumeMesh
            cloud: ps.PointCloud
            surf: ps.SurfaceMesh
            pc_nodes: npt.NDArray
            pc_dists: npt.NDArray

            # We need the coordinate data to plot by. This is how we'll get it
            c: str
            coord_index: list[str] = [self.output_mapping.get(c, c) for c in ["COORD1", "COORD2", "COORD3",]]

            current_frame_structs: list[tuple[npt.NDArray, ps.PointCloud | ps.SurfaceMesh | ps.VolumeMesh, str] | tuple[npt.NDArray, ps.PointCloud | ps.SurfaceMesh | ps.VolumeMesh]] = []

            match self.plot_type:
                case self.ODBPPLOTYPE.ONLY_SURFACE | self.ODBPPLOTYPE.ONLY_POINT_CLOUD | self.ODBPPLOTYPE.ONLY_ELEMS as plot_type:
                    nodes: pd.DataFrame = frame.node_data.iloc[np.unique(frame.element_connectivity - 1)]
                    if nodes.shape[0] > 0:
                        new_conn_mapping: dict[int, int] = {o:i for i, o in enumerate(np.sort(np.unique(frame.element_connectivity)))}
                        new_elem_conn: npt.NDArray
                        if frame.element_connectivity.shape[0] > 0:
                            new_elem_conn = np.vectorize(new_conn_mapping.__getitem__)(frame.element_connectivity)
                        else:
                            new_elem_conn = frame.element_connectivity

                        match plot_type:
                            case self.ODBPPLOTYPE.ONLY_SURFACE:
                                hull = ConvexHull(nodes[coord_index].to_numpy())
                                surf = ps.register_surface_mesh("surf", hull.points, hull.simplices, enabled=True, edge_color=(0., 0., 0.,), edge_width=1, material=self.material)
                                if use_cmap:
                                    current_frame_structs.append((nodes[[target_key]].to_numpy().flatten(), surf, surf_key_quantity_def))

                            case self.ODBPPLOTYPE.ONLY_ELEMS:
                                mesh = ps.register_volume_mesh("mesh", nodes[coord_index].to_numpy(), mixed_cells=new_elem_conn, enabled=True, edge_color=(0, 0, 0), edge_width=1, material=self.material)
                                if use_cmap:
                                    current_frame_structs.append((nodes[[target_key]].to_numpy().flatten(), mesh, mesh_key_quantity_def))

                            case self.ODBPPLOTYPE.ONLY_POINT_CLOUD:
                                pc_nodes = nodes[coord_index].to_numpy()
                                pc_dists = cdist(pc_nodes, pc_nodes, "euclidean")
                                pc_dists[pc_dists == 0.0] = np.inf
                                if self.point_cloud_dynamic_radius:
                                    pc_dists = np.min(pc_dists, axis=0)
                                else:
                                    pc_dists = np.full((pc_nodes.shape[0],), np.min(pc_dists))
                                pc_dists /= 2.0
                                cloud = ps.register_point_cloud("cloud", pc_nodes, enabled=True, material=self.point_cloud_material)
                                cloud.add_scalar_quantity("radius", pc_dists)
                                cloud.set_point_radius_quantity("radius", autoscale=False)
                                if use_cmap:
                                    current_frame_structs.append((nodes[[target_key]].to_numpy().flatten(), cloud))

                case self.ODBPPLOTYPE.SURFACE_WITH_REMAINING_NODES | self.ODBPPLOTYPE.POINT_CLOUD_WITH_REMAINING_NODES as plot_type:
                    node_label_key: list[str] = [self.output_mapping.get("Node Label", "Node Label")]
                    selected_node_indices: npt.NDArray = np.concatenate((np.unique(frame.element_connectivity), frame.remaining_nodes_not_in_elements[node_label_key].to_numpy().flatten()), axis=0).astype(int)
                    selected_nodes: pd.DataFrame = frame.node_data.iloc[selected_node_indices - 1]
                    if selected_nodes.shape[0] > 0:
                        match plot_type:
                            case self.ODBPPLOTYPE.SURFACE_WITH_REMAINING_NODES:
                                hull = ConvexHull(selected_nodes[coord_index].to_numpy())
                                surf = ps.register_surface_mesh("surf", hull.points, hull.simplices, enabled=True, edge_color=(0,0,0), edge_width=1, material=self.material)
                                if use_cmap:
                                    current_frame_structs.append((selected_nodes[[target_key]].to_numpy().flatten(), surf, surf_key_quantity_def))
                            
                            case self.ODBPPLOTYPE.POINT_CLOUD_WITH_REMAINING_NODES:
                                pc_nodes = selected_nodes[coord_index].to_numpy()
                                pc_dists = cdist(pc_nodes, pc_nodes, "euclidean")
                                pc_dists[pc_dists == 0.0] = np.inf
                                if self.point_cloud_dynamic_radius:
                                    pc_dists = np.min(pc_dists, axis=0)
                                else:
                                    pc_dists = np.full((pc_nodes.shape[0],), np.min(pc_dists))
                                pc_dists /= 2.0
                                cloud = ps.register_point_cloud("cloud", pc_nodes, enabled=True, material=self.point_cloud_material)
                                cloud.add_scalar_quantity("radius", pc_dists)
                                cloud.set_point_radius_quantity("radius", autoscale=False)
                                if use_cmap:
                                    current_frame_structs.append((selected_nodes[[target_key]].to_numpy().flatten(), cloud))

                case self.ODBPPLOTYPE.ELEMS_AND_POINT_CLOUD_OF_REMAINING_NODES | self.ODBPPLOTYPE.ELEMS_AND_SURFACE_OF_REMAINING_NODES as plot_type:
                    nodes: pd.DataFrame = frame.node_data.iloc[np.unique(frame.element_connectivity - 1)]
                    new_conn_mapping: dict[int, int] = {o:i for i, o in enumerate(np.sort(np.unique(frame.element_connectivity)))}
                    new_elem_conn: npt.NDArray
                    if frame.element_connectivity.shape[0] > 0:
                        new_elem_conn = np.vectorize(new_conn_mapping.__getitem__)(frame.element_connectivity)
                    else:
                        new_elem_conn = frame.element_connectivity

                    if nodes.shape[0] > 0:
                        mesh = ps.register_volume_mesh("mesh", nodes[coord_index].to_numpy(), mixed_cells=new_elem_conn, enabled=True, edge_color=(0, 0, 0), edge_width=1, material=self.material)
                        if use_cmap:
                            current_frame_structs.append((nodes[[target_key]].to_numpy().flatten(), mesh, mesh_key_quantity_def))

                        match plot_type:
                            case self.ODBPPLOTYPE.ELEMS_AND_POINT_CLOUD_OF_REMAINING_NODES:
                                if frame.remaining_nodes_not_in_elements is not None:
                                    pc_nodes = frame.remaining_nodes_not_in_elements[coord_index].to_numpy()
                                    pc_dists = cdist(pc_nodes, pc_nodes, "euclidean")
                                    pc_dists[pc_dists == 0.0] = np.inf
                                    if self.point_cloud_dynamic_radius:
                                        pc_dists = np.min(pc_dists, axis=0)
                                    else:
                                        pc_dists = np.full((pc_nodes.shape[0],), np.min(pc_dists))
                                    pc_dists /= 2.0
                                    cloud = ps.register_point_cloud("cloud", pc_nodes, enabled=True, material=self.point_cloud_material)
                                    cloud.add_scalar_quantity("radius", pc_dists)
                                    cloud.set_point_radius_quantity("radius", autoscale=False)
                                    if use_cmap:
                                        current_frame_structs.append((frame.remaining_nodes_not_in_elements[[target_key]].to_numpy().flatten(), cloud))

                            case self.ODBPPLOTYPE.ELEMS_AND_SURFACE_OF_REMAINING_NODES: 
                                if frame.remaining_nodes_not_in_elements is not None:
                                    hull = ConvexHull(frame.remaining_nodes_not_in_elements[coord_index].to_numpy())
                                    surf = ps.register_surface_mesh("surf", hull.points, hull.simplices, enabled=True, edge_color=(0, 0, 0), edge_width=1, material=self.material)
                                    if use_cmap:
                                        current_frame_structs.append((frame.remaining_nodes_not_in_elements[[target_key]].to_numpy().flatten(), surf, surf_key_quantity_def))

            frame_struct: tuple[npt.NDArray, ps.PointCloud | ps.SurfaceMesh | ps.VolumeMesh, str] | tuple[npt.NDArray, ps.PointCloud | ps.SurfaceMesh | ps.VolumeMesh]
            struct: ps.PointCloud | ps.SurfaceMesh | ps.VolumeMesh
            for frame_struct in current_frame_structs:
                scalar_data: npt.NDArray
                if len(frame_struct) == 3:
                    def_key: str
                    scalar_data, struct, def_key = frame_struct
                    struct.add_scalar_quantity(target_key, scalar_data, defined_on=def_key, enabled=True, vminmax=(target_key_lower_bound, target_key_upper_bound), cmap=self.colormap)
                elif len(frame_struct) == 2:
                    scalar_data, struct = frame_struct
                    struct.add_scalar_quantity(target_key, scalar_data, enabled=True, vminmax=(target_key_lower_bound, target_key_upper_bound), cmap=self.colormap)

                    
            bounds: tuple[npt.NDArray, npt.NDArray] = ps.get_bounding_box()
            dimensions: npt.NDArray = np.abs(bounds[1] - bounds[0])
            midpoint: npt.NDArray = (bounds[1] + bounds[0]) / 2.0
            camera_location_scalar: npt.NDArray
            up_dir: npt.NDArray
            camera_location_scalar, up_dir = self._views[self.view]
            camera_location: npt.NDArray = (1.1 * camera_location_scalar * dimensions) + midpoint
            ps.look_at_dir(camera_location, midpoint, up_dir)

            if self.interactive:
                ps.show()

            # Screenshot here
            if self.save:
                filename: str = self.filename if self.filename != "" else self.h5_path.stem
                ext: str = self.save_format if self.save_format.startswith(".") else f".{self.save_format}"
                target_file_path: pathlib.Path = self._result_dir / pathlib.Path(f"{filename}_{frame.node_data[self._iterator_key].to_numpy()[0]}{ext}")
                ps.screenshot(str(target_file_path), transparent_bg=self.transparent_background)                

            for frame_struct in current_frame_structs:
                struct = frame_struct[1]
                struct.remove()

    #def get_odb_state(self) -> str:
    #    return self.get_odb_settings_state()


class DataLoader:
    def load_h5(
        self: Self, h5_path: pathlib.Path, cpus: int, output_mapping: dict[str, str]
    ) -> tuple[dict[str, str], OdbpData]:
        return get_h5_data(h5_path, cpus, output_mapping)


class DataUnloader:
    def unload_h5(self: Self) -> None:
        pass
