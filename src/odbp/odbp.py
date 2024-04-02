#!/usr/bin/env python3

"""
ODBPlotter odb.py
ODBPlotter
https://www.github.com/Computational-Mechanics-Materials-Lab/ODBPlotter
MIT License (c) 2023

Implement the main ODB class, which is used to process the .hdf5 data and
filter it by desired values.
"""

import subprocess
import shutil
import pathlib
import pickle
import h5py
import multiprocessing

import numpy as np
import numpy.typing as npt
import pandas as pd
import polyscope as ps

from typing import Any, Iterator, BinaryIO, Self
from io import BufferedReader

from .odbp_settings import OdbpSettings, ExtremaDict
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

    def __init__(self) -> None:
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

        self._h5_status: dict[str, str]


    def __iter__(self) -> Self:
        return self

    def __next__(self) -> Iterator[OdbpData]:
        data_for_iteration: npt.NDArray = np.copy(self._iterable_cols_and_vals[self._iterator_key])
        iter_min: Any
        iter_max: Any
        iter_min, iter_max = self.extrema[self._iterator_key]
        data_for_iteration = data_for_iteration[data_for_iteration > iter_min]
        data_for_iteration = data_for_iteration[data_for_iteration < iter_max]
        try:
            if self._iterator_ind >= len(data_for_iteration):
                self._iterator_ind = 0
                raise StopIteration

            ind: int = self._iterator_ind
            self._iterator_ind += 1
            return_data: OdbpData = OdbpData(
                element_data=self.data.element_data,
                node_data = self.data.node_data,
                element_connectivity=self.data.element_connectivity,
                element_sets_to_elements=self.data.element_sets_to_elements,
                node_sets_to_nodes=self.data.node_sets_to_nodes,
                part_to_element_set=self.data.part_to_element_set,
                part_to_node_set=self.data.part_to_node_set
            )
            key: str
            key_min: Any
            key_max: Any
            for key, (key_min, key_max) in self.extrema.items():
                if key == self._iterator_key:
                    continue

                if key in return_data.node_data:
                    return_data.node_data = return_data.node_data[return_data.node_data[key] > key_min]
                    return_data.node_data = return_data.node_data[return_data.node_data[key] < key_max]

                if key in return_data.element_data:
                    return_data.element_data = return_data.element_data[return_data.element_data[key] > key_min]
                    return_data.element_data = return_data.element_data[return_data.element_data[key] < key_max]

            return_data.node_data = return_data.node_data[return_data.node_data[self._iterator_key] == data_for_iteration[ind]]
            return_data.element_data = return_data.element_data[return_data.element_data[self._iterator_key] == data_for_iteration[ind]]
            return return_data

        except AttributeError:
            raise AttributeError(
                "Odbp() object only functions as an iterator"
                "After load_h5() has been called."
            )

    @property
    def data(self) -> OdbpData:
        return self._data

    @data.deleter
    def data(self) -> None:
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
    def h5_status(self) -> dict[str, str]:
        return self._h5_status

    def convert(
        self,
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

    def _convert(self, h5_path: pathlib.Path, odb_path: pathlib.Path | None) -> None:
        if odb_path is None:
            raise ValueError("odb_path attribute is not set!")
        # TODO
        # convert_pickle_input_dict: dict[str, int | str | list[str] | list[int] | list[chain[Any]] None] = {
        convert_pickle_input_dict: dict[str, Any] = {
            "cpus": self.cpus,
            "nodes": self.nodes,
            "nodesets": self.nodesets,
            "time_step": self.time_step,
            "parts": self.parts,
            "steps": self.steps,
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

        temp_low = self.temp_low if hasattr(self, "temp_low") else None
        temp_high = self.temp_high if hasattr(self, "temp_high") else None

        convert_npz_to_h5(
            h5_path,
            result_dir,
            temp_low,
            temp_high,
            self.time_step,
            self.nodesets,
            self.nodes,
            self.parts,
            self.steps,
            self.output_mapping,
            odb_path,
        )

        if result_dir.exists():
            shutil.rmtree(result_dir)

    def get_odb_info(self) -> None:
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
        self, path: pathlib.Path | None = None
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

    def load_h5(self) -> None:
        if not hasattr(self, "h5_path"):
            raise AttributeError(
                "h5_path attribute must be set before " "calling load_h5 method."
            )

        if isinstance(self._data_handler, DataLoader):
            # Only case where this should be set, bypass the setter
            self._h5_status, self._data = self._data_handler.load_h5(
                self.h5_path, self.cpus
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

            col: str
            col_data: npt.NDArray
            col_min: Any
            col_max: Any
            bounds_dict: dict[str, tuple[Any, Any]] = {}
            for col in self.data.node_data.columns.values:
                col_data = self.data.node_data[col].to_numpy()
                col_min = np.min(col_data)
                col_max = np.max(col_data)
                bounds_dict[col] = (col_min, col_max)

            for col in self.data.element_data.columns.values:
                col_data = self.data.element_data[col].to_numpy()
                col_min = np.min(col_data)
                col_max = np.max(col_data)
                bounds_dict[col] = (col_min, col_max)

            self._extrema = ExtremaDict(bounds_dict)

        else:
            raise AttributeError(
                "load_h5 can only be used once in a row, "
                "before a .hdf5 file is loaded. Call unload_h5 on this "
                "Odbp before calling load_h5."
            )

    def unload_h5(self) -> None:
        if isinstance(self._data_handler, DataUnloader):
            self._data_handler.unload_h5()
            # Unified deleter
            del self.data
            self._data_handler = DataLoader()
            del self._iterable_cols_and_vals

        else:
            raise AttributeError("unload_h5 can only be called after " "load_h5.")

    def _find_iterable_keys(self) -> None:
        node_cols: set[str] = set(self.data.node_data.columns.values)
        elem_cols: set[str] = set(self.data.element_data.columns.values)
        iterable_cols: list[str] = list(node_cols & elem_cols)
        col: str
        for col in iterable_cols:
            nodal_unique_vals: npt.NDArray = np.sort(np.unique(self.data.node_data[col]))
            elemental_unique_vals: npt.NDArray = np.sort(np.unique(self.data.element_data[col]))
            if np.array_equal(nodal_unique_vals, elemental_unique_vals):
                self._iterable_cols_and_vals[col] = nodal_unique_vals

    def plot(self) -> None:
        ps.init()
        ps.set_ground_plane_mode("none")
        ps.set_up_dir("z_up")
        ps.set_front_dir("y_front")
        frame: OdbpData
        for frame in self:
            nodes: npt.NDArray = frame.node_data[["X", "Y", "Z"]].to_numpy() 
            target_connectivity: npt.NDArray = frame.element_connectivity - 1
            status_elems: npt.NDArray = frame.element_data["STATUS"] == 1
            target_connectivity = target_connectivity[status_elems.to_numpy()]
            mesh: ps.VolumeMesh = ps.register_volume_mesh(f"mesh", nodes, mixed_cells=target_connectivity, enabled=True, edge_color=(0, 0, 0), edge_width=1)
            mesh.add_scalar_quantity("temp", frame.node_data["Temp"].to_numpy(), defined_on="vertices", vminmax=(300.0, 1727.0), cmap="turbo", enabled=True)
            ps.show()
    ## 3D Plotting
    # def plot_3d_all_times(
    #    self,
    #    target_output: str,
    #    *,
    #    title: str | None = None,
    #    target_nodes: pd.DataFrame | None = None,
    #    plot_type: str | None = None,
    # ) -> list[pathlib.Path]:
    #    """ """
    #    if not PYVISTA_AVAILABLE:
    #        raise Exception(
    #            "Plotting cabailities are not included."
    #            ' Please install pyvista via pip install odb-plotter["plot"]'
    #            ' or odb-plotter["all"] rather than pip install odb-plotter'
    #            " Or export the data from Odbp.extract()",
    #            " or Odbp.convert() to another tool,"
    #            " such as matplotlib, plotly, or bokeh.",
    #        )

    #    title = self.title
    #    title = self.h5_path.stem if (title is None or not title) else title

    #    if target_nodes is None:
    #        if not hasattr(self, "data"):
    #            self.load_h5()

    #        target_nodes = self.data

    #    if self.results_dir is not None:
    #        if not self.result_dir.exists():
    #            self.result_dir.mkdir()

    #    target_times = target_nodes["Time"].unique()
    #    # There should be more elegant ways to do this, but np.where was misbehaving, and this works fine
    #    target_times = target_times[target_times >= self.time_low]
    #    target_times = target_times[target_times <= self.time_high]
    #    if self.interactive:
    #        results = []
    #        for time in target_times:
    #            results.append(
    #                self._plot_3d_single(
    #                    time, title, target_output, target_nodes, plot_type
    #                )
    #            )

    #    else:
    #        with multiprocessing.Pool(processes=self.cpus) as pool:
    #            results = pool.starmap(
    #                self._plot_3d_single,
    #                (
    #                    (time, title, target_output, target_nodes, plot_type)
    #                    for time in target_times
    #                ),
    #            )

    #    return results

    # def _plot_3d_single(
    #    self,
    #    time: float,
    #    title: str,
    #    target_output: str,
    #    target_nodes: pd.DataFrame,
    #    plot_type: str | None,
    # ) -> pathlib.Path | None:
    #    """ """
    #    if not PYVISTA_AVAILABLE:
    #        raise Exception(
    #            "Plotting cabailities are not included."
    #            ' Please install pyvista via pip install odb-plotter["plot"]'
    #            ' or odb-plotter["all"] rather than pip install odb-plotter'
    #            " Or export the data from Odbp.extract() to another tool,"
    #            " such as matplotlib, plotly,  or bokeh."
    #        )

    #    combined_label: str = f"{title}-{round(time, 2):.2f}"

    #    plotter: pv.Plotter = pv.Plotter(
    #        off_screen=(not self.interactive), window_size=(1920, 1080)
    #    )
    #    plotter.add_light(pv.Light(light_type="headlight"))

    #    plotter.add_text(
    #        combined_label,
    #        position="upper_edge",
    #        color=self.font_color,
    #        font=self.font,
    #        font_size=self.font_size,
    #    )

    #    mesh = self.get_mesh(time, target_nodes, target_output)

    #    epsilon: float = np.finfo(float).eps
    #    plotter.add_mesh(
    #        mesh,
    #        scalars=target_output,
    #        cmap=pv.LookupTable(
    #            cmap=self._colormap,
    #            # Handle Epsilon
    #            scalar_range=(self.temp_low + epsilon, self.temp_high - epsilon),
    #            above_range_color=self.above_range_color,
    #            below_range_color=self.below_range_color,
    #        ),
    #        scalar_bar_args={
    #            "vertical": True,
    #            "title": "Nodal Temperature (Kelvin)",  # TODO !!!
    #            "font_family": self.font,
    #            "title_font_size": self.font_size + 4,
    #            "label_font_size": self.font_size,
    #            "color": self.font_color,
    #            "fmt": "%.2f",
    #            "position_x": 0.05,
    #            "position_y": 0.05,
    #        },
    #    )

    #    if self.show_axes:
    #        # TODO Dynamically update these
    #        x_low, x_high, y_low, y_high, z_low, z_high = mesh.bounds

    #        x_pad = (x_high - x_low) / 4.0
    #        y_pad = (y_high - y_low) / 4.0
    #        z_pad = (z_high - z_low) / 4.0
    #        pads = [x_pad, y_pad, z_pad]
    #        pads.sort()
    #        pad = pads[1]

    #        ruler_x = plotter.add_ruler(
    #            pointa=(x_low, y_high + pad, z_low - pad),
    #            pointb=(x_high, y_high + pad, z_low - pad),
    #            label_format="%.2f",
    #            font_size_factor=0.4,
    #            label_color=self.axis_text_color,
    #            title="X Axis",
    #        )
    #        ruler_x.SetRange(x_low, x_high)

    #        ruler_y = plotter.add_ruler(
    #            pointa=(x_high + pad, y_low, z_low - pad),
    #            pointb=(x_high + pad, y_high, z_low - pad),
    #            label_format="%.2f",
    #            font_size_factor=0.4,
    #            label_color=self.axis_text_color,
    #            title="Y Axis",
    #        )
    #        ruler_y.SetRange(y_low, y_high)

    #        ruler_z = plotter.add_ruler(
    #            pointa=(x_high + pad, y_low - pad, z_low),
    #            pointb=(x_high + pad, y_low - pad, z_high),
    #            label_format="%.2f",
    #            font_size_factor=0.4,
    #            label_color=self.axis_text_color,
    #            title="Z Axis",
    #        )
    #        ruler_z.SetRange(z_low, z_high)

    #    plotter.set_background(color=self.background_color)

    #    invalid_view = True
    #    for k in self._views.keys():
    #        if self.view in k:
    #            invalid_view = False
    #            view_angle, viewup, roll = self._views[k]
    #            break

    #    if invalid_view:
    #        raise RuntimeError("View Panic")
    #    plotter.view_vector(view_angle, viewup=viewup)
    #    plotter.camera.roll = roll

    #    #if not self.save:
    #    #    plotter.show(interactive_update=True)
    #    #else:
    #    #    plotter.show(
    #    #        before_close_callback=lambda p: p.screenshot(
    #    #            self.result_dir
    #    #            / f"{plot_type + '_' if plot_type is not None else ''}{combined_label}{self.save_format}"
    #    #        )
    #    #    )

    #    return

    # def get_mesh(self, time, target=None, output=None) -> pv.PolyData:
    #    if target is None:
    #        target = self.data
    #    filtered_target_nodes: pd.DataFrame = target[target["Time"] == time]
    #    filtered_target_nodes = filtered_target_nodes[
    #        filtered_target_nodes["X"] >= self.x_low
    #    ]
    #    filtered_target_nodes = filtered_target_nodes[
    #        filtered_target_nodes["X"] <= self.x_high
    #    ]
    #    filtered_target_nodes = filtered_target_nodes[
    #        filtered_target_nodes["Y"] >= self.y_low
    #    ]
    #    filtered_target_nodes = filtered_target_nodes[
    #        filtered_target_nodes["Y"] <= self.y_high
    #    ]
    #    filtered_target_nodes = filtered_target_nodes[
    #        filtered_target_nodes["Z"] >= self.z_low
    #    ]
    #    filtered_target_nodes = filtered_target_nodes[
    #        filtered_target_nodes["Z"] <= self.z_high
    #    ]
    #    # points: pv.PointSet = pv.PointSet(
    #    points = pv.PointSet(filtered_target_nodes[["X", "Y", "Z"]].to_numpy())

    #    # if output is not None:
    #    #    points[output] = filtered_target_nodes[output].to_numpy()

    #    mesh = points.delaunay_3d()
    #    mesh.plot()

    #    vox = pv.voxelize(mesh)
    #    vox.plot()

    #    if output is not None:
    #        vox[output] = filtered_target_nodes[output].to_numpy()

    #    return pv.voxelize(points.delaunay_3d())
    #    # return points.delaunay_3d()

    def get_odb_state(self) -> str:
        return self.get_odb_settings_state()


class DataLoader:
    def load_h5(
        self, h5_path: pathlib.Path, cpus: int
    ) -> tuple[dict[str, str], OdbpData]:
        return get_h5_data(h5_path, cpus)


class DataUnloader:
    def unload_h5(self) -> None:
        pass
