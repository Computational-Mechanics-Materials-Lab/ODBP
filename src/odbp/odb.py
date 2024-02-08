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
import pandas as pd

from typing import TextIO, Union, Any, Tuple, List, Dict, Optional, Iterator, BinaryIO
from abc import abstractmethod

from .odb_settings import OdbSettings
from .writer import convert_npz_to_hdf
from .reader import get_odb_data
from .types import DataFrameType, NDArrayType, H5PYFileType
from .magic import ensure_magic, ODB_MAGIC_NUM, HDF_MAGIC_NUM

try:
    import pyvista as pv
except ImportError:
    PYVISTA_AVAILABLE = False
else:
    PYVISTA_AVAILABLE = True


class Odb(OdbSettings):
    """
    Stores Data from a .hdf5, implements extractor methods
    to transfer from .odb to .hdf5
    Implements abilities to resize the dimenisons or timeframe of the data
    """

    __slots__ = (
        # file-defined components
        "_odb_handler",
        "_odb",
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
        "_times",
        "_frame_keys",
        "_frame_keys_per_step",
        "_frame_range",
        "_step_names",
        "_step_lens",
        "_nodeset_names",
        "_part_names",
        "_node_range",
        "_node_ranges_per_part",
        "_extracted_nodes",
        "_hdf_status",
    )

    def __init__(self) -> None:
        """
        Type Hints and hard-coded parameters. See the @staticmethod
        "constructors" of this class in order to learn about initialization
        """

        super().__init__()

        self._odb_handler: Union[OdbLoader, OdbUnloader] = OdbLoader()
        self._odb: DataFrameType

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

        self._frame_keys: List[str]
        self._frame_keys_per_step: Dict[str, List[str]]
        self._frame_range: Tuple[int, int]
        self._step_names: List[str]
        self._step_lens: Dict[str, int]
        self._nodeset_names: List[str]
        self._part_names: List[str]
        self._node_range: Tuple[int, int]
        self._node_ranges_per_part: Dict[str, Tuple[int, int]]

        self._extracted_nodes: DataFrameType

        self._iterator_ind: int = 0
        self._times: NDArrayType

        self.hdf_status: Dict[str, str]

    def __iter__(self) -> Iterator[DataFrameType]:
        return self

    def __next__(self) -> DataFrameType:
        try:
            if self._iterator_ind >= len(self._times):
                self._iterator_ind = 0
                raise StopIteration

            ind: int = self._iterator_ind
            self._iterator_ind += 1
            return self[self["Time"] == self._times[ind]]

        except AttributeError:
            raise AttributeError(
                "Odb() object only functions as an iterator"
                "After load_hdf() has been called."
            )

    def __getitem__(self, key: Any) -> Any:
        try:
            return self.odb[key]

        except AttributeError:
            raise AttributeError(
                "Odb key access is not available before "
                "The load_hdf() method is called."
            )

    @property
    def odb(self) -> DataFrameType:
        return self._odb

    @odb.deleter
    def odb(self) -> None:
        del self._odb

    @property
    def frame_range(self) -> "Tuple[int, int]":
        return self._frame_range

    @property
    def frame_keys(self) -> "List[str]":
        return self._frame_keys

    @property
    def step_names(self) -> "List[str]":
        return self._step_names

    @property
    def step_lens(self) -> "Dict[str, int]":
        return self._step_lens

    @property
    def frame_keys_per_step(self) -> "Dict[str, List[str]]":
        return self._frame_keys_per_step

    @property
    def nodeset_names(self) -> "List[str]":
        return self._nodeset_names

    @property
    def part_names(self) -> "List[str]":
        return self._part_names

    @property
    def node_range(self) -> "Tuple[int, int]":
        return self._node_range

    @property
    def node_ranges_per_part(self) -> "Dict[str, Tuple[int, int]]":
        return self._node_ranges_per_part

    @property
    def hdf_status(self) -> "Dict[str, str]":
        return self._hdf_status

    def convert(
        self,
        hdf_path: "Optional[pathlib.Path]" = None,
        *,
        odb_path: "Optional[pathlib.Path]" = None,
        set_odb: bool = False,
        set_hdf: bool = False,
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

        if hdf_path is not None:
            hdf_path = pathlib.Path(hdf_path)
            if set_hdf:
                self.hdf_path = hdf_path

        else:
            if not hasattr(self, "hdf_path"):
                raise AttributeError("Path to target .hdf5 file " "is not set or given")

            else:
                hdf_path = self.hdf_path

        self._convert(hdf_path, odb_path)

    @classmethod
    def convert_by_path(cls, hdf_path: pathlib.Path, odb_path: pathlib.Path) -> None:
        hdf_path = pathlib.Path(hdf_path)
        odb_path = pathlib.Path(odb_path)
        cls()._convert(hdf_path, odb_path)

    def _convert(self, hdf_path: pathlib.Path, odb_path: pathlib.Path) -> None:
        convert_pickle_input_dict: Dict[str, Optional[Union[List[str], List[int]]]] = {
            "cpus": self.cpus,
            "nodes": self.nodes,
            "nodesets": self.nodesets,
            "time_step": self.time_step,
            "parts": self.parts,
            "steps": self.steps,
            "coord_key": self.coord_key,
            "target_outputs": self.target_outputs,
            #"data_model": self._data_model,
        }

        pickle_file: BinaryIO
        with open(self._convert_pickle_path, "wb") as pickle_file:
            pickle.dump(convert_pickle_input_dict, pickle_file, protocol=2)

        odb_convert_args: List[Union[pathlib.Path, str]] = [
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

        except FileNotFoundError:
            raise FileNotFoundError(
                f"File {self._convert_result_path} was not found. See previous Python 2 errors"
            )

        pathlib.Path.unlink(self._convert_result_path)

        temp_low = self.temp_low if hasattr(self, "temp_low") else None
        temp_high = self.temp_high if hasattr(self, "temp_high") else None

        convert_npz_to_hdf(
            hdf_path,
            #self.data_model,
            result_dir,
            temp_low,
            temp_high,
            self.time_step,
            self.nodesets,
            self.nodes,
            self.parts,
            self.steps,
            self.coord_key,
            self.target_outputs,
            odb_path,
        )

        if result_dir.exists():
            shutil.rmtree(result_dir)

    @classmethod
    def extract_by_path(
        cls,
        path: pathlib.Path,
    ) -> DataFrameType:
        if ensure_magic(path, HDF_MAGIC_NUM):
            # Extract from .hdf5
            return cls.extract_from_hdf(path)

        elif ensure_magic(path, ODB_MAGIC_NUM):
            # extract from .odb
            return cls.extract_from_odb(path)

    def extract(self) -> DataFrameType:
        if hasattr(self, "hdf_path") or hasattr(self, "odb"):
            result: DataFrameType = self.extract_from_hdf()
            self._extracted_nodes = result
            return result

        elif hasattr(self, "odb_path"):
            result: DataFrameType = self.extract_from_odb()
            self._extracted_nodes = result
            return result

        else:
            raise AttributeError(
                "This Odb object does not have a .odb file or a .hdf5 file from which to extract"
            )

    def extract_from_odb(
        self, target_file: Optional[pathlib.Path] = None
    ) -> DataFrameType:
        if target_file is None:
            target_file = self.odb_path

        if target_file is None:
            raise ValueError("odb_path must be set to extract from .odb file")

        extract_odb_pickle_input_dict: Dict[
            str, Optional[Union[List[str], List[int]]]
        ] = {
            "cpus": self.cpus,
            "nodes": self.nodes,
            "nodesets": self.nodesets,
            "time_step": self.time_step,
            "parts": self.parts,
            "steps": self.steps,
            "coord_key": self.coord_key,
            "target_outputs": self.target_outputs,
        }

        temp_file: BinaryIO
        with open(self._extract_pickle_path, "wb") as temp_file:
            pickle.dump(extract_odb_pickle_input_dict, temp_file, protocol=2)

        args_list: List[Union[str, pathlib.Path]] = [
            self.abaqus_executable,
            "python",
            self._extract_script_path,
            target_file,
            self._extract_pickle_path,
            self._extract_result_path,
        ]

        # TODO
        # shell=True is bad practice, but abaqus python will not run without it.
        subprocess.run(args_list, shell=True)

        temp_file: TextIO
        try:
            with open(self._extract_result_path, "rb") as temp_file:
                # From the Pickle spec, decoding python 2 numpy arrays must use
                # "latin-1" encoding
                results: List[Dict[str, float]]
                results = pickle.load(temp_file, encoding="latin-1")

        except FileNotFoundError:
            raise FileNotFoundError(
                f"File {self._extract_result_path} was not found. See previous Python 2 errors"
            )

        results = sorted(results, key=lambda d: d["time"])

        results_df_list: List[DataFrameType] = list()
        for result in results:
            time = result.pop("time")
            results_df_list.append(
                pd.DataFrame.from_dict({time: result}, orient="index")
            )

        result_df: pd.DataFrame = pd.concat(results_df_list)

        if self._extract_result_path.exists():
            self._extract_result_path.unlink()

        return result_df

    def extract_from_hdf(
        self, target_file: Optional[pathlib.Path] = None
    ) -> DataFrameType:
        if target_file is not None:
            if hasattr(self, "odb"):
                raise AttributeError(
                    "Do not pass in a new path to an "
                    "existing Odb() object for extracting. Use the classmethod "
                    "instead."
                )

            else:
                self.hdf_path = target_file
                self.load_hdf()

        else:
            if not hasattr(self, "odb"):
                self.load_hdf()

        results: List[DataFrameType] = list()
        frame: DataFrameType
        for frame in self:
            time: float = frame["Time"].values[0]
            output: str
            frame_dict: Dict[int, Dict[str, float]] = {time: {}}
            chosen_outputs = (
                self.target_outputs
                if (hasattr(self, "target_outputs") and self.target_outputs is not None)
                else frame.keys()
            )
            for output in chosen_outputs:
                output_data: DataFrameType = frame[output].values
                if output in ("NT11",):
                    output_data = output_data[output_data != 300.0]
                    output_data = output_data[output_data != 0.0]
                output_data = output_data[output_data != np.nan]
                min_val: float = np.min(output_data) if len(output_data) > 0 else np.nan
                max_val: float = np.max(output_data) if len(output_data) > 0 else np.nan
                mean_val: float = (
                    np.mean(output_data) if len(output_data) > 0 else np.nan
                )
                frame_dict[time][f"{output}_min"] = min_val
                frame_dict[time][f"{output}_max"] = max_val
                frame_dict[time][f"{output}_mean"] = mean_val

            results.append(pd.DataFrame.from_dict(frame_dict, orient="index"))

        results_df: pd.DataFrame = pd.concat(results)

        return results_df

    def get_odb_info(self) -> None:
        # Ideally this would not work this way, but
        # the python2 makes transferring a raw dict the easiest option
        result: Dict[
            str, Union[Tuple[int, int], List[str], Dict[str, Tuple[int, int]]]
        ] = self._get_odb_info()
        # No setters for these, just this method
        self._frame_range = result["frame_range"]
        self._frame_keys = result["frame_keys"]
        self._frame_keys_per_step = result["frame_keys_per_step"]
        self._step_names = result["step_names"]
        self._step_lens = result["step_lens"]
        self._nodeset_names = result["nodeset_names"]
        self._part_names = result["part_names"]
        self._node_range = result["node_range"]
        self._node_ranges_per_part = result["node_ranges_per_part"]

        if hasattr(self, "hdf_path"):
            hdf5_file: H5PYFileType
            with h5py.File(self.hdf_path, "r+") as hdf5_file:
                total_name: str = str(self.hdf_path.stem)
                hdf5_file[total_name].attrs["frame_range"] = self._frame_range
                hdf5_file[total_name].attrs["frame_keys"] = self._frame_keys
                for step, frame_keys in self._frame_keys_per_step.items():
                    hdf5_file[total_name].attrs[f"frame_keys_per_{step}"] = frame_keys
                hdf5_file[total_name].attrs["step_names"] = self._step_names
                for step, length in self._step_lens.items():
                    hdf5_file[total_name].attrs[f"step_{step}_length"] = length
                hdf5_file[total_name].attrs["nodeset_names"] = self._nodeset_names
                hdf5_file[total_name].attrs["part_names"] = self._part_names
                hdf5_file[total_name].attrs["node_range"] = self._node_range
                for part, node_range in self._node_ranges_per_part.items():
                    hdf5_file[total_name].attrs[f"node_ranges_per_{part}"] = node_range

    @classmethod
    def get_odb_info_from_file(
        cls, path: pathlib.Path
    ) -> "Dict[str, Union[Tuple[int, int], List[str], Dict[str, Tuple[int, int]]]]":
        return cls()._get_odb_info(path)

    def _get_odb_info(
        self, path: Optional[pathlib.Path] = None
    ) -> "Dict[str, Union[Tuple[int, int], List[str], Dict[str, Tuple[int, int]]]]":
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
                self._get_odb_info_script_path,
                path,
                self._get_odb_info_result_path,
            ],
            shell=True,
        )

        result_file: TextIO
        try:
            with open(self._get_odb_info_result_path, "rb") as result_file:
                return pickle.load(result_file)

        except FileNotFoundError:
            raise FileNotFoundError(
                f"File {self._get_odb_info_result_path} was not found. See previous Python 2 errors"
            )

    def load_hdf(self) -> None:
        if not hasattr(self, "hdf_path"):
            raise AttributeError(
                "hdf_path attribute must be set before " "calling load_hdf method."
            )

        try:
            # Only case where this should be set, bypass the setter
            self._hdf_status, self._odb = self._odb_handler.load_hdf(
                self.hdf_path, self.cpus
            )
            self._odb_handler = OdbUnloader()
            self._times = np.sort(self["Time"].unique())

        except AttributeError as e:
            raise AttributeError(
                "load_hdf can only be used once in a row, "
                "before a .hdf5 file is loaded. Call unload_hdf on this "
                "OdbViewer before calling load_hdf."
            ) from e

    def unload_hdf(self) -> None:
        try:
            self._odb_handler.unload_hdf()
            # Unified deleter
            del self.odb
            self._odb_handler = OdbLoader()

        except AttributeError:
            raise AttributeError("unload_hdf can only be called after " "load_hdf.")

    # 2D Plotting

    # Structure of the dataframe:
    # Times are repeated
    # We'll assume, by default, that you want (one of) min, max, mean of duplicated time

    def plot_key_versus_time(
        self,
        target_output: str,
        mean_max_both: str = "both",
        title: Optional[str] = None,
    ) -> "Optional[pathlib.Path]":
        # TODO What if I want to 2d-plot only 1 nodeset, but I extractor more stuff
        # or DIDN'T extract the nodeset at all. Same w/ 3D. Metadata?

        if not PYVISTA_AVAILABLE:
            raise Exception(
                "Plotting cabailities are not included."
                ' Please install pyvista via pip install odb-plotter["plot"]'
                ' or odb-plotter["all"] rather than pip install odb-plotter'
                " Or export the data from Odb.extract() to another tool,"
                " such as matplotlib, plotly, or bokeh."
            )

        if not hasattr(self, "_extracted_nodes"):
            _ = self.extract()

        target_data = self._extracted_nodes[
            (self.time_low <= self._extracted_nodes["Time_mean"])
            & (self._extracted_nodes["Time_mean"] <= self.time_high)
        ]
        time_data: List[float] = list(target_data.index)

        title = (
            title
            if title is not None
            else self.hdf_path.stem
            if hasattr(self, "hdf_path")
            else self.odb_path.stem
        )
        title += f" {target_output} versus Time"

        temp_v_time: pv.Chart2D = pv.Chart2D()
        #    x_label="Time (seconds)", y_label="Temperature (Kelvin)"
        #)
        #temp_v_time.title = title

        if mean_max_both.lower() in ("mean", "both"):
            temp_v_time.line(
                time_data,
                target_data[f"{target_output}_mean"].values,
                color="#0000FF",  # TODO param
                #label=f"Mean {target_output}",
                width=5.0
            )

        if mean_max_both.lower() in ("max", "both"):
            temp_v_time.line(
                time_data,
                target_data[f"{target_output}_max"].values,
                color="#FF0000",  # TODO param
                #label=f"Max {target_output}",
                width=5.0
            )

        screenshot: Union[bool, pathlib.Path] = (
            self.result_dir / f"{title}.png" if self.save else False
        )
        if self.save:
            if not self.result_dir.exists():
                self.result_dir.mkdir()

        save_path: pathlib.Path = (
            self.result_dir / f"{mean_max_both + '_'}{target_output + '_'}{title}.png"
        )
        temp_v_time.show(interactive=True, off_screen=False, screenshot=screenshot)

        if self.save:
            return save_path

    def plot_single_node(
        self, target_output: str, node: int, title: Optional[str] = None
    ) -> "Optional[pathlib.Path]":
        if not PYVISTA_AVAILABLE:
            raise Exception(
                "Plotting cabailities are not included."
                ' Please install pyvista via pip install odb-plotter["plot"]'
                ' or odb-plotter["all"] rather than pip install odb-plotter'
                " Or export the data from Odb.extract() to another tool,"
                " such as matplotlib, plotly, or bokeh."
            )

        if not hasattr(self, "_odb"):
            self.load_hdf()

        node_vals = self.odb[self.odb["Node Label"] == node]

        title = (
            title
            if title is not None
            else self.hdf_path.stem
            if hasattr(self, "hdf_path")
            else self.odb_path.stem
        )
        title += f" {target_output} versus Time for Node {node}"

        temp_v_time: pv.Chart2D = pv.Chart2D()
        #    x_label="Time (seconds)", y_label="Temperature (Kelvin)"
        #)
        #temp_v_time.title = title

        data_to_plot = node_vals.drop(
            columns=list(set(node_vals.keys()) - set(("Time", target_output)))
        )
        data_to_plot = data_to_plot[
            (self.time_low <= data_to_plot["Time"])
            & (data_to_plot["Time"] <= self.time_high)
        ]
        data_to_plot = data_to_plot.sort_values(by="Time", ascending=True)
        temp_v_time.line(
            data_to_plot["Time"],
            data_to_plot[target_output],
            color="#FF0000",  # TODO param
            #label=f"{target_output} per time for Node {node}",
            width=5.0
        )

        screenshot: Union[bool, pathlib.Path] = (
            self.result_dir / f"{target_output}_Node_{node}_{title}.png"
            if self.save
            else False
        )
        if self.save:
            if not self.result_dir.exists():
                self.result_dir.mkdir()

        save_path: pathlib.Path = self.result_dir / f"{title}.png"
        temp_v_time.show(interactive=True, off_screen=False, screenshot=screenshot)

        if self.save:
            return save_path

    # 3D Plotting
    def plot_3d_all_times(
        self,
        target_output: str,
        *,
        title: "Optional[str]" = None,
        target_nodes: "Optional[DataFrameType]" = None,
        plot_type: "Optional[str]" = None,
    ) -> "List[pathlib.Path]":
        """ """
        if not PYVISTA_AVAILABLE:
            raise Exception(
                "Plotting cabailities are not included."
                ' Please install pyvista via pip install odb-plotter["plot"]'
                ' or odb-plotter["all"] rather than pip install odb-plotter'
                " Or export the data from Odb.extract()",
                " or Odb.convert() to another tool,"
                " such as matplotlib, plotly, or bokeh.",
            )

        title = self.title
        title = self.hdf_path.stem if (title is None or not title) else title

        if target_nodes is None:
            if not hasattr(self, "odb"):
                self.load_hdf()

            target_nodes = self.odb

        if not self.result_dir.exists():
            self.result_dir.mkdir()

        target_times = target_nodes["Time"].unique()
        # There should be more elegant ways to do this, but np.where was misbehaving, and this works fine
        target_times = target_times[target_times >= self.time_low]
        target_times = target_times[target_times <= self.time_high]
        if self.interactive:
            results = []
            for time in target_times:
                results.append(
                    self._plot_3d_single(
                        time, title, target_output, target_nodes, plot_type
                    )
                )

        else:
            with multiprocessing.Pool(processes=self.cpus) as pool:
                results = pool.starmap(self._plot_3d_single, ((time, title, target_output, target_nodes, plot_type) for time in target_times))

        #time: float
        #for time in np.sort(target_nodes["Time"].unique()):
        #    if self.time_low <= time <= self.time_high:
        #        results.append(
        #            self._plot_3d_single(
        #                time, title, target_output, target_nodes, plot_type
        #            )
        #        )

        return results

    def _plot_3d_single(
        self,
        time: float,
        title: str,
        target_output: str,
        target_nodes: DataFrameType,
        plot_type: "Optional[str]",
    ) -> "Optional[pathlib.Path]":
        """ """
        if not PYVISTA_AVAILABLE:
            raise Exception(
                "Plotting cabailities are not included."
                ' Please install pyvista via pip install odb-plotter["plot"]'
                ' or odb-plotter["all"] rather than pip install odb-plotter'
                " Or export the data from Odb.extract() to another tool,"
                " such as matplotlib, plotly,  or bokeh."
            )

        combined_label: str = f"{title}-{round(time, 2):.2f}"


        plotter: pv.Plotter = pv.Plotter(
            off_screen=(not self.interactive), window_size=(1920, 1080)
        )
        plotter.add_light(pv.Light(light_type="headlight"))

        plotter.add_text(
            combined_label, position="upper_edge", color=self.font_color, font=self.font, font_size=self.font_size
        )

        mesh: pv.PolyData = self.get_mesh(time, target_nodes, target_output)

        epsilon: float = np.finfo(float).eps
        plotter.add_mesh(
            mesh,
            scalars=target_output,
            cmap=pv.LookupTable(
                cmap=self._colormap,
                # Handle Epsilon
                scalar_range=(self.temp_low + epsilon, self.temp_high - epsilon),
                above_range_color=self.above_range_color,
                below_range_color=self.below_range_color,
            ),
            scalar_bar_args={
                "vertical": True,
                "title": "Nodal Temperature (Kelvin)", # TODO
                "font_family": self.font,
                "title_font_size": self.font_size + 4,
                "label_font_size": self.font_size,
                "color": self.font_color,
                "fmt": "%.2f",
                "position_x": 0.05,
                "position_y": 0.05,
            },
        )

        #plotter.show_bounds(
        #    location="outer",
        #    font_size=self.font_size,
        #    font_family=self.font,
        #    color=self.font_color,
        #    n_zlabels=3,
        #    fmt="%.2f",
        #)

        if self.show_axes:
            # TODO Dynamically update these
            x_low, x_high, y_low, y_high, z_low, z_high = mesh.bounds

            x_pad = (x_high - x_low) / 4.0
            y_pad = (y_high - y_low) / 4.0
            z_pad = (z_high - z_low) / 4.0
            pads = [x_pad, y_pad, z_pad]
            pads.sort()
            pad = pads[1]

            ruler_x = plotter.add_ruler(
                pointa=(x_low, y_high + pad, z_low - pad),
                pointb=(x_high, y_high + pad, z_low - pad),
                label_format="%.2f",
                font_size_factor=0.4,
                label_color=self.axis_text_color,
                title="X Axis"
            )
            ruler_x.SetRange(x_low, x_high)

            ruler_y = plotter.add_ruler(
                pointa=(x_high + pad, y_low, z_low - pad),
                pointb=(x_high + pad, y_high, z_low - pad),
                label_format="%.2f",
                font_size_factor=0.4,
                label_color=self.axis_text_color,
                title="Y Axis"
            )
            ruler_y.SetRange(y_low, y_high)

            ruler_z = plotter.add_ruler(
                pointa=(x_high + pad, y_low - pad, z_low),
                pointb=(x_high + pad, y_low - pad, z_high),
                label_format="%.2f",
                font_size_factor=0.4,
                label_color=self.axis_text_color,
                title="Z Axis"
            )
            ruler_z.SetRange(z_low, z_high)

        plotter.set_background(color=self.background_color)

        invalid_view = True
        for k in self._views.keys():
            if self.view in k:
                invalid_view = False
                view_angle, viewup, roll = self._views[k]
                break

        if invalid_view:
            raise RuntimeError("View Panic")
        plotter.view_vector(view_angle, viewup=viewup)
        plotter.camera.roll = roll

        if not self.save:
            plotter.show(interactive_update=True)
        else:
            plotter.show(
                before_close_callback=lambda p: p.screenshot(self.result_dir / f"{plot_type + '_' if plot_type is not None else ''}{combined_label}{self.save_format}")
                )

        return

    def get_odb_state(self) -> str:
        return self.get_odb_settings_state()

    def get_mesh(self, time, target=None, output=None) -> pv.PolyData:
        if target is None:
            target = self.odb
        dims_columns: set[str] = {"X", "Y", "Z"}
        filtered_target_nodes: DataFrameType = target[
            target["Time"] == time
        ]
        filtered_target_nodes = filtered_target_nodes[
            filtered_target_nodes["X"] >= self.x_low
        ]
        filtered_target_nodes = filtered_target_nodes[
            filtered_target_nodes["X"] <= self.x_high
        ]
        filtered_target_nodes = filtered_target_nodes[
            filtered_target_nodes["Y"] >= self.y_low
        ]
        filtered_target_nodes = filtered_target_nodes[
            filtered_target_nodes["Y"] <= self.y_high
        ]
        filtered_target_nodes = filtered_target_nodes[
            filtered_target_nodes["Z"] >= self.z_low
        ]
        filtered_target_nodes = filtered_target_nodes[
            filtered_target_nodes["Z"] <= self.z_high
        ]
        points: pv.PolyData = pv.PolyData(
            filtered_target_nodes.drop(
                columns=list(
                    set(filtered_target_nodes.columns.values.tolist()) - dims_columns
                )
            ).to_numpy()
        )

        if output is not None:
            points[output] = filtered_target_nodes[output].to_numpy()

        return points.delaunay_3d()

class OdbLoader:
    def load_hdf(
        self, hdf_path: pathlib.Path, cpus: int
    ) -> "Tuple[Dict[str, str], DataFrameType]":
        return get_odb_data(hdf_path, cpus)


class OdbUnloader:
    @abstractmethod
    def unload_hdf(self) -> None:
        pass
