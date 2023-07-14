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
import sys

import numpy as np
import pandas as pd

from typing import TextIO, Union, Any, Tuple, List, Dict, Optional, Iterator,\
    BinaryIO
from abc import abstractmethod

from .odb_settings import OdbSettings
from .writer import convert_npz_to_hdf
from .reader import get_odb_data
from .types import DataFrameType, NDArrayType
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
        "_convert_script_path",
        "_convert_pickle_path",
        "_convert_result_path",
        "_extract_script_path",
        "_extract_pickle_path",
        "_extract_result_path",
        "_collect_state_script_path",
        "_collect_state_result_path",
        "_iterator_ind",
        "_times",
        "_frame_keys",
        "_frame_range",
        "_step_names",
        "_frame_ranges_per_step",
        "_nodeset_names",
        "_part_names",
        "_node_range",
        "_node_ranges_per_part",
        "_extracted_nodes",
        )


    def __init__(self) -> None:
        """
        Type Hints and hard-coded parameters. See the @staticmethod
        "constructors" of this class in order to learn about initialization
        """

        self._odb_handler: Union[OdbLoader, OdbUnloader] = OdbLoader()
        self._odb: DataFrameType 

        # Hardcoded paths for Python 3 - 2 communication
        self._convert_script_path: pathlib.Path = pathlib.Path(
            pathlib.Path(__file__).parent,
            "py2_scripts",
            "converter.py"
        ).absolute()

        self._extract_script_path: pathlib.Path = pathlib.Path(
            pathlib.Path(__file__).parent,
            "py2_scripts",
            "extractor.py"
        ).absolute()

        self._convert_pickle_path: pathlib.Path = pathlib.Path(
            pathlib.Path.cwd().absolute(),
            "convert.pickle"
        )

        self._extract_pickle_path: pathlib.Path = pathlib.Path(
            pathlib.Path.cwd().absolute(),
            "extract_from_odb.pickle"
        )

        self._convert_result_path: pathlib.Path = pathlib.Path(
            pathlib.Path.cwd().absolute(),
            "convert_result.pickle"
        )

        self._extract_result_path: pathlib.Path = pathlib.Path(
            pathlib.Path.cwd().absolute(),
            "extract_results.pickle"
        )

        self._collect_state_script_path: pathlib.Path = pathlib.Path(
            pathlib.Path(__file__).parent,
            "py2_scripts",
            "state_collector.py"
        ).absolute()

        self._collect_state_result_path: pathlib.Path = pathlib.Path(
            pathlib.Path.cwd(),
            "collect_state_result.pickle"
        ).absolute()

        self._frame_keys: List[str]
        self._frame_range: Tuple[int, int]
        self._step_names: List[str]
        self._frame_ranges_per_step: Dict[str, Tuple[int, int]]
        self._nodeset_names: List[str]
        self._part_names: List[str]
        self._node_range: Tuple[int, int]
        self._node_ranges_per_part: Dict[str, Tuple[int, int]]

        self._extracted_nodes: DataFrameType

        self._iterator_ind: int = 0
        self._times: NDArrayType


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
            raise AttributeError("Odb() object only functions as an iterator"
                                 "After load_hdf() has been called.")

    def __getitem__(self, key: Any) -> Any:
        try:
            return self.odb[key]

        except AttributeError:
            raise AttributeError("Odb key access is not available before "
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
    def frame_ranges_per_step(self) -> "Dict[str, Tuple[int, int]]":
        return self._frame_ranges_per_step


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


    def convert(
            self,
            hdf_path: "Optional[pathlib.Path]" = None,
            *,
            odb_path: "Optional[pathlib.Path]" = None,
            set_odb: bool = False,
            set_hdf: bool = False
            ) -> None:

        if odb_path is not None:
            odb_path = pathlib.Path(odb_path)
            if set_odb:
                self.odb_path = odb_path

        else:
            if not hasattr(self, "odb_path"):
                raise AttributeError("Path to target .odb file "
                        "is not set or given.")

            else:
                odb_path = self.odb_path

        if hdf_path is not None:
            hdf_path = pathlib.Path(hdf_path)
            if set_hdf:
                self.hdf_path = hdf_path

        else:
            if not hasattr(self, "hdf_path"):
                raise AttributeError("Path to target .hdf5 file "
                        "is not set or given")

            else:
                hdf_path = self.hdf_path

        assert odb_path is not None
        self._convert(hdf_path, odb_path)


    @classmethod
    def convert_by_path(
            cls,
            hdf_path: pathlib.Path,
            odb_path: pathlib.Path
            ) -> None:
        hdf_path = pathlib.Path(hdf_path)
        odb_path = pathlib.Path(odb_path)
        cls()._convert(hdf_path, odb_path)


    def _convert(
            self,
            hdf_path: pathlib.Path,
            odb_path: pathlib.Path
            ) -> None:

        convert_pickle_input_dict: Dict[
            str, Optional[Union[List[str], List[int]]]
            ] = {
                "cpus": self.cpus,
                "nodes": self.nodes,
                "nodesets": self.nodesets,
                "frames": self.frames,
                "time_step": self.time_step,
                "parts": self.parts,
                "steps": self.steps,
                "coord_key": self._coord_key,
                "temp_key": self._temp_key
            }

        pickle_file: BinaryIO
        with open(
            self._convert_pickle_path, "wb") as pickle_file:
            pickle.dump(convert_pickle_input_dict, pickle_file, protocol=2)

        odb_convert_args: List[Union[pathlib.Path, str]]  = [
            self.abaqus_executable,
            "python",
            self._convert_script_path,
            odb_path,
            self._convert_pickle_path,
            self._convert_result_path
        ]

        # shell=True is BAD PRACTICE, but abaqus python won't run without it
        subprocess.run(odb_convert_args, shell=True)

        result_file: BinaryIO
        result_dir: pathlib.Path
        
        try:
            with open(self._convert_result_path, "rb") as result_file:
                result_dir = pathlib.Path(pickle.load(result_file))

        except FileNotFoundError:
            raise FileNotFoundError(f"File {self._convert_result_path} was not found. See previous Python 2 errors")

        pathlib.Path.unlink(self._convert_result_path)

        convert_npz_to_hdf(hdf_path, result_dir)

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
            raise AttributeError("This Odb object does not have")


    def extract_from_odb(
        self,
        target_file: Optional[pathlib.Path] = None
        )-> DataFrameType:

        if target_file is None:
            target_file = self.odb_path

        extract_odb_pickle_input_dict: Dict[
            str, Optional[Union[List[str], List[int]]]
            ] = {
                "nodes": self.nodes,
                "nodesets": self.nodesets,
                "frames": self.frames,
                "time_step": self.time_step,
                "parts": self.parts,
                "steps": self.steps,
                "temp_key": self.temp_key,
                "cpus": self.cpus
            }
            
        temp_file: BinaryIO
        with open(self._extract_pickle_path, "wb") as temp_file:
            pickle.dump(
                extract_odb_pickle_input_dict,
                temp_file,
                protocol=2
                )

        assert target_file is not None
        args_list: List[Union[str, pathlib.Path]] = [
            self.abaqus_executable,
            "python",
            self._extract_script_path,
            target_file,
            self._extract_pickle_path,
            self._extract_result_path
            ]

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
            raise FileNotFoundError(f"File {self._extract_result_path} was not found. See previous Python 2 errors")

        results = sorted(results, key=lambda d: d["time"])

        results_df_list: List[DataFrameType] = list()
        for result in results:
            time = result.pop("time")
            results_df_list.append(pd.DataFrame.from_dict({time: result}, orient="index"))

        result_df: pd.DataFrame = pd.concat(results_df_list)

        if self._extract_result_path.exists():
            self._extract_result_path.unlink()

        return result_df


    def extract_from_hdf(
        self,
        target_file: Optional[pathlib.Path] = None
        ) -> DataFrameType:

        if target_file is not None:
            if hasattr(self, "odb"):
                raise AttributeError("Do not pass in a new path to an "
                "existing Odb() object for extracting. Use the classmethod "
                "instead.")

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
            temp_df: DataFrameType = frame[frame["Temp"] != 300]
            temp_df = temp_df[temp_df["Temp"] != 0]
            temp_df = temp_df[temp_df["Temp"] != np.nan]
            temp_vals: NDArrayType = temp_df["Temp"].values
            min: float = np.min(temp_vals) if len(temp_vals) > 0 else np.nan
            max: float = np.max(temp_vals) if len(temp_vals) > 0 else np.nan
            mean: float = np.mean(temp_vals).values[0] if len(temp_vals) > 0 else np.nan
            results.append(pd.DataFrame.from_dict({time: {"max": max, "min": min, "mean": mean}}, orient="index"))

        results_df: pd.DataFrame = pd.concat(results)

        return results_df


    def collect_state(self) -> None:
        # Ideally this would not work this way, but
        # the python2 makes transferring a raw dict the easiest option
        result: Dict[
            str,
            Union[
                Tuple[int, int],
                List[str],
                Dict[str, Tuple[int, int]]
                ]
            ] = self._collect_state()
        # No setters for these, just this method
        self._frame_range = result["frame_range"]
        self._frame_keys = result["frame_keys"]
        self._step_names = result["step_names"]
        self._frame_ranges_per_step = result["frame_ranges_per_step"]
        self._nodeset_names = result["nodeset_names"]
        self._part_names = result["part_names"]
        self._node_range = result["node_range"]
        self._node_ranges_per_part = result["node_ranges_per_part"]


    @classmethod
    def collect_state_from_file(
        cls,
        path: pathlib.Path
        ) -> "Dict[str, Union[Tuple[int, int], List[str], Dict[str, Tuple[int, int]]]]":
        return cls()._collect_state(path)


    def _collect_state(
        self,
        path: Optional[pathlib.Path] = None
        ) -> "Dict[str, Union[Tuple[int, int], List[str], Dict[str, Tuple[int, int]]]]":
        if path is None:
            if hasattr(self, "odb_path"):
                path = self.odb_path

            else:
                raise AttributeError("Either pass in or set odb_path")

        # shell=True is bad practice, but abaqus python won't run without it
        subprocess.run([
            self.abaqus_executable,
            "python",
            self._collect_state_script_path,
            path,
            self._collect_state_result_path
            ], shell=True)

        result_file: TextIO
        try:
            with open(self._collect_state_result_path, "rb") as result_file:
                return pickle.load(result_file)

        except FileNotFoundError:
            raise FileNotFoundError(f"File {self._collect_state_result_path} was not found. See previous Python 2 errors")


    def load_hdf(self) -> None:

        if not hasattr(self, "hdf_path"):
            raise AttributeError("hdf_path attribute must be set before "
                    "calling load_hdf method.")

        try:
            # Only case where this should be set, bypass the setter
            self._odb = self._odb_handler.load_hdf(self.hdf_path, self.cpus)
            self._odb_handler = OdbUnloader()
            self._times = np.sort(self["Time"].unique())

        except AttributeError:
            raise AttributeError("load_hdf can only be used once in a row, "
                    "before a .hdf5 file is loaded. Call unload_hdf on this "
                    "OdbViewer before calling load_hdf.")


    def unload_hdf(self) -> None:

        try:
            self._odb_handler.unload_hdf()
            # Unified deleter
            del self.odb
            self._odb_handler = OdbLoader()

        except AttributeError:
            raise AttributeError("unload_hdf can only be called after "
                "load_hdf.")


    # 2D Plotting

    # Structure of the dataframe:
    # Times are repeated
    # We'll assume, by default, that you want (one of) min, max, mean of duplicated time

    def plot_temp_versus_time(
        self,
        mean_max_both: str = "both"
        ) -> Optional[pathlib.Path]:
        # TODO What if I want to 2d-plot only 1 nodeset, but I extractor more stuff
        # or DIDN'T extract the nodeset at all. Same w/ 3D. Metadata?

        if not PYVISTA_AVAILABLE:
            raise Exception("Plotting cabailities are not included."
                ' Please install pyvista via pip install odb-plotter["plot"]'
                ' or odb-plotter["all"] rather than pip install odb-plotter'
                " Or export the data from Odb.extract() to another tool,"
                " such as matplotlib or bokeh.")

        if not hasattr(self, "_extracted_nodes"):
            _ = self.extract()

        time_data: List[float] = list(self._extracted_nodes.index)

        title: str = self.hdf_path.stem if hasattr(self, "hdf_path") else self.odb_path.stem
        title += " Temperature versus Time"

        temp_v_time: pv.Chart2D = pv.Chart2D(x_label="Time (seconds)", y_label="Temperature (Kelvin)")
        temp_v_time.title = title

        if mean_max_both.lower() in ("mean", "both"):
            temp_v_time.line(
                time_data,
                self._extracted_nodes["mean"].values,
                color="#FF7F00",
                label="Mean Temperature")

        if mean_max_both.lower() in ("max", "both"):
            temp_v_time.line(
                time_data,
                self._extracted_nodes["max"].values,
                color="#FF0000",
                label="Max Temperature")

        if self.save:
            if not self.result_dir.exists():
                self.result_dir.mkdir()

            save_path: pathlib.Path = self.result_dir / f"{title}.png"
            temp_v_time.show(
                interactive=True,
                off_screen=False,
                screenshot=self.result_dir / f"{title}.png"
                )

            return save_path

        elif self.interactive:
            temp_v_time.show(
                interactive=True,
                off_screen=False,
                screenshot=False)


    # 3D Plotting
    def plot_3d_all_times(
            self,
            *,
            label: Optional[str] = None,
            target_nodes: Optional[DataFrameType] = None
            ) -> "List[pathlib.Path]":
        """

        """
        if not PYVISTA_AVAILABLE:
            raise Exception("Plotting cabailities are not included."
                ' Please install pyvista via pip install odb-plotter["plot"]'
                ' or odb-plotter["all"] rather than pip install odb-plotter'
                " Or export the data from Odb.extract()",
                " or Odb.convert() to another tool,"
                " such as matplotlib or bokeh.")

        label = self.hdf_path.stem if label is None else label

        if target_nodes is None:
            if not hasattr(self, "odb"):
                self.load_hdf()

            target_nodes = self.odb

        if not self.result_dir.exists():
            self.result_dir.mkdir()

        results: List[pathlib.Path] = list()
        frame: DataFrameType
        for frame in self:
            time: float = frame["Time"].values[0]
            if self.time_low <= time <= self.time_high:
                results.append(self._plot_3d_single(time, label, target_nodes))

        return results


    def _plot_3d_single(
        self,
        time: float,
        label: str,
        target_nodes: DataFrameType,
        )-> Optional[pathlib.Path]:
        """
        """

        if not PYVISTA_AVAILABLE:
            raise Exception("Plotting cabailities are not included."
                ' Please install pyvista via pip install odb-plotter["plot"]'
                ' or odb-plotter["all"] rather than pip install odb-plotter'
                " Or export the data from Odb.extract() to another tool,"
                " such as matplotlib or bokeh.")

        dims_columns: set[str] = {"X", "Y", "Z"}
        combined_label: str = f"{label}-{round(time, 2):.2f}"

        filtered_target_nodes: DataFrameType = target_nodes[
            target_nodes["Time"] == time
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

        plotter: pv.Plotter = pv.Plotter(
            off_screen=False,
            window_size=(1920, 1080),
            lighting="three lights"
            )

        plotter.add_text(
            combined_label,
            position="upper_edge",
            color="#000000",
            font="courier"
        )

        points: pv.PolyData = pv.PolyData(
            filtered_target_nodes.drop(
                columns=list(
                    set(filtered_target_nodes.columns.values.tolist())
                    - dims_columns
                    )
                ).to_numpy()
            )


        points["Temp"] = filtered_target_nodes["Temp"].to_numpy()
        mesh: pv.PolyData = points.delaunay_3d()

        plotter.add_mesh(
            mesh,
            scalars="Temp",
            cmap = pv.LookupTable(
                cmap=self._colormap,
                scalar_range=(
                    self.temp_low,
                    self.temp_high
                    ),
                above_range_color=(
                    0.75,
                    0.75,
                    0.75,
                    1.0
                )
            ),
            scalar_bar_args={
                "title": "Nodal Temperature (Kelvin)",
                "font_family": "courier",
                "color": "#000000",
                "fmt": "%.2f",
                "position_y": 0
            }
        )

        plotter.show_bounds(
            location="outer",
            ticks="both",
            font_size=14.0,
            font_family="courier",
            color="#000000",
            axes_ranges=points.bounds
            )

        plotter.set_background(color="#FFFFFF")

        # TODO
        #plotter.camera.elevation = 0
        #plotter.camera.azimuth = 270
        #plotter.camera.roll = 300
        #plotter.camera_set = True

        plotter.show()

        if self.save:
            final_name: str = f"{combined_label}{self.save_format}"
            plotter.screenshot(self.result_dir / final_name)
            return final_name

        return


class OdbLoader:

    def load_hdf(self, hdf_path: pathlib.Path, cpus: int) -> DataFrameType:
        return get_odb_data(hdf_path, cpus)


class OdbUnloader:

    @abstractmethod
    def unload_hdf(self) -> None:
        pass