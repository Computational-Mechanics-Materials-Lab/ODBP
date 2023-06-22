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
import multiprocessing

import numpy as np
import pandas as pd

from typing import TextIO, Union, Any, Tuple, List, Dict, Optional, Iterator
from abc import abstractmethod

from .writer import convert_npz_to_hdf
from .reader import get_odb_data
from .util import NullableIntList, NullableStrList, DataFrameType,\
    NDArrayType, NullableNodeType, NodeType, MultiprocessingPoolType,\
    ODB_MAGIC_NUM, HDF_MAGIC_NUM

try:
    import pyvista as pv
except ImportError:
    PYVISTA_AVAILABLE = False
else:
    PYVISTA_AVAILABLE = True


class Odb():
    """
    Stores Data from a .hdf5, implements extractor methods to transfer from .odb to .hdf5
    Implements abilities to resize the dimenisons or timeframe of the data
    """

    # TODO user settings sub-section, overload getattr
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

        # User-defined components
        # TODO turn these into a subclass member, overload
        # __getattr__/__setattr__ on this object
        "_x_low",
        "_x_high",
        "_y_low",
        "_y_high",
        "_z_low",
        "_z_high",
        "_temp_low",
        "_temp_high",
        "_time_low",
        "_time_high",
        "_odb_path",
        "_odb_source_dir",
        "_hdf_path",
        "_hdf_source_dir",
        "_result_dir",
        "_abaqus_executable",
        "_cpus",
        "_nodesets",
        "_frames",
        "_nodes",
        "_parts",
        "_steps",
        "_coord_key",
        "_temp_key",
        "_interactive",
        "_angle",
        "_colormap",
        "_save",
        "_save_format",
        )


    def __init__(self) -> None:
        """
        Type Hints and hard-coded parameters. See the @staticmethod
        "constructors" of this class in order to learn about initialization
        """

        self._odb_handler: Union[OdbLoader, OdbUnloader] = OdbLoader()
        self._odb: DataFrameType 

        self._odb_path: pathlib.Path
        self._odb_source_dir: Optional[pathlib.Path]
        self._odb_source_dir = pathlib.Path.cwd().absolute() / "odbs"

        self._hdf_path: pathlib.Path
        self._hdf_source_dir: Optional[pathlib.Path]
        self._hdf_source_dir = pathlib.Path.cwd().absolute() / "hdfs"

        self._result_dir: Optional[pathlib.Path]
        self._result_dir = pathlib.Path.cwd().absolute() / "results"

        self._abaqus_executable: str = "abaqus"

        self._nodes: NullableNodeType = None
        self._nodesets: NullableStrList = None
        self._frames: NullableIntList = None
        self._parts: NullableStrList = None
        self._steps: NullableStrList = None

        self._coord_key: str = "COORD"
        self._temp_key: str = "NT11"

        self._x_low: float
        self._x_high: float
        self._y_low: float
        self._y_high: float
        self._z_low: float
        self._z_high: float

        self._temp_low: float
        self._temp_high: float

        self._time_low: float
        self._time_high: float

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

        self._cpus = multiprocessing.cpu_count()

        self._interactive: bool = False
        self._colormap: str = "turbo"
        self._save_format: str = ".png"
        self._save: bool = True

        self._extracted_nodes: DataFrameType

        # TODO
        self._angle = Union[str, Tuple[float, float, float]]

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
    def x_low(self) -> float:
        return self._x_low


    @x_low.setter
    def x_low(self, value: float) -> None:
        if not isinstance(value, float):
            try:
                # Handle str/int input
                value = float(value)
            except ValueError:
                raise ValueError("x_low must be a float")

        # Don't let users set improper dimensions
        if hasattr(self, "x_high"): # If high is set
            if value > self.x_high:
                raise ValueError(f"The value for x_low ({value}) must not be greater than the value for x_high ({self.x_high})")

        self._x_low = value


    @property
    def x_high(self) -> float:
        return self._x_high


    @x_high.setter
    def x_high(self, value: float) -> None:
        if not isinstance(value, float):
            try:
                # Handle str/int input
                value = float(value)
            except ValueError:
                raise ValueError("x_high must be a float")

        # Don't let users set improper dimensions
        if hasattr(self, "x_low"): # If low is set
            if value < self.x_low:
                raise ValueError(f"The value for x_high ({value}) must not be less than the value for x_low ({self.x_low})")

        self._x_high = value


    @property
    def y_low(self) -> float:
        return self._y_low


    @y_low.setter
    def y_low(self, value: float) -> None:
        if not isinstance(value, float):
            try:
                # Handle str/int input
                value = float(value)
            except ValueError:
                raise ValueError("y_low must be a float")

        # Don't let users set improper dimensions
        if hasattr(self, "y_high"): # If high is set
            if value > self.y_high:
                raise ValueError(f"The value for y_low ({value}) must not be greater than the value for y_high ({self.y_high})")

        self._y_low = value


    @property
    def y_high(self) -> float:
        return self._y_high


    @y_high.setter
    def y_high(self, value: float) -> None:
        if not isinstance(value, float):
            try:
                # Handle str/int input
                value = float(value)
            except ValueError:
                raise ValueError("y_high must be a float")

        # Don't let users set improper dimensions
        if hasattr(self, "y_low"): # If low is set
            if value < self.y_low:
                raise ValueError(f"The value for y_high ({value}) must not be less than the value for y_low ({self.y_low})")

        self._y_high = value


    @property
    def z_low(self) -> float:
        return self._z_low


    @z_low.setter
    def z_low(self, value: float) -> None:
        if not isinstance(value, float):
            try:
                # Handle str/int input
                value = float(value)
            except ValueError:
                raise ValueError("z_low must be a float")

        # Don't let users set improper dimensions
        if hasattr(self, "z_high"): # If high is set
            if value > self.z_high:
                raise ValueError(f"The value for z_low ({value}) must not be greater than the value for z_high ({self.z_high})")

        self._z_low = value


    @property
    def z_high(self) -> float:
        return self._z_high


    @z_high.setter
    def z_high(self, value: float) -> None:
        if not isinstance(value, float):
            try:
                # Handle str/int input
                value = float(value)
            except ValueError:
                raise ValueError("z_high must be a float")

        # Don't let users set improper dimensions
        if hasattr(self, "z_low"): # If low is set
            if value < self.z_low:
                raise ValueError(f"The value for z_high ({value}) must not be less than the value for z_low ({self.z_low})")

        self._z_high = value


    @property
    def temp_low(self) -> float:
        return self._temp_low


    @temp_low.setter
    def temp_low(self, value: float) -> None:
        if not isinstance(value, float):
            try:
                # Handle str/int input
                value = float(value)
            except ValueError:
                raise ValueError("temp_low must be a float")

        if value < 0:
            raise ValueError("temp_low must be greater than or equal to 0 (Kelvins)")

        if hasattr(self, "temp_high"):
            if value > self.temp_high:
                raise ValueError(f"The value for temp_low ({value}) must not be greater than the value for temp_high ({self.temp_high})")

        self._temp_low = value


    @property
    def temp_high(self) -> float:
        return self._temp_high


    @temp_high.setter
    def temp_high(self, value: float) -> None:
        if not isinstance(value, float):
            try:
                # Handle str/int input
                value = float(value)
            except ValueError:
                raise ValueError("temp_high must be a float")

        if hasattr(self, "temp_low"):
            if value < self.temp_low:
                raise ValueError(f"The value for temp_high ({value}) must not be less than the value for temp_low ({self.temp_low})")

        self._temp_high = value


    @property
    def time_low(self) -> float:
        return self._time_low


    @time_low.setter
    def time_low(self, value: float) -> None:
        if not isinstance(value, float):
            try:
                # Handle str/int input
                value = float(value)
            except ValueError:
                raise ValueError("time_low must be a float")

        if value < 0:
            raise ValueError("time_low must be greater than or equal to 0 (Kelvins)")

        if hasattr(self, "time_high"):
            if value > self.time_high:
                raise ValueError(f"The value for time_low ({value}) must not be greater than the value for time_high ({self.time_high})")

        self._time_low = value


    @property
    def time_high(self) -> float:
        return self._time_high


    @time_high.setter
    def time_high(self, value: float) -> None:
        if not isinstance(value, float):
            try:
                # Handle str/int input
                value = float(value)
            except ValueError:
                raise ValueError("time_high must be a float")

        if hasattr(self, "time_low"):
            if value < self.time_low:
                raise ValueError(f"The value for time_high ({value}) must not be less than the value for time_low ({self.time_low})")

        self._time_high = value


    @property
    def odb_source_dir(self) -> "Optional[pathlib.Path]":
        return self._odb_source_dir


    @odb_source_dir.setter
    def odb_source_dir(self, value: pathlib.Path) -> None:
        value = pathlib.Path(value)

        if not value.exists():
            raise FileNotFoundError(f"Directory {value} does not exist")

        self._odb_source_dir = value


    @property
    def odb_path(self) -> "Optional[pathlib.Path]":
        return self._odb_path


    @staticmethod
    def ensure_magic(file_path: pathlib.Path, magic: bytes) -> bool:
        file: TextIO
        first_line: bytes
        with open(file_path, "rb") as file:
            first_line = file.readline()

        return first_line.startswith(magic)


    @odb_path.setter
    def odb_path(self, value: pathlib.Path) -> None:
        value = pathlib.Path(value)

        target_path: pathlib.Path
        cwd_path: pathlib.Path = pathlib.Path.cwd() / value
        if value.exists():
            target_path = value

        elif self.odb_source_dir is not None:
            source_dir_path: pathlib.Path = self.odb_source_dir / value
            if source_dir_path.exists():
                target_path = source_dir_path

        elif cwd_path.exists():
            target_path = cwd_path

        else:
            raise FileNotFoundError(f"File {value} could not be found")

        # Ensure magic numbers
        if not self.ensure_magic(target_path, ODB_MAGIC_NUM):
            raise ValueError(f"Given file {value} is not a .odb object"
            "database file.")

        self._odb_path = target_path


    @property
    def hdf_source_dir(self) -> "Optional[pathlib.Path]":
        return self._hdf_source_dir


    @hdf_source_dir.setter
    def hdf_source_dir(self, value: pathlib.Path) -> None:
        value = pathlib.Path(value)

        if not value.exists():
            raise FileNotFoundError(f"Directory {value} does not exist")

        self._hdf_source_dir = value


    @property
    def hdf_path(self) -> pathlib.Path:
        return self._hdf_path


    @hdf_path.setter
    def hdf_path(self, value: pathlib.Path) -> None:
        value = pathlib.Path(value)

        if not value.is_absolute() and hasattr(self, "hdf_source_dir"):
            value = self.hdf_source_dir / value

        if value.exists():
            # Ensure magic numbers
            if not self.ensure_magic(value, HDF_MAGIC_NUM):
                raise ValueError(f"Given file {value} is not a .hdf5 hierarchical"
                "data format file.")

        self._hdf_path = value

    
    @property
    def result_dir(self) -> pathlib.Path:
        return self._result_dir


    @result_dir.setter
    def result_dir(self, value: pathlib.Path) -> None:
        value = pathlib.Path(value)

        if not value.exists():
            value.mkdir()

        self._result_dir = value


    @property
    def abaqus_executable(self) -> str:
        return self._abaqus_executable


    @abaqus_executable.setter
    def abaqus_executable(self, value: str) -> None:
        self._abaqus_executable = value


    @property
    def cpus(self) -> int:
        return self._cpus


    @cpus.setter
    def cpus(self, value: int) -> None:
        assert value > 0
        self._cpus = value


    @property
    def nodes(self) -> NullableNodeType:
        return self._nodes


    @nodes.setter
    def nodes(self, value: NodeType) -> None:
        self._nodes = value


    @property
    def nodesets(self) -> NullableStrList:
        return self._nodesets


    @nodesets.setter
    def nodesets(self, value: "List[str]") -> None:
        self._nodesets = value


    @property
    def parts(self) -> NullableStrList:
        return self._parts


    @parts.setter
    def parts(self, value: "List[str]") -> None:
        self._parts = value
    

    @property
    def steps(self) -> NullableStrList:
        return self._steps

    
    @steps.setter
    def steps(self, value: "List[str]") -> None:
        self._steps = value


    @property
    def frames(self) -> NullableIntList:
        return self._frames


    @frames.setter
    def frames(self, value: "List[int]") -> None:
        self._frames = value


    @property
    def coord_key(self) -> str:
        return self._coord_key


    @coord_key.setter
    def coord_key(self, value: str) -> None:
        self._coord_key = value


    @property
    def temp_key(self) -> str:
        return self._temp_key


    @temp_key.setter
    def temp_key(self, value: str) -> None:
        self._temp_key = value


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


    @property
    def colormap(self) -> str:
        return self._colormap


    @colormap.setter
    def colormap(self, value: str) -> None:
        self._colormap = value


    @property
    def interactive(self) -> bool:
        return self._interactive


    @interactive.setter
    def interactive(self, value: bool) -> None:
        self._interactive = value


    @property
    def save_format(self) -> str:
        return self._save_format


    @save_format.setter
    def save_format(self, value: str) -> None:
        if not value.startswith("."):
            value = "." + value

        self._save_format = value


    @property
    def save(self) -> bool:
        return self._save


    @save.setter
    def save(self, value: bool) -> None:
        self._save = value


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
                "parts": self.parts,
                "steps": self.steps,
                "coord_key": self._coord_key,
                "temp_key": self._temp_key
            }

        pickle_file: TextIO
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

        result_file: TextIO
        result_dir: pathlib.Path
        with open(self._convert_result_path, "rb") as result_file:
            result_dir = pathlib.Path(pickle.load(result_file))

        pathlib.Path.unlink(self._npz_result_path)

        convert_npz_to_hdf(hdf_path, result_dir)

        if result_dir.exists():
            shutil.rmtree(result_dir)


    @classmethod
    def extract_by_path(
            cls,
            path: pathlib.Path,
            ) -> DataFrameType:

        if cls.ensure_magic(path, HDF_MAGIC_NUM):
            # Extract from .hdf5
            return cls.extract_from_hdf(path)

        elif cls.ensure_magic(path, ODB_MAGIC_NUM):
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
                "parts": self.parts,
                "steps": self.steps,
                "temp_key": self.temp_key,
                "cpus": self.cpus
            }
            
        temp_file: TextIO
        with open(self._extract_pickle_path, "wb") as temp_file:
            pickle.dump(
                extract_odb_pickle_input_dict,
                temp_file,
                protocol=2
                )

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
        with open(self._extract_result_path, "rb") as temp_file:
            # From the Pickle spec, decoding python 2 numpy arrays must use
            # "latin-1" encoding
            results: List[Dict[str, float]]
            results = pickle.load(temp_file, encoding="latin-1")

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
            mean: float = np.mean(temp_vals) if len(temp_vals) > 0 else np.nan
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
        with open(self._collect_state_result_path, "rb") as result_file:
            return pickle.load(result_file)


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
                interactive=self.interactive,
                off_screen=(not self.interactive),
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
                " Or export the data from Odb.extract() to another tool,"
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
            off_screen=(not self.interactive),
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

        if self.interactive:
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