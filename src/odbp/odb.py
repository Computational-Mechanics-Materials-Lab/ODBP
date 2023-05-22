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
import operator

import pandas as pd

from typing import TextIO, Union, Callable
from os import PathLike

from .npz_to_hdf import convert_npz_to_hdf
from .read_hdf5 import get_node_coords, get_node_times_temps
from .util import NullableIntListUnion, NullableStrListUnion, DataFrameType, MultiprocessingPoolType


class Odb():
    """
    Stores Data from a .hdf5, implements extractor methods to transfer from .odb to .hdf5
    Implements abilities to resize the dimenisons or timeframe of the data
    """

    __slots__ = (
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
        "_abaqus_executable",
        "_nodesets",
        "_frames",
        "_odb_to_npz_script_path",
        "_odb_to_npz_conversion_pickle_path",
        "_npz_result_path",
        "_bounded_nodes",
        "_target_nodes",
        "_filtered_nodes"
        )

    def __init__(self) -> None:
        """
        """

        super().__init__()

        self._odb_path: PathLike
        self._odb_source_dir: Union[PathLike, None]

        self._hdf_path: PathLike
        self._hdf_source_dir: Union[PathLike, None]

        self._abaqus_executable: str = "abaqus"

        self._nodesets: NullableStrListUnion = None
        self._frames: NullableIntListUnion = None

        self._bounded_nodes: DataFrameType # Spatially Bounded Nodes
        self._target_nodes: DataFrameType # Spatially Bounded Nodes with Time/Temp
        self._filtered_nodes: DataFrameType # Thermally/Temporally filtered

        # Hardcoded paths for Python 3 - 2 communication
        self._odb_to_npz_script_path: PathLike = pathlib.Path(
            pathlib.Path(__file__).parent,
            "py2_scripts",
            "odb_to_npz.py"
        )
        self._odb_to_npz_conversion_pickle_path: PathLike = pathlib.Path(
            pathlib.Path.cwd(),
            "odb_to_npz_conversion.pickle"
        )
        self._npz_result_path: PathLike = pathlib.Path(
            pathlib.Path.cwd(),
            "npz_path.pickle"
        )

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

        # TODO
        """self._parts: NullableStrListUnion
        self._nodes: dict[str, list[int]]
        
        self._frame_sample: int

        self._hdf_processed: bool = False
        self._loaded: bool = False

        self.bounded_nodes: Any

        self.out_nodes: Any
        self.out_nodes_low_time: Any"""


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
        if hasattr(self, "_x_high"): # If high is set
            if value > self.x_high:
                raise ValueError(f"The value for x_low ({value}) must not be greater than the value for")
        
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
                raise Exception("temp_low must be a float")
        
        if value < 0:
            raise Exception("temp_low must be greater than or equal to 0 (Kelvins)")
        
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
                raise Exception("temp_high must be a float")
        
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
                raise Exception("time_low must be a float")
        
        if value < 0:
            raise Exception("time_low must be greater than or equal to 0")
        
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
                raise Exception("time_high must be a float")
        
        self._time_high = value


    @property
    def odb_source_dir(self) -> Union[PathLike, None]:
        return self._odb_source_dir

    @odb_source_dir.setter
    def odb_source_dir(self, value: PathLike) -> None:
        if not isinstance(value, PathLike):
            value = pathlib.Path(value)

        if not value.exists():
            raise FileNotFoundError(f"Directory {value} does not exist")
        
        self._odb_source_dir = value

    @property
    def odb_path(self) -> PathLike:
        return self._odb_path

    
    @odb_path.setter
    def odb_path(self, value: PathLike) -> None:
        if not isinstance(value, PathLike):
            value = pathlib.Path(value)

        if value.exists():
            self._odb_path = value
            return

        if self.odb_source_dir is not None:
            if (self.odb_source_dir / value).exists():
                self._odb_path = self.odb_source_dir / value
                return

        if (pathlib.Path.cwd() / value).exists():
            self._odb_path = pathlib.Path.cwd() / value
            return

        raise FileNotFoundError(f"File {value} could not be found")

    @property
    def hdf_source_dir(self) -> Union[PathLike, None]:
        return self._hdf_source_dir

    @hdf_source_dir.setter
    def hdf_source_dir(self, value: PathLike) -> None:
        if not isinstance(value, PathLike):
            value = pathlib.Path(value)

        if not value.exists():
            raise FileNotFoundError(f"Directory {value} does not exist")
        
        self._hdf_source_dir = value

    @property
    def hdf_path(self) -> PathLike:
        return self._hdf_path

    
    @hdf_path.setter
    def hdf_path(self, value: PathLike) -> None:
        if not isinstance(value, PathLike):
            value = pathlib.Path(value)

        if value.exists():
            self._hdf_path = value
            return

        if self.hdf_source_dir is not None:
            if (self.hdf_source_dir / value).exists():
                self._hdf_path = self.hdf_source_dir / value
                return

        if (pathlib.Path.cwd() / value).exists():
            self._hdf_path = pathlib.Path.cwd() / value
            return

        raise FileNotFoundError(f"File {value} could not be found")


    @property
    def abaqus_executable(self) -> str:
        return self._abaqus_executable
    

    @abaqus_executable.setter
    def abaqus_executable(self, value: str) -> None:
        self._abaqus_executable = value

    
    @property
    def frames(self) -> list[int]:
        return self._frames

    
    @frames.setter
    def frames(self, value: list[int]) -> None:
        self._frames = value


    """def set_parts(self, parts: "list[str]") -> None:
        if not isinstance(parts, list):
            new_list: list[str] = [parts]
            self.parts = new_list

        else:
            self.parts = parts

    
    def set_nodes(self, nodes: "dict[str, list[int]]") -> None:
        if not isinstance(nodes, dict):
            nodes = dict(nodes)

        self.nodes = nodes

    
    def set_nodesets(self, nodesets: "list[str]") -> None:
        if not isinstance(nodesets, list):
            new_list: list[str] = [nodesets]
            self.nodesets = new_list

        else:
            self.nodesets = nodesets"""


    def convert_odb_to_hdf(self, hdf_path: Union[PathLike, None] = None) -> None:
        
        if not hasattr(self, "odb_path"):
            raise AttributeError("Path to target .odb file is not set")

        # If an hdf_path is passed in, update it on the user
        if hdf_path is not None:
            if not isinstance(hdf_path, pathlib.Path):
                hdf_path = pathlib.Path(hdf_path)
            
            if hdf_path.is_absolute():
                self._hdf_path = hdf_path
            
            else:
                try:
                    self._hdf_path = self._hdf_source_dir / hdf_path

                except AttributeError:
                    self._hdf_path = pathlib.Path.cwd() / hdf_path

        assert hasattr(self, "_hdf_path")

        # TODO Dataclass
        odb_to_npz_pickle_input_dict: dict[
            str,Union[list[str], list[int], None]
            ] = {
                "nodesets": self._nodesets,
                "frames": self._frames
            }
        pickle_file: TextIO
        with open(
            self._odb_to_npz_conversion_pickle_path, "wb") as pickle_file:
            pickle.dump(odb_to_npz_pickle_input_dict, pickle_file, protocol=2)

        odb_convert_args: list[Union[PathLike, str]]  = [
            self._abaqus_executable,
            "python",
            self._odb_to_npz_script_path,
            self._odb_path,
            self._odb_to_npz_conversion_pickle_path,
            self._npz_result_path
        ]

        subprocess.run(odb_convert_args, shell=True)

        result_file: TextIO
        result_dir: PathLike
        with open(self._npz_result_path, "rb") as result_file:
            result_dir = pathlib.Path(pickle.load(result_file))
        
        pathlib.Path.unlink(self._npz_result_path)

        print(self.hdf_path)
        convert_npz_to_hdf(self.hdf_path, result_dir)
        
        if result_dir.exists():
            shutil.rmtree(result_dir)


    def spatially_bind_nodes(self) -> None:
        coords_df: DataFrameType = get_node_coords(self.hdf_path)
        self._bounded_nodes = coords_df[
            (coords_df["X"] >= self.x_low)
            & (coords_df["X"] <= self.x_high)
            & (coords_df["Y"] >= self.y_low)
            & (coords_df["Y"] <= self.y_high)
            & (coords_df["Z"] >= self.z_low)
            & (coords_df["Z"] <= self.z_high)
        ]
        bounded_nodes_labels: DataFrameType = self._bounded_nodes[
            "Node Labels"
            ]

        node: int
        # TODO dataclass
        time_temp_extraction_args: list[
            tuple[
                PathLike, int, int, float, float, float
                ]
            ] = [
            (
                self.hdf_path,
                node - 1,
                1, # TODO Frame Ind
                self._bounded_nodes[bounded_nodes_labels == node]["X"],
                self._bounded_nodes[bounded_nodes_labels == node]["Y"],
                self._bounded_nodes[bounded_nodes_labels == node]["Z"],
                )
                for node in bounded_nodes_labels
            ]

        pool: MultiprocessingPoolType
        with multiprocessing.Pool() as pool:
            extracted_dfs: list[
                DataFrameType
                ] = pool.starmap(
                    get_node_times_temps,
                    time_temp_extraction_args
                    )

        self._target_nodes = pd.concat(extracted_dfs)
        self._filtered_nodes = self._target_nodes

        self._filtered_nodes = self.filter_nodes(
            "Time", 
            self._time_low,
            operator.ge
            )
        self._filtered_nodes = self.filter_nodes(
            "Time",
            self._time_high,
            operator.le
            )


    def filter_nodes(
        self,
        key: str,
        value: float, op: Union[Callable[[float, float], bool], str]
        ) -> DataFrameType:
        """
        """

        if isinstance(op, str):
            op_mapping: dict[str, Callable[[float, float], bool]] = {
                "==": operator.eq,
                "!=": operator.ne,
                ">": operator.gt,
                "<": operator.lt,
                ">=": operator.ge,
                "<=": operator.le
            }
            try:
                op = op_mapping[op]
            except KeyError:
                raise KeyError(f'Unsupported operator {op}. Must be one of' \
                    '"==", "!=", ">", "<", ">=", or "<="')

        return self._filtered_nodes[
            op(self._filtered_nodes[key], value)
            ]