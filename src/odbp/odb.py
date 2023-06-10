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

import pandas as pd

from typing import TextIO, Callable, Union, Any, List, Dict
from abc import abstractmethod

from .npz_to_hdf import convert_npz_to_hdf
from .read_hdf5 import get_odb_data
from .util import NullableIntList, NullableStrList, DataFrameType, MultiprocessingPoolType


class Odb():
    """
    Stores Data from a .hdf5, implements extractor methods to transfer from .odb to .hdf5
    Implements abilities to resize the dimenisons or timeframe of the data
    """

    # TODO user settings sub-section, overload getattr
    __slots__ = (
        "_odb_handler",
        "_odb",
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
        "_odb_to_npz_script_path",
        "_odb_to_npz_conversion_pickle_path",
        "_npz_result_path",
        "_nodesets",
        "_frames",
        "_cpus",
        )

    def __init__(self) -> None:
        """
        Type Hints and hard-coded parameters. See the @staticmethod
        "constructors" of this class in order to learn about initialization
        """

        self._odb_handler: Union[OdbLoader, OdbUnloader] = OdbLoader()
        self._odb: DataFrameType 

        self._odb_path: pathlib.Path
        self._odb_source_dir: Union[pathlib.Path, None]

        self._hdf_path: pathlib.Path
        self._hdf_source_dir: Union[pathlib.Path, None]

        self._abaqus_executable: str = "abaqus"

        self._nodesets: NullableStrList = None
        self._frames: NullableIntList = None

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
        self._odb_to_npz_script_path: pathlib.Path = pathlib.Path(
            pathlib.Path(__file__).parent,
            "py2_scripts",
            "odb_to_npz.py"
        )

        self._odb_to_npz_conversion_pickle_path: pathlib.Path = pathlib.Path(
            pathlib.Path.cwd(),
            "odb_to_npz_conversion.pickle"
        )

        self._npz_result_path: pathlib.Path = pathlib.Path(
            pathlib.Path.cwd(),
            "npz_path.pickle"
        )

        self._cpus = multiprocessing.cpu_count()

        # TODO
        """self._parts: NullableStrList
        self._nodes: dict[str, list[int]]"""


    # def __new__(self) -> None:
    #     raise Exception("Do not instantiate OdbViewer instances this way."
    #             "Please use One of the following odbp.OdbViewer methods"
    #             "instead:\n"
    #             "")


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


    @odb.setter
    def odb(self, value: Any) -> None:
        raise Exception('"odb" property should not be set this way')


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
    def odb_source_dir(self) -> "Union[pathlib.Path, None]":
        return self._odb_source_dir


    @odb_source_dir.setter
    def odb_source_dir(self, value: pathlib.Path) -> None:
        value = pathlib.Path(value)

        if not value.exists():
            raise FileNotFoundError(f"Directory {value} does not exist")

        self._odb_source_dir = value


    @property
    def odb_path(self) -> "Union[pathlib.Path, None]":
        return self._odb_path


    @odb_path.setter
    def odb_path(self, value: pathlib.Path) -> None:
        value = pathlib.Path(value)

        if value.exists():
            self._odb_path = value
            return

        if self.odb_source_dir is not None:
            source_dir_path: pathlib.Path = self.odb_source_dir / value
            if source_dir_path.exists():
                self._odb_path = source_dir_path
                return

        cwd_path: pathlib.Path = pathlib.Path.cwd() / value
        if cwd_path.exists():
            self._odb_path = cwd_path
            return

        raise FileNotFoundError(f"File {value} could not be found")


    @property
    def hdf_source_dir(self) -> "Union[pathlib.Path, None]":
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

        if value.is_absolute():
            self._hdf_path = value
            return

        if hasattr(self, "_hdf_source_dir"):
            self._hdf_path = self.hdf_source_dir / value
            return

        self._hdf_path = value


    @property
    def abaqus_executable(self) -> str:
        return self._abaqus_executable


    @abaqus_executable.setter
    def abaqus_executable(self, value: str) -> None:
        self._abaqus_executable = value


    @property
    def frames(self) -> "List[int]":
        return self._frames


    @frames.setter
    def frames(self, value: "List[int]") -> None:
        self._frames = value


    @property
    def nodesets(self) -> "List[int]":
        return self._nodesets


    @nodesets.setter
    def nodesets(self, value: "List[int]") -> None:
        self._nodesets = value


    @property
    def cpus(self) -> int:
        return self._cpus


    @cpus.setter
    def cpus(self, value: int) -> None:
        assert value > 0
        self._cpus = value


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


    def convert_odb_to_hdf(
            self,
            hdf_path: "Union[pathlib.Path, None]" = None,
            *,
            odb_path: "Union[pathlib.Path, None]" = None,
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

        self._convert_odb_to_hdf(hdf_path, odb_path)


    @classmethod
    def convert(
            cls,
            hdf_path: pathlib.Path,
            odb_path: pathlib.Path
            ) -> None:
        hdf_path = pathlib.Path(hdf_path)
        odb_path = pathlib.Path(odb_path)
        cls()._convert_odb_to_hdf(hdf_path, odb_path)


    def _convert_odb_to_hdf(
            self,
            hdf_path: pathlib.Path,
            odb_path: pathlib.Path
            ) -> None:

        odb_to_npz_pickle_input_dict: Dict[
            str, Union[List[str], List[int], None]
            ] = {
                "nodesets": self._nodesets,
                "frames": self._frames,
                "cpus": self.cpus
            }
        pickle_file: TextIO
        with open(
            self._odb_to_npz_conversion_pickle_path, "wb") as pickle_file:
            pickle.dump(odb_to_npz_pickle_input_dict, pickle_file, protocol=2)

        odb_convert_args: List[Union[pathlib.Path, str]]  = [
            self.abaqus_executable,
            "python",
            self._odb_to_npz_script_path,
            odb_path,
            self._odb_to_npz_conversion_pickle_path,
            self._npz_result_path
        ]

        subprocess.run(odb_convert_args, shell=True)

        result_file: TextIO
        result_dir: pathlib.Path
        with open(self._npz_result_path, "rb") as result_file:
            result_dir = pathlib.Path(pickle.load(result_file))

        pathlib.Path.unlink(self._npz_result_path)

        convert_npz_to_hdf(hdf_path, result_dir)

        if result_dir.exists():
            shutil.rmtree(result_dir)


    def load_hdf(self) -> None:

        if not hasattr(self, "hdf_path"):
            raise AttributeError("hdf_path attribute must be set before "
                    "calling load_hdf method.")

        try:
            # Only case where this should be set, bypass the setter
            self._odb = self._odb_handler.load_hdf(self.hdf_path, self.cpus)
            self._odb_handler = OdbUnloader()

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


    # TODO generator method for time steps. Overload __iter__?


class OdbLoader:

    def load_hdf(self, hdf_path: pathlib.Path, cpus: int) -> DataFrameType:
        return get_odb_data(hdf_path, cpus)


class OdbUnloader:

    @abstractmethod
    def unload_hdf(self) -> None:
        ...
