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

import numpy as np
import pandas as pd

<<<<<<< HEAD
from typing import TextIO, Union, Any, Tuple, List, Dict, Optional, Iterator
=======
from typing import TextIO, Callable, Union, Any, Tuple, List, Dict, Optional, Iterator
>>>>>>> e5096a3e1d21e9c5d8b0f783afcbd47f9876b482
from abc import abstractmethod

from .npz_to_hdf import convert_npz_to_hdf
from .read_hdf5 import get_odb_data
from .util import NullableIntList, NullableStrList, DataFrameType, NDArrayType, NullableNodeType, NodeType, ODB_MAGIC_NUM, HDF_MAGIC_NUM, H5PYFileType, H5PYGroupType

"""try:
    import pyvista as pv
except ImportError:
    PYVISTA_AVILABLE = False
else:
    PYVISTA_AVAILABLE = True"""

try:
    import pyvista as pv
except ImportError:
    PYVISTA_AVILABLE = False
else:
    PYVISTA_AVAILABLE = True


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
        "_extract_from_odb_script_path",
        "_extract_from_odb_pickle_path",
        "_extract_result_path",
        "_nodes",
        "_nodesets",
        "_frames",
<<<<<<< HEAD
<<<<<<< a6aac100c2a26d98ddcb557f5a6002969770018e
<<<<<<< 21a136b0d8bcaeb68e44c96b26dbe823e2798029
<<<<<<< a573ef9c876b6fa2f1de8a4a293a941abf5b9c0e
        "_cpus",
=======
        "_interactive",
        "_angle",
        "_colormap",
>>>>>>> Starting to implement 0.6.0
=======
        "_nodes",
=======
        "_parts",
        "_steps",
>>>>>>> 0.6.0, API is almost fully functional apart from iterator and plotting
=======
        "_nodes",
>>>>>>> e5096a3e1d21e9c5d8b0f783afcbd47f9876b482
        "_coord_key",
        "_temp_key",
        "_interactive",
        "_angle",
        "_colormap",
        "_iterator_ind",
        "_times"
<<<<<<< HEAD
>>>>>>> 0.6.0 Moving to Desktop
=======
>>>>>>> e5096a3e1d21e9c5d8b0f783afcbd47f9876b482
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

        self._hdf_path: pathlib.Path
        self._hdf_source_dir: Optional[pathlib.Path]

        self._abaqus_executable: str = "abaqus"

        self._nodes: NullableNodeType = None
        self._nodesets: NullableStrList = None
        self._frames: NullableIntList = None
<<<<<<< HEAD
        self._parts: NullableStrList = None
        self._steps: NullableStrList = None
=======
        self._nodes: Optional[Dict[str, List[int]]] = None
>>>>>>> e5096a3e1d21e9c5d8b0f783afcbd47f9876b482

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
        self._odb_to_npz_script_path: pathlib.Path = pathlib.Path(
            pathlib.Path(__file__).parent,
            "py2_scripts",
            "odb_to_npz.py"
        )

        self._extract_from_odb_script_path: pathlib.Path = pathlib.Path(
            pathlib.Path(__file__).parent,
            "py2_scripts",
            "extract.py"
        )

        self._odb_to_npz_conversion_pickle_path: pathlib.Path = pathlib.Path(
            pathlib.Path.cwd(),
            "odb_to_npz_conversion.pickle"
        )

        self._extract_from_odb_pickle_path: pathlib.Path = pathlib.Path(
            pathlib.Path.cwd(),
            "extract_from_odb.pickle"
        )

        self._npz_result_path: pathlib.Path = pathlib.Path(
            pathlib.Path.cwd(),
            "npz_path.pickle"
        )

<<<<<<< HEAD
<<<<<<< a6aac100c2a26d98ddcb557f5a6002969770018e
<<<<<<< a573ef9c876b6fa2f1de8a4a293a941abf5b9c0e
        self._cpus = multiprocessing.cpu_count()
=======
=======
        self._extract_result_path: pathlib.Path = pathlib.Path(
            pathlib.Path.cwd(),
            "extract_results.pickle"
        )

>>>>>>> 0.6.0, API is almost fully functional apart from iterator and plotting
        self._interactive: bool = False
        self._colormap: str = "turbo"

        # TODO
        self._angle = Union[str, Tuple[float, float, float]]
>>>>>>> Starting to implement 0.6.0
=======
        self._interactive: bool = False
        self._colormap: str = "turbo"

        # TODO
        self._angle = Union[str, Tuple[float, float, float]]

        # TODO
        """self._parts: NullableStrList
        self._steps: NullableStrList"""
>>>>>>> e5096a3e1d21e9c5d8b0f783afcbd47f9876b482

        self._iterator_ind: int = 0
        self._times: NDArrayType


    # def __new__(self) -> None:
    #     raise Exception("Do not instantiate OdbViewer instances this way."
    #             "Please use One of the following odbp.OdbViewer methods"
    #             "instead:\n"
    #             "")


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


    @odb.setter
    def odb(self, value: Any) -> None:
        _ = value
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
    def abaqus_executable(self) -> str:
        return self._abaqus_executable


    @abaqus_executable.setter
    def abaqus_executable(self, value: str) -> None:
        self._abaqus_executable = value


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
<<<<<<< a6aac100c2a26d98ddcb557f5a6002969770018e
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
=======
    def parts(self) -> NullableStrList:
        return self._parts
>>>>>>> 0.6.0, API is almost fully functional apart from iterator and plotting


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
<<<<<<< HEAD
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


    """
    @property
=======
>>>>>>> e5096a3e1d21e9c5d8b0f783afcbd47f9876b482
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
<<<<<<< HEAD
"""
=======

>>>>>>> e5096a3e1d21e9c5d8b0f783afcbd47f9876b482

    def convert_odb_to_hdf(
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
            str, Optional[Union[List[str], List[int]]]
            ] = {
<<<<<<< a6aac100c2a26d98ddcb557f5a6002969770018e
                "nodesets": self._nodesets,
                "frames": self._frames,
<<<<<<< HEAD
<<<<<<< 21a136b0d8bcaeb68e44c96b26dbe823e2798029
                "cpus": self.cpus
=======
                "coord_key": self._coord_key,
                "temp_key": self._temp_key
>>>>>>> 0.6.0 Moving to Desktop
=======
                "nodes": self.nodes,
                "nodesets": self.nodesets,
                "frames": self.frames,
                "parts": self.parts,
                "steps": self.steps,
                "coord_key": self.coord_key,
                "temp_key": self.temp_key
>>>>>>> 0.6.0, API is almost fully functional apart from iterator and plotting
=======
                "coord_key": self._coord_key,
                "temp_key": self._temp_key
>>>>>>> e5096a3e1d21e9c5d8b0f783afcbd47f9876b482
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

        # shell=True is BAD PRACTICE, but abaqus python won't run without it
        subprocess.run(odb_convert_args, shell=True)

        result_file: TextIO
        result_dir: pathlib.Path
        with open(self._npz_result_path, "rb") as result_file:
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
        if hasattr(self, "hdf_path"):
            return self._extract_from_hdf()
        elif hasattr(self, "odb_path"):
            return self.extract_from_odb()


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
                "temp_key": self.temp_key
            }
            
        temp_file: TextIO
        with open(self._extract_from_odb_pickle_path, "wb") as temp_file:
            pickle.dump(
                extract_odb_pickle_input_dict,
                temp_file,
                protocol=2
                )
        args_list: List[Union[str, pathlib.Path]] = [
            self.abaqus_executable,
            "python",
            self._extract_from_odb_script_path,
            target_file,
            self._extract_from_odb_pickle_path,
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
            results_df_list.append(pd.DataFrame({time: result}, orient="index"))

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

        frame: DataFrameType
        for frame in self:
            min: frame["Temp"].min()
            max: frame["Temp"].max()




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


<<<<<<< HEAD
    """def plot_3d_all_times(
=======
    def plot_3d_all_times(
>>>>>>> e5096a3e1d21e9c5d8b0f783afcbd47f9876b482
            self,
            label: str = "",
            ) -> "List[pv.Plotter]":
        """

<<<<<<< HEAD
    """
=======
        """
>>>>>>> e5096a3e1d21e9c5d8b0f783afcbd47f9876b482
        if not PYVISTA_AVAIL:
            raise Exception("Plotting capabilities are not included."
                            'Please pip install odb-plotter["plot"]'
                            'or pip install odb-plotter["all"] to use'
                            "three-dimensional plotting")

        times: DataFrameType = np.sort(self["Time"].unique())

        plotting_args: List[
            Tuple[
                float,
                str
                ]
            ] = [(time, label) for time in times]
        results: List[pv.Plotter] = list()
        time: float
        for time in times:
            result = self._plot_3d_single(time, label)
            if result is not None:
<<<<<<< HEAD
                results.append(result)"""
        # TODO Any way to make this work?
    """
=======
                results.append(result)
        # TODO Any way to make this work?
        """
>>>>>>> e5096a3e1d21e9c5d8b0f783afcbd47f9876b482
        # TODO dataclass
        plotting_args: List[
            Tuple[
                float,
                str
                ]
            ] = [(time, label) for time in times]
        results: List[pv.Plotter] = list()
        pool: MultiprocessingPoolType
        with multiprocessing.Pool() as pool:
            results: list[pv.Plotter] = pool.starmap(
                self._plot_3d_single,
                plotting_args
                )"""

<<<<<<< HEAD
        #return results 


    """def _plot_3d_single(
=======
        return results 


    def _plot_3d_single(
>>>>>>> e5096a3e1d21e9c5d8b0f783afcbd47f9876b482
        self,
        time: float,
        label: str
        )-> "Optional[pv.Plotter]":
        """
<<<<<<< HEAD
    """
=======
        """
>>>>>>> e5096a3e1d21e9c5d8b0f783afcbd47f9876b482

        if not PYVISTA_AVAIL:
            raise Exception("Plotting capabilities are not included."
                            'Please pip install odb-plotter["plot"]'
                            'or pip install odb-plotter["all"] to use'
                            "three-dimensional plotting")

        dims_columns: set[str] = {"X", "Y", "Z"}
        combined_label: str = f"{label}-{round(time, 2):.2f}"

        plotter: pv.Plotter = pv.Plotter(
            off_screen=(not self._interactive),
            window_size=(1920, 1080),
            lighting="three lights"
            )

        plotter.add_text(
            combined_label,
            position="upper_edge",
            color="#000000",
            font="courier"
        )

        instance_nodes: DataFrameType = self.filter_nodes(
            "Time",
            time,
            operator.eq
        )

        if not instance_nodes.empty:
            points: pv.PolyData = pv.PolyData(
                instance_nodes.drop(
                    columns=list(
                        set(self._target_nodes.columns.values.tolist())
                        - dims_columns
                        )
                    ).to_numpy()
                )
            
            points["Temp"] = instance_nodes["Temp"].to_numpy()
            mesh: pv.PolyData = points.delaunay_3d()

            plotter.add_mesh(
                mesh,
                scalars="Temp",
                cmap = pv.LookupTable(
                    cmap=self._colormap,
                    scalar_range=(
                        self._temp_low,
                        self._temp_high
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
            plotter.camera.elevation = 0
            plotter.camera.azimuth = 270
            plotter.camera.roll = 300
            plotter.camera_set = True

<<<<<<< HEAD
            return plotter"""
=======
            return plotter
>>>>>>> e5096a3e1d21e9c5d8b0f783afcbd47f9876b482


class OdbLoader:

<<<<<<< a6aac100c2a26d98ddcb557f5a6002969770018e
    def load_hdf(self, hdf_path: pathlib.Path, cpus: int) -> DataFrameType:
=======
    def load_hdf(self, hdf_path: pathlib.Path) -> DataFrameType:
>>>>>>> 0.6.0, API is almost fully functional apart from iterator and plotting
        return get_odb_data(hdf_path, cpus)


class OdbUnloader:

    @abstractmethod
    def unload_hdf(self) -> None:
        ...
