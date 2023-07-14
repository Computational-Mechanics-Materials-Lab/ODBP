#!/usr/bin/env python

import pathlib
import multiprocessing

from typing import Optional, Union, Tuple, List

from .types import NullableIntList, NullableStrList, NullableNodeType,\
    NodeType
from .magic import ensure_magic, HDF_MAGIC_NUM, ODB_MAGIC_NUM

class OdbSettings():
    __slots__ = (
        "_x_low", # Done
        "_x_high", # Done
        "_y_low", # Done
        "_y_high", # Done
        "_z_low", # Done
        "_z_high", # Done
        "_temp_low", # Done
        "_temp_high", # Done
        "_time_low", # Done
        "_time_high", # Done
        "_time_step", # Done
        "_odb_path", # Done
        "_odb_source_dir", # ???
        "_hdf_path", # Done
        "_hdf_source_dir", # ???
        "_result_dir", # ???
        "_abaqus_executable",
        "_cpus",
        "_nodesets",
        "_frames",
        "_nodes",
        "_parts",
        "_steps",
        "_coord_key",
        "_temp_key",
        "_angle",
        "_colormap",
        "_save",
        "_save_format",
    )

    def __init__(self):

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
        self._time_step: int = 1

        self._time_low: float
        self._time_high: float

        self._cpus = multiprocessing.cpu_count()

        self._colormap: str = "turbo"
        self._save_format: str = ".png"
        self._save: bool = True

        # TODO
        self._angle = Union[str, Tuple[float, float, float]]


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
                raise ValueError(f"The value for x_low ({value})"
                " must not be greater than the value for x_high ({self.x_high})")

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
                raise ValueError(f"The value for x_high ({value})"
                " must not be less than the value for x_low ({self.x_low})")

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
                raise ValueError(f"The value for y_low ({value})"
                " must not be greater than the value for y_high"
                " ({self.y_high})")

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
                raise ValueError(f"The value for y_high ({value})"
                " must not be less than the value for y_low"
                " ({self.y_low})")

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
                raise ValueError(f"The value for z_low ({value})"
                " must not be greater than the value for z_high"
                " ({self.z_high})")

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
                raise ValueError(f"The value for z_high ({value})"
                " must not be less than the value for z_low"
                " ({self.z_low})")

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
            raise ValueError("temp_low must be greater than"
            " or equal to 0 (Kelvins)")

        if hasattr(self, "temp_high"):
            if value > self.temp_high:
                raise ValueError(f"The value for temp_low ({value})"
                " must not be greater than the value for temp_high"
                " ({self.temp_high})")

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
                raise ValueError(f"The value for temp_high ({value})"
                " must not be less than the value for temp_low"
                " ({self.temp_low})")

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
            raise ValueError("time_low must be greater than"
            " or equal to 0 (Kelvins)")

        if hasattr(self, "time_high"):
            if value > self.time_high:
                raise ValueError(f"The value for time_low ({value})"
                " must not be greater than the value for time_high"
                " ({self.time_high})")

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
                raise ValueError(f"The value for time_high ({value})"
                " must not be less than the value for time_low"
                " ({self.time_low})")

        self._time_high = value


    @property
    def time_step(self) -> int:
        return self._time_step

    
    @time_step.setter
    def time_step(self, value: int) -> None:
        if not isinstance(value, int):
            try:
                value = int(value)
                if value < 1:
                    raise ValueError

            except ValueError:
                raise ValueError("time_step must be an integer greater than or equal to 1")

        self._time_step = value


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


    @odb_path.setter
    def odb_path(self, value: pathlib.Path) -> None:
        value = pathlib.Path(value)

        target_path: pathlib.Path = value
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
        if not ensure_magic(target_path, ODB_MAGIC_NUM):
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

        if (
            not value.is_absolute()
            and hasattr(self, "hdf_source_dir")
            and self.hdf_source_dir is not None
        ):
            value = self.hdf_source_dir / value

        if value.exists():
            # Ensure magic numbers
            if not ensure_magic(value, HDF_MAGIC_NUM):
                raise ValueError(f"Given file {value} is not"
                " a .hdf5 hierarchical data format file.")

        self._hdf_path = value

    
    @property
    def result_dir(self) -> Optional[pathlib.Path]:
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
    def colormap(self) -> str:
        return self._colormap


    @colormap.setter
    def colormap(self, value: str) -> None:
        self._colormap = value


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