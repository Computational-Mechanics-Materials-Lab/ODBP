#!/usr/bin/env python

import pathlib
import multiprocessing

import numpy as np

from typing import Optional, BinaryIO, Dict, List, Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from .types import NullableStrList, NullableNodeType, NodeType
from .magic import ensure_magic, HDF_MAGIC_NUM, ODB_MAGIC_NUM


class OdbSettings:
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
        "_time_step",
        "_odb_path",
        "_odb_source_dir",  # TODO
        "_hdf_path",
        "_hdf_source_dir",  # TODO
        "_result_dir",  # TODO
        "_abaqus_executable",
        "_cpus",
        #"_data_model",
        "_nodesets",
        "_nodes",
        "_parts",
        "_steps",
        "_coord_key",
        "_target_outputs",
        "_views",
        "_view",
        "_interactive",
        "_colormap",
        "_save",
        "_save_format",
        "_font",
        "_font_color",
        "_font_size",
        "_background_color",
        "_below_range_color",
        "_above_range_color",
        "_axis_text_color",
        "_filename",
        "_title",
        "_show_axes",
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
        self._parts: NullableStrList = None
        self._steps: NullableStrList = None

        self._coord_key: str = "COORD"
        self._target_outputs: NullableStrList = None

        self._x_low: float = -1 * np.inf
        self._x_high: float = np.inf
        self._y_low: float = -1 * np.inf
        self._y_high: float = np.inf
        self._z_low: float = -1 * np.inf
        self._z_high: float = np.inf

        self._temp_low: float = 0
        self._temp_high: float = np.inf
        self._time_step: int = 1

        self._time_low: float = 0
        self._time_high: float = np.inf

        self._cpus: int = multiprocessing.cpu_count()

        # We'll have to use a simple enum, because of python 2-3 conversion
        #self._data_model: int = 0

        self._colormap: str = "turbo"
        self._save_format: str = ".png"
        self._save: bool = True
        self._font: str = "courier"
        self._font_size: int = 16
        # Any because we can't rely on pyvista.colorlike
        self._font_color: Any = "#000000"
        self._background_color: Any = "#FFFFFF"
        self._below_range_color: Any = "#000000"
        self._above_range_color: Any = "#F0F0F0"
        self._axis_text_color: Any = "#000000"

        self._views: dict = {}
        tf: BinaryIO
        with open((pathlib.Path(__file__).parent / "data") / "views.toml", "rb") as tf:
            temp_views: dict[str, list[str]] = tomllib.load(tf)

        reversed_views = {}
        for v in temp_views.values():
            new_v = v[:]
            new_v = [tuple(n) if isinstance(n, list) else n for n in new_v]
            new_v = tuple(new_v)
            reversed_views[new_v] = []
        #reversed_views: dict = {tuple(v): [] for v in temp_views.values()}

        for k, v in temp_views.items():
            new_v = v[:]
            new_v = [tuple(n) if isinstance(n, list) else n for n in new_v]
            new_v = tuple(new_v)
            reversed_views[new_v].append(k)

        self._views = {tuple(v): k for k, v in reversed_views.items()}

        self._view: str = "UFR-U"

        self._interactive: bool = True

        self._filename: str = ""
        self._title: str = ""

        self._show_axes: bool = True

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
        if hasattr(self, "x_high"):  # If high is set
            if value > self.x_high:
                raise ValueError(
                    f"The value for x_low ({value})"
                    " must not be greater than the value for x_high ({self.x_high})"
                )

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
        if hasattr(self, "x_low"):  # If low is set
            if value < self.x_low:
                raise ValueError(
                    f"The value for x_high ({value})"
                    " must not be less than the value for x_low ({self.x_low})"
                )

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
        if hasattr(self, "y_high"):  # If high is set
            if value > self.y_high:
                raise ValueError(
                    f"The value for y_low ({value})"
                    " must not be greater than the value for y_high"
                    " ({self.y_high})"
                )

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
        if hasattr(self, "y_low"):  # If low is set
            if value < self.y_low:
                raise ValueError(
                    f"The value for y_high ({value})"
                    " must not be less than the value for y_low"
                    " ({self.y_low})"
                )

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
        if hasattr(self, "z_high"):  # If high is set
            if value > self.z_high:
                raise ValueError(
                    f"The value for z_low ({value})"
                    " must not be greater than the value for z_high"
                    " ({self.z_high})"
                )

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
        if hasattr(self, "z_low"):  # If low is set
            if value < self.z_low:
                raise ValueError(
                    f"The value for z_high ({value})"
                    " must not be less than the value for z_low"
                    " ({self.z_low})"
                )

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
            raise ValueError("temp_low must be greater than" " or equal to 0 (Kelvins)")

        if hasattr(self, "temp_high"):
            if value > self.temp_high:
                raise ValueError(
                    f"The value for temp_low ({value})"
                    " must not be greater than the value for temp_high"
                    " ({self.temp_high})"
                )

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
                raise ValueError(
                    f"The value for temp_high ({value})"
                    " must not be less than the value for temp_low"
                    " ({self.temp_low})"
                )

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
            raise ValueError("time_low must be greater than" " or equal to 0 (Kelvins)")

        if hasattr(self, "time_high"):
            if value > self.time_high:
                raise ValueError(
                    f"The value for time_low ({value})"
                    " must not be greater than the value for time_high"
                    " ({self.time_high})"
                )

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
                raise ValueError(
                    f"The value for time_high ({value})"
                    " must not be less than the value for time_low"
                    " ({self.time_low})"
                )

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
                raise ValueError(
                    "time_step must be an integer greater than or equal to 1"
                )

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
            raise ValueError(
                f"Given file {value} is not a .odb object" "database file."
            )

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
                raise ValueError(
                    f"Given file {value} is not"
                    " a .hdf5 hierarchical data format file."
                )

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
        if value <= 0:
            raise ValueError("cpus must be an integer greater than 0")
        self._cpus = value

    #@property
    #def data_model(self) -> str:
    #    if self._data_model == 0:
    #        return "thermal"
    #    elif self._data_model == 1:
    #        return "mechanical"

    #    raise ValueError

    #@data_model.setter
    #def data_model(self, value: str) -> None:
    #    if value.lower() == "thermal":
    #        self._data_model = 0
    #    elif value.lower() == "mechanical":
    #        self._data_model = 1
    #    else:
    #        raise ValueError

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
    def coord_key(self) -> str:
        return self._coord_key

    @coord_key.setter
    def coord_key(self, value: str) -> None:
        self._coord_key = value

    @property
    def target_outputs(self) -> NullableStrList:
        return self._target_outputs

    @target_outputs.setter
    def target_outputs(self, value: "List[str]") -> None:
        self._target_outputs = value

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

    @property
    def font(self) -> str:
        return self._font

    @font.setter
    def font(self, value: str) -> None:
        self._font = value

    @property
    def font_color(self) -> Any:
        return self._font_color

    @font_color.setter
    def font_color(self, value: Any) -> None:
        self._font_color = value

    @property
    def font_size(self) -> int:
        return self._font_size

    @font_size.setter
    def font_size(self, value: int) -> None:
        self._font_size = value

    @property
    def background_color(self) -> Any:
        return self._background_color

    @background_color.setter
    def background_color(self, value: Any) -> None:
        self._background_color = value

    @property
    def below_range_color(self) -> Any:
        return self._below_range_color

    @below_range_color.setter
    def below_range_color(self, value: Any) -> None:
        self._below_range_color = value

    @property
    def above_range_color(self) -> Any:
        return self._above_range_color

    @above_range_color.setter
    def above_range_color(self, value: Any) -> None:
        self._above_range_color = value

    @property
    def axis_text_color(self) -> Any:
        return self._axis_text_color

    @axis_text_color.setter
    def axis_text_color(self, value: Any) -> None:
        if type(value) is str:
            if value.lower() not in ("white", "black", "#000000", "#ffffff"):
                raise ValueError(f"Axis color must be a form of the colors white or black, not {value}")

        self._axis_text_color = value

    @property
    def view(self) -> str:
        return self._view

    @view.setter
    def view(self, value: str) -> None:
        value = value.upper()
        self._view = value

    @property
    def filename(self) -> str:
        return self._filename

    @filename.setter
    def filename(self, value: str) -> None:
        self._filename = value

    @property
    def title(self) -> str:
        return self._title

    @title.setter
    def title(self, value: str) -> None:
        self._title = value


    @property
    def show_axes(self) -> bool:
        return self._show_axes


    @show_axes.setter
    def show_axes(self, value: bool) -> None:
        self._show_axes = value

    @property
    def interactive(self) -> bool:
        return self._interactive

    @interactive.setter
    def interactive(self, value: bool) -> None:
        self._interactive = value

    def get_odb_settings_state(self) -> str:
        result: str = "Current state of the ODB Object:"

        # Files
        result += "\n\nFiles:"
        hdf_file: str = str(self.hdf_path) if hasattr(self, "hdf_path") else "Not Set"
        odb_file: str = str(self.odb_path) if hasattr(self, "odb_path") else "Not Set"
        result += f"\n\n\t.hdf5 file: {hdf_file}"
        result += f"\n\t.odb file: {odb_file}"
        hdf_source_dir: str = (
            str(self.hdf_source_dir) if hasattr(self, "hdf_source_dir") else "Not Set"
        )
        odb_source_dir: str = (
            str(self.odb_source_dir) if hasattr(self, "odb_source_dir") else "Not Set"
        )
        result_dir: str = (
            str(self.result_dir) if hasattr(self, "result_dir") else "Not Set"
        )
        result += f"\n\n\tSource Directory for .hdf5 files: {hdf_source_dir}"
        result += f"\n\tSource Directory for .odb files: {odb_source_dir}"
        result += f"\n\tDirectory for resulting images: {result_dir}"

        # Ranges
        result += "\n\nRanges:\n\n\tSpatial Ranges:"

        x_low: str = str(self.x_low) if hasattr(self, "x_low") else "Not Set"
        x_high: str = str(self.x_high) if hasattr(self, "x_high") else "Not Set"
        result += f"\n\t\tX Range: {x_low} to {x_high}"

        y_low: str = str(self.y_low) if hasattr(self, "y_low") else "Not Set"
        y_high: str = str(self.y_high) if hasattr(self, "y_high") else "Not Set"
        result += f"\n\t\tY Range: {y_low} to {y_high}"

        z_low: str = str(self.z_low) if hasattr(self, "z_low") else "Not Set"
        z_high: str = str(self.z_high) if hasattr(self, "z_high") else "Not Set"
        result += f"\n\t\tZ Range: {z_low} to {z_high}"

        result += "\n\n\tTemporal Range:"
        time_low: str = str(self.time_low) if hasattr(self, "time_low") else "Not Set"
        time_high: str = (
            str(self.time_high) if hasattr(self, "time_high") else "Not Set"
        )
        result += f"\n\t\tTime Range: {time_low} to {time_high}"

        result += "\n\n\tThermal Range:"
        temp_low: str = str(self.temp_low) if hasattr(self, "temp_low") else "Not Set"
        temp_high: str = (
            str(self.temp_high) if hasattr(self, "temp_high") else "Not Set"
        )
        result += f"\n\t\tTemperature Range: {temp_low} to {temp_high}"

        result += "\n\nProcessing:"
        abaqus_executable: str = (
            self.abaqus_executable if hasattr(self, "abaqus_executable") else "Not Set"
        )
        result += f"\n\n\tAbaqus Executable: {abaqus_executable}"
        cpus: str = str(self.cpus) if hasattr(self, "cpus") else "Not Set"
        result += f"\n\tNumber of CPU Cores to Use: {cpus}"

        result += "\n\nSelected Values:"
        nodesets: str = (
            str(self.nodesets)
            if hasattr(self, "nodesets") and self.nodesets is not None
            else "Not Set"
        )
        parts: str = (
            str(self.parts)
            if hasattr(self, "parts") and self.parts is not None
            else "Not Set"
        )
        steps: str = (
            str(self.steps)
            if hasattr(self, "steps") and self.steps is not None
            else "Not Set"
        )
        nodes: str = (
            str(self.parse_chain(self.nodes))
            if hasattr(self, "nodes") and self.nodes is not None
            else "All Nodes"
        )
        coord_key: str = (
            str(self.coord_key) if hasattr(self, "coord_key") else "Not Set"
        )
        target_outputs: str = (
            str(self.target_outputs)
            if hasattr(self, "target_outputs") and self.target_outputs is not None
            else "All Outputs"
        )
        result += f"\n\tSelected Nodesets: {nodesets}"
        result += f"\n\tSelected Parts: {parts}"
        result += f"\n\tSelected Steps: {steps}"
        result += f"\n\tSelected Nodes: {nodes}"
        time_step: str = (
            str(self.time_step) if hasattr(self, "time_step") else "Not Set"
        )
        result += f"\n\tSample every nth frame where n is: {time_step}"
        result += f"\n\n\tCoordinate Key: {coord_key}"
        result += f"\n\tTemperature Key: {target_outputs}"

        view: str = str(self.view) if hasattr(self, "view") else "Not Set"
        colormap: str = str(self.colormap) if hasattr(self, "colormap") else "Not Set"
        save_format: str = (
            str(self.save_format) if hasattr(self, "save_format") else "Not Set"
        )
        font: str = str(self.font) if hasattr(self, "font") else "Not Set"
        font_color: str = (
            str(self.font_color) if hasattr(self, "font_color") else "Not Set"
        )
        font_size: str = (
            str(self.font_size) if hasattr(self, "font_size") else "Not Set"
        )
        background_color: str = (
            str(self.background_color)
            if hasattr(self, "background_color")
            else "Not Set"
        )
        below_range_color: str = (
            str(self.below_range_color)
            if hasattr(self, "below_range_color")
            else "Not Set"
        )
        above_range_color: str = (
            str(self.above_range_color)
            if hasattr(self, "above_range_color")
            else "Not Set"
        )

        result += "\n\nPlotting Options:"
        result += f"\n\tViewing Angle: {view}"
        result += f"\n\tInteractive Viewing?: {'Yes' if self.interactive else 'No'}"
        result += f"\n\tColormap: {colormap}"
        result += f"\n\tWill images be saved: {'Yes' if self.save else 'No'}"
        result += f"\n\tImage format as which images will be saved: {save_format}"
        result += f"\n\tFont family/size/color: {font}/{font_size}/{font_color}"
        result += f"\n\tImage Background Color: {background_color}"
        result += f"\n\tColor for values below given range: {below_range_color}"
        result += f"\n\tColor for values above given range: {above_range_color}"

        result += f"\n\nFilename under which images are saved: {self.filename}"
        result += f"\nTitle placed on images: {self.title}"
        result += "\n\nODB Data Loaded: "

        return result
