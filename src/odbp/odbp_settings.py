#!/usr/bin/env python

import sys
import pathlib
import multiprocessing

from typing import BinaryIO, Any, Collection, Self, Iterator
#from collections import UserDict
from enum import Enum, auto

import numpy as np
import numpy.typing as npt

if sys.version_info.major >= 3 and sys.version_info.minor >= 11:
    import tomllib
else:
    import tomli as tomllib

PLOTTING_AVAILABLE: bool
try:
    import polyscope as ps
except (ImportError, ModuleNotFoundError):
    PLOTTING_AVAILABLE = False
else:
    PLOTTING_AVAILABLE = True

from .magic import ensure_magic, H5_MAGIC_NUM, ODB_MAGIC_NUM
from .reader import OdbpData


class OdbpOutputBoundsRange(Enum):
    BETWEEN = auto()
    OUTSIDE = auto()


class OdbpOutput:
    __slots__ = (
        "_ODBPOUTPUTBOUNDSRANGE",
        "_old_name",
        "_name",
        "_abs_min",
        "bound_min",
        "cmap_min",
        "_abs_max",
        "bound_max",
        "cmap_max",
        "_bounds_between_or_outside",
        "bound_min_equal",
        "bound_max_equal",
        "cmap_min_equal",
        "cmap_max_equal",
        "_default",
    )
    def __init__(self: Self, name: str, data: OdbpData, output_mapping: dict[str, float], defaults_for_outputs: dict[str, float]) -> None:

        self._ODBPOUTPUTBOUNDSRANGE: OdbpOutputBoundsRange = OdbpOutputBoundsRange 

        # Getters, both-getter
        self._old_name: str = name
        self._name: str
        self._name = output_mapping.get(self._old_name, self._old_name)

        name_data: npt.NDArray
        if self._name in data.node_data:
            name_data = data.node_data[self._name].to_numpy()
        
        elif self._name in data.element_data:
            name_data = data.element_data[self._name].to_numpy()
        
        name_min: float = np.min(name_data)
        name_max: float = np.max(name_data)

        self._abs_min: float = name_min
        self.bound_min: float = name_min
        self.cmap_min: float = name_min

        self._abs_max: float = name_max
        self.bound_max: float = name_max
        self.cmap_max: float = name_max

        # TODO
        ## appender/remover/resetter
        #self.bounds_list: list[float] = []

        self._bounds_between_or_outside: str = self._ODBPOUTPUTBOUNDSRANGE.BETWEEN

        self.bound_min_equal: bool = False
        self.bound_max_equal: bool = False

        self.cmap_min_equal: bool = False
        self.cmap_max_equal: str = False

        self._default = defaults_for_outputs.get(self._old_name)

    @property
    def old_name(self: Self) -> str:
        return self._old_name

    @property
    def name(self: Self) -> str:
        return self._name

    @property
    def names(self: Self) -> tuple[str, ...]:
        if self.name == self.old_name:
            return self.name
        return self.old_name, self.name

    @property
    def abs_min(self: Self) -> float:
        return self._abs_min

    @property
    def abs_max(self: Self) -> float:
        return self._abs_max

    @property
    def abs_extrema(self: Self) -> tuple[float, float]:
        return self.abs_min, self.abs_max

    def set_bounds(self: Self, lower: float, upper: float | None = None) -> None:
        if upper is not None:
            self.bound_min = lower
            self.bound_max = upper

        else:
            self.bound_min = lower
            self.bound_max = lower
            # In this case, we need to guarantee lt/gt
            self.bound_max_equal = False
            self.bound_min_equal = False

    def set_cmap_bounds(self: Self, lower: float, upper: float | None = None) -> None:
        if upper is not None:
            self.cmap_min = lower
            self.cmap_max = upper

        else:
            self.cmap_min = lower
            self.cmap_max = lower

    @property
    def bounds(self: Self) -> None:
        return self.bound_min, self.bound_max

    @property
    def cmap_bounds(self: Self) -> None:
        return self.cmap_min, self.cmap_max

    def reset_bound_min(self: Self) -> None:
        self.bound_min = self.abs_min

    def reset_bound_max(self: Self) -> None:
        self.bound_max = self.abs_max

    def reset_bounds(self: Self) -> None:
        self.reset_bound_min()
        self.reset_bound_max()

    def reset_cmap_bound_min(self: Self) -> None:
        self.cmap_min = self.abs_min

    def reset_cmap_bound_max(self: Self) -> None:
        self.cmap_max = self.abs_max

    def reset_cmap_bounds(self: Self) -> None:
        self.reset_cmap_bound_min()
        self.reset_cmap_bound_max()

    @property
    def bounds_between_or_outside(self: Self) -> OdbpOutputBoundsRange:
        return self._bounds_between_or_outside

    @bounds_between_or_outside.setter
    def bounds_between_or_outside(self: Self, val: OdbpOutputBoundsRange) -> None:
        if not isinstance(val, OdbpOutputBoundsRange):
            raise ValueError(f"The given value must be one of the OdbpOutputBoundsRange enumerations, not {val}")
        
        self._bounds_between_or_outside = val

    @property
    def default(self: Self) -> float | None:
        return self._default

    @property
    def ODBPOUTPUTBOUNDSRANGE(self: Self) -> OdbpOutputBoundsRange:
        return self._ODBPOUTPUTBOUNDSRANGE


class OdbpOutputs:
    __slots__ = (
        "outputs",
        "outputs_by_names",
    )
    def __init__(self: Self, data: OdbpData, output_mapping: dict[str, float], defaults_for_outputs: dict[str, float]) -> None:
        col: str
        all_old_names: list[str] = list(set(list(data.node_data.columns.values) + list(data.element_data.columns.values)))
        self.outputs: list[OdbpOutput] = []
        self.outputs_by_names: dict[str, OdbpOutput] = {}
        for col in all_old_names:
            new_output: OdbpOutput = OdbpOutput(col, data, output_mapping, defaults_for_outputs)
            self.outputs.append(new_output)
            self.outputs_by_names[new_output.name] = new_output
            #new_output_names: tuple[str, ...] = new_output.names
            #name: str
            #for name in new_output_names:
            #    print(name)
            #    self.outputs_by_names[name] = new_output

    #def __contains__(self: Self, key: Any) -> bool:
    #    return key in self.outputs

    #def __iter__(self: Self) -> Self:
    #    return self

    #def __next__(self: Self) -> Iterator[OdbpOutput]:
    #    i: int = 0
    #    try:
    #        while True:
    #            v: OdbpOutput = self.outputs[i]
    #            yield v
    #            i += 1
    #    except IndexError:
    #        return

    #def __getattr__(self: Self, key: str) -> Any:
    #    try:
    #        return self.__dict__[key]
    #    except KeyError as e:
    #        name: str
    #        output_key: OdbpOutput
    #        for name, output_key in self.outputs_by_names.items():
    #            if key.strip().lower() == name.strip("_").lower():
    #                return output_key

    #        raise e


#class ExtremaDict(UserDict):
#    def __init__(self, bounds: dict[str, tuple[Any, Any]]) -> None:
#        self.bounds = bounds
#        self.data: dict[str, tuple[Any | None, Any | None]] = self.bounds.copy()
#
#    def __setitem__(self, key: str, value: Any) -> None:
#        target_key: str
#        old_data: list[Any]
#        old_key: str
#
#        for old_key in self.data.keys():
#            if key.lower() == old_key.lower():
#                if not isinstance(value, Collection) or not (len(value) == 2):
#                    raise KeyError(f'To set "{key}", please pass a two-element Collection for the upper- and lower-bounds (these will be sorted). Alternatively set "{key}_upper" and "{key}_lower" individually. To set the same value for both, either pass a 2 element Collection with the same value twice, or set "{key}_both"')
#                upper_val: Any
#                lower_val: Any
#                lower_val, upper_val = sorted(value)
#                self.data[old_key] = (lower_val, upper_val)
#                return
#        
#            elif key.endswith("_lower"):
#                target_key = key[:-6]
#                if target_key.lower() == old_key.lower():
#                    if isinstance(value, Collection):
#                        if len(value) > 1:
#                            raise KeyError(f'To set "{key}", please pass only a single value')
#                        else:
#                            value = value[0]
#
#                    old_data = list(self.data[old_key])
#                    if value > old_data[1]:
#                        self.data[old_key] = (value, self.bounds[old_key][1])
#                    else:
#                        self.data[old_key] = (value, old_data[1])
#                    return
#                
#            elif key.endswith("_upper"):
#                target_key = key[:-6]
#                if target_key.lower() == old_key.lower():
#                    if isinstance(value, Collection):
#                        if len(value) > 1:
#                            raise KeyError(f'To set "{key}", please pass only a single value')
#                        else:
#                            value = value[0]
#
#                    old_data = list(self.data[target_key])
#                    if value < old_data[0]:
#                        self.data[old_key] = (self.bounds[old_key][0], value)
#                    else:
#                        self.data[old_key] = (old_data[0], value)
#                    return
#                
#            elif key.endswith("_both"):
#                target_key = key[:-5]
#                if target_key.lower() == old_key.lower():
#                    if isinstance(value, Collection):
#                        if len(value) > 1:
#                            raise KeyError(f'To set "{key}", please pass only a single value')
#                        else:
#                            value = value[0]
#
#                    self.data[old_key] = (value, value)
#                    return
#
#        raise KeyError(f"{key} is not recognized")
#
#    def __getitem__(self, key: str) -> Any:
#        target_key: str
#        old_key: str
#        for old_key in self.data.keys():
#            if  key.lower() == old_key.lower():
#                return self.data[old_key]
#            
#            elif key.endswith("_lower"):
#                target_key = key[:-6]
#                if target_key.lower() == old_key.lower():
#                    return self.data[old_key][0]
#
#            elif key.endswith("_upper"):
#                target_key = key[:-6]
#                if target_key.lower() == old_key.lower():
#                    return self.data[old_key][1]
#
#            elif key.endswith("_both"):
#                target_key = key[:-5]
#                if target_key.lower() == old_key.lower():
#                    return self.data[old_key]
#
#        raise KeyError(f'"{key}" is not recognzied')
#            
#    def __delitem__(self, key: str) -> None:
#        target_key: str
#        old_key: str
#        old_data: list[Any | None]
#        for old_key in self.data.keys():
#            if key.lower() ==  old_key.lower():
#                self.data[old_key] = (self.bounds[old_key][0], self.bounds[old_key][0])
#            
#            elif key.endswith("_lower"):
#                target_key = key[:-6]
#                if target_key.lower() == old_key.lower():
#                    old_data = list(self.data[old_key])
#                    self.data[old_key] = (self.bounds[old_key][0], old_data[1])
#
#            elif key.endswith("_upper"):
#                target_key = key[:-6]
#                if target_key.lower() == old_key.lower():
#                    old_data = list(self.data[old_key])
#                    self.data[old_key] = (old_data[0], self.bounds[old_key][1])
#
#            elif key.endswith("_both"):
#                target_key = key[:-5]
#                if target_key.lower() == old_key.lower():
#                    self.data[old_key] = (self.bounds[old_key][0], self.bounds[old_key][1])
#
#        raise KeyError(f'"{key}" is not recognzied')
#
#    def get(self) -> NotImplemented:
#        return NotImplemented
#        
#    def pop(self) -> NotImplemented:
#        return NotImplemented
#        
#    def popitem(self) -> NotImplemented:
#        return NotImplemented
#        
#    def clear(self) -> NotImplemented:
#        return NotImplemented
#        
#    def update(self) -> NotImplemented:
#        return NotImplemented
#        
#    def setdefault(self) -> NotImplemented:
#        return NotImplemented


class OdbpPlotType(Enum):
    ONLY_SURFACE = auto()
    ONLY_POINT_CLOUD = auto()
    ONLY_ELEMS = auto()
    SURFACE_WITH_REMAINING_NODES = auto()
    POINT_CLOUD_WITH_REMAINING_NODES = auto()
    ELEMS_AND_SURFACE_OF_REMAINING_NODES = auto()
    ELEMS_AND_POINT_CLOUD_OF_REMAINING_NODES = auto()
        

class OdbpSettings:
    #__slots__ = (
    #    "_odb_path",
    #    "_odb_source_dir",
    #    "_odb_source_dir",
    #    "_h5_path",
    #    "_h5_source_dir",
    #    "_h5_source_dir",
    #    "_result_dir",
    #    "_result_dir",
    #    "abaqus_executable",
    #    "_outputs",
    #    "cpus",
    #    "output_mapping",
    #    "defaults_for_outputs",
    #    "_ODBPPLOTTYPE",
    #    "plot_type",
    #    "colormap", 
    #    "_views",
    #    "_view",
    #    "interactive",
    #    "filename",
    #    "save_format",
    #    "view_projection_mode",
    #    "material",
    #    "point_cloud_material",
    #    "point_cloud_dynamic_radius",
    #    "save",
    #    "_background_color",
    #    "transparent_background",
    #    "font",
    #    "font_size",
    #    "font_color",
    #    "below_range_color",
    #    "above_range_color",
    #    "axis_text_color",
    #    "self.title",
    #    "show_axes",
    #)
    def __init__(self: Self) -> None:
        self._odb_path: pathlib.Path
        # TODO
        self._odb_source_dir: pathlib.Path | None
        self._odb_source_dir = pathlib.Path.cwd().absolute() / "odbs"

        self._h5_path: pathlib.Path
        # TODO
        self._h5_source_dir: pathlib.Path | None
        self._h5_source_dir = pathlib.Path.cwd().absolute() / "h5s"

        # TODO
        self._result_dir: pathlib.Path | None
        self._result_dir = pathlib.Path.cwd().absolute() / "results"

        self.abaqus_executable: str = "abaqus"

        #self._nodes: NodeType | None = None
        #self._nodesets: list[str] | None = None
        #self._parts: list[str] | None = None
        #self._steps: list[str] | None = None

        # Don't instantiate until the .hdf5 is loaded
        self._outputs: OdbpOutputs

        self.cpus: int = multiprocessing.cpu_count()

        self.output_mapping: dict[str, str] = {
            "NT11": "Temp",
            "COORD1": "X",
            "COORD2": "Y",
            "COORD3": "Z",
        }

        self.defaults_for_outputs: dict[str, float] = {
            "STATUS": 1.0
        }

        #"_nodesets",  # TODO
        #"_nodes",  # TODO
        #"_parts",  # TODO
        #"_steps",  # TODO

        if PLOTTING_AVAILABLE:
            self._ODBPPLOTTYPE: OdbpPlotType = OdbpPlotType
            self.plot_type: OdbpPlotType = self._ODBPPLOTTYPE.ONLY_ELEMS

            self.colormap: str = "turbo"

            tf: BinaryIO
            with open((pathlib.Path(__file__).parent / "data") / "odbp_views.toml", "rb") as tf:
                temp_views: dict[str, list[list[float]]] = tomllib.load(tf)

            
            self._views: dict[str, tuple[npt.NDArray, npt.NDArray]] = {k: (np.array(v1), np.array(v2)) for k, (v1, v2) in temp_views.items()}

            self._view: str = "PXPYPZ-PZ"

            self.interactive: bool = True

            self.filename: str = ""
            self.save_format: str = ".png"

            self.view_projection_mode: str = "perspective"
            self.material: str = "flat"
            self.point_cloud_material: str = "clay"
            self.point_cloud_dynamic_radius: bool = False
            self.save: bool = True
            self._background_color: tuple[float, float, float] | tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
            self.transparent_background: bool = False
 
            # TODO vvv
            self.font: str = "courier"
            self.font_size: int = 16
            # Any because we can't rely on pyvista.colorlike
            self.font_color: Any = "#000000"
            self.below_range_color: Any = "#000000"
            self.above_range_color: Any = "#A0A0A0"
            self.axis_text_color: Any = "#000000"
            self.title: str = ""
            self.show_axes: bool = True
            # TODO ^^^

    #@property
    #def nodes(self) -> NodeType | None:
    #    return self._nodes

    #@nodes.setter
    #def nodes(self, value: NodeType) -> None:
    #    self._nodes = value

    #@property
    #def nodesets(self) -> list[str] | None:
    #    return self._nodesets

    #@nodesets.setter
    #def nodesets(self, value: list[str]) -> None:
    #    self._nodesets = value

    #@property
    #def parts(self) -> list[str] | None:
    #    return self._parts

    #@parts.setter
    #def parts(self, value: list[str]) -> None:
    #    self._parts = value

    #@property
    #def steps(self) -> list[str] | None:
    #    return self._steps

    #@steps.setter
    #def steps(self, value: list[str]) -> None:
    #    self._steps = value

    @property
    def odb_source_dir(self: Self) -> pathlib.Path | None:
        return self._odb_source_dir

    @odb_source_dir.setter
    def odb_source_dir(self: Self, value: pathlib.Path) -> None:
        value = pathlib.Path(value)

        if not value.exists():
            raise FileNotFoundError(f"Directory {value} does not exist")

        self._odb_source_dir = value

    @property
    def odb_path(self: Self) -> pathlib.Path | None:
        return self._odb_path

    @odb_path.setter
    def odb_path(self: Self, value: pathlib.Path) -> None:
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
    def h5_source_dir(self: Self) -> pathlib.Path | None:
        return self._h5_source_dir

    @h5_source_dir.setter
    def h5_source_dir(self: Self, value: pathlib.Path) -> None:
        value = pathlib.Path(value)

        if not value.exists():
            raise FileNotFoundError(f"Directory {value} does not exist")

        self._h5_source_dir = value

    @property
    def h5_path(self: Self) -> pathlib.Path:
        return self._h5_path

    @h5_path.setter
    def h5_path(self: Self, value: pathlib.Path) -> None:
        value = pathlib.Path(value)

        if (
            not value.is_absolute()
            and hasattr(self, "h5_source_dir")
            and self.h5_source_dir is not None
        ):
            value = self.h5_source_dir / value

        if value.exists():
            # Ensure magic numbers
            if not ensure_magic(value, H5_MAGIC_NUM):
                raise ValueError(
                    f"Given file {value} is not"
                    " a .hdf5 hierarchical data format file."
                )

        self._h5_path = value

    @property
    def result_dir(self: Self) -> pathlib.Path | None:
        return self._result_dir

    @result_dir.setter
    def result_dir(self: Self, value: pathlib.Path) -> None:
        value = pathlib.Path(value)

        if not value.exists():
            value.mkdir()

        self._result_dir = value

    @property
    def view(self: Self) -> str:
        return self._view

    @view.setter
    def view(self: Self, value: str) -> None:
        value = value.upper()
        if value not in self._views.keys():
            raise ValueError(f"{value} is not a valid view")
        self._view = value

    @property
    def background_color(self: Self) -> tuple[float, float, float] | tuple[float, float, float, float]:
        return self._background_color

    @background_color.setter
    def background_color(self: Self, val: Any) -> None:
        if isinstance(val, Collection):
            if len(val) == 3 or len(val) == 4:
                v: Any
                if all(isinstance(v, int) for v in val):
                    val = [v / 255.0 for v in val]

                if all(isinstance(v, float) for v in val):
                    if all(0.0 <= v <= 1.0 for v in val):
                        self._background_color = tuple(val)
                        return

        raise ValueError("background_color must be a 3- or 4-tuple of RGB/RGBA values. Either Between 0.0 and 1.0, or between 0 and 255 (converted to float)")

    @property
    def outputs(self: Self) -> OdbpOutputs:
        if hasattr(self, "_outputs"):
            return self._outputs

        raise AttributeError("Output ranges can only be set after load_h5() has been called")

    @property
    def ODBPPLOTYPE(self: Self) -> OdbpPlotType:
        return self._ODBPPLOTTYPE

    #def get_odb_settings_state(self) -> str:
    #    result: str = "Current state of the ODB Object:"
#
    #    # Files
    #    result += "\n\nFiles:"
    #    h5_file: str = str(self.h5_path) if hasattr(self, "h5_path") else "Not Set"
    #    odb_file: str = str(self.odb_path) if hasattr(self, "odb_path") else "Not Set"
    #    result += f"\n\n\t.hdf5 file: {h5_file}"
    #    result += f"\n\t.odb file: {odb_file}"
    #    h5_source_dir: str = (
    #        str(self.h5_source_dir) if hasattr(self, "h5_source_dir") else "Not Set"
    #    )
    #    odb_source_dir: str = (
    #        str(self.odb_source_dir) if hasattr(self, "odb_source_dir") else "Not Set"
    #    )
    #    result_dir: str = (
    #        str(self.result_dir) if hasattr(self, "result_dir") else "Not Set"
    #    )
    #    result += f"\n\n\tSource Directory for .hdf5 files: {h5_source_dir}"
    #    result += f"\n\tSource Directory for .odb files: {odb_source_dir}"
    #    result += f"\n\tDirectory for resulting images: {result_dir}"
#
    #    # Ranges
    #    result += "\n\nRanges:\n\n\tSpatial Ranges:"
#
    #    x_low: str = str(self.x_low) if hasattr(self, "x_low") else "Not Set"
    #    x_high: str = str(self.x_high) if hasattr(self, "x_high") else "Not Set"
    #    result += f"\n\t\tX Range: {x_low} to {x_high}"
#
    #    y_low: str = str(self.y_low) if hasattr(self, "y_low") else "Not Set"
    #    y_high: str = str(self.y_high) if hasattr(self, "y_high") else "Not Set"
    #    result += f"\n\t\tY Range: {y_low} to {y_high}"
#
    #    z_low: str = str(self.z_low) if hasattr(self, "z_low") else "Not Set"
    #    z_high: str = str(self.z_high) if hasattr(self, "z_high") else "Not Set"
    #    result += f"\n\t\tZ Range: {z_low} to {z_high}"
#
    #    result += "\n\n\tTemporal Range:"
    #    time_low: str = str(self.time_low) if hasattr(self, "time_low") else "Not Set"
    #    time_high: str = (
    #        str(self.time_high) if hasattr(self, "time_high") else "Not Set"
    #    )
    #    result += f"\n\t\tTime Range: {time_low} to {time_high}"
#
    #    result += "\n\n\tThermal Range:"
    #    temp_low: str = str(self.temp_low) if hasattr(self, "temp_low") else "Not Set"
    #    temp_high: str = (
    #        str(self.temp_high) if hasattr(self, "temp_high") else "Not Set"
    #    )
    #    result += f"\n\t\tTemperature Range: {temp_low} to {temp_high}"
#
    #    result += "\n\nProcessing:"
    #    abaqus_executable: str = (
    #        self.abaqus_executable if hasattr(self, "abaqus_executable") else "Not Set"
    #    )
    #    result += f"\n\n\tAbaqus Executable: {abaqus_executable}"
    #    cpus: str = str(self.cpus) if hasattr(self, "cpus") else "Not Set"
    #    result += f"\n\tNumber of CPU Cores to Use: {cpus}"
#
    #    result += "\n\nSelected Values:"
    #    nodesets: str = (
    #        str(self.nodesets)
    #        if hasattr(self, "nodesets") and self.nodesets is not None
    #        else "Not Set"
    #    )
    #    parts: str = (
    #        str(self.parts)
    #        if hasattr(self, "parts") and self.parts is not None
    #        else "Not Set"
    #    )
    #    steps: str = (
    #        str(self.steps)
    #        if hasattr(self, "steps") and self.steps is not None
    #        else "Not Set"
    #    )
    #    nodes: str = "All Nodes"
    #    # nodes: str = ( # TODO
    #    #    str(self.parse_chain(self.nodes))
    #    #    if hasattr(self, "nodes") and self.nodes is not None
    #    #    else "All Nodes"
    #    # )
    #    result += f"\n\tSelected Nodesets: {nodesets}"
    #    result += f"\n\tSelected Parts: {parts}"
    #    result += f"\n\tSelected Steps: {steps}"
    #    result += f"\n\tSelected Nodes: {nodes}"
#
    #    view: str = str(self.view) if hasattr(self, "view") else "Not Set"
    #    colormap: str = str(self.colormap) if hasattr(self, "colormap") else "Not Set"
    #    save_format: str = (
    #        str(self.save_format) if hasattr(self, "save_format") else "Not Set"
    #    )
    #    font: str = str(self.font) if hasattr(self, "font") else "Not Set"
    #    font_color: str = (
    #        str(self.font_color) if hasattr(self, "font_color") else "Not Set"
    #    )
    #    font_size: str = (
    #        str(self.font_size) if hasattr(self, "font_size") else "Not Set"
    #    )
    #    background_color: str = (
    #        str(self.background_color)
    #        if hasattr(self, "background_color")
    #        else "Not Set"
    #    )
    #    below_range_color: str = (
    #        str(self.below_range_color)
    #        if hasattr(self, "below_range_color")
    #        else "Not Set"
    #    )
    #    above_range_color: str = (
    #        str(self.above_range_color)
    #        if hasattr(self, "above_range_color")
    #        else "Not Set"
    #    )
#
    #    result += "\n\nPlotting Options:"
    #    result += f"\n\tViewing Angle: {view}"
    #    result += f"\n\tInteractive Viewing?: {'Yes' if self.interactive else 'No'}"
    #    result += f"\n\tColormap: {colormap}"
    #    result += f"\n\tWill images be saved: {'Yes' if self.save else 'No'}"
    #    result += f"\n\tImage format as which images will be saved: {save_format}"
    #    result += f"\n\tFont family/size/color: {font}/{font_size}/{font_color}"
    #    result += f"\n\tImage Background Color: {background_color}"
    #    result += f"\n\tColor for values below given range: {below_range_color}"
    #    result += f"\n\tColor for values above given range: {above_range_color}"
#
    #    result += f"\n\nFilename under which images are saved: {self.filename}"
    #    result += f"\nTitle placed on images: {self.title}"
    #    result += "\n\nODB Data Loaded: "
#
    #    return result
#