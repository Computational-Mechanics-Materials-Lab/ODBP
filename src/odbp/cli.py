#!/usr/bin/env python3

"""
Built-in CLI for ODB Plotter, allowing for interactive system access without writing scripts
"""

import os
import sys
import cmd
import pathlib
import numpy as np
from typing import Union, List, Tuple, Dict, Optional
from .odb import Odb
from .types import DataFrameType
from .process_input import process_input
from odbp import __version__


class OdbPlotterCLI(cmd.Cmd):
    def __init__(self) -> None:
        super().__init__()
        self.prompt: str = "> "
        self.intro: str = f"ODBPlotter {__version__}"
        self.odb: Odb
        self.odb = process_input()

    # Gotta overload this method in order to get desired control flow
    def cmdloop(self, intro=None):
        """Repeatedly issue a prompt, accept input, parse an initial prefix
        off the received input, and dispatch to action methods, passing them
        the remainder of the line as argument.

        """

        self.preloop()
        if self.use_rawinput and self.completekey:
            try:
                import readline

                self.old_completer = readline.get_completer()
                readline.set_completer(self.complete)
                readline.parse_and_bind(self.completekey + ": complete")
            except ImportError:
                pass
        try:
            if intro is not None:
                self.intro = intro
            if self.intro:
                self.stdout.write(str(self.intro) + "\n")
            stop = None
            while not stop:
                try:
                    if self.cmdqueue:
                        line = self.cmdqueue.pop(0)
                    else:
                        if self.use_rawinput:
                            try:
                                line = input(self.prompt)
                            except EOFError:
                                line = "EOF"
                        else:
                            self.stdout.write(self.prompt)
                            self.stdout.flush()
                            line = self.stdin.readline()
                            if not len(line):
                                line = "EOF"
                            else:
                                line = line.rstrip("\r\n")

                    if line == "EOF":
                        raise EOFError

                    line = self.precmd(line)
                    stop = self.onecmd(line)
                    stop = self.postcmd(stop, line)
                except KeyboardInterrupt:
                    self.stdout.write(
                        "Caught a Control-C. Returning to main command line\n"
                    )

                    if os.name == "posix":
                        eof_str: str = " (or Control-D)"
                    elif os.name == "nt":
                        eof_str: str = " (or Control-Z+Return)"
                    else:
                        eof_str: str = ""

                    self.stdout.write(
                        f'Please use the "quit", "q", or "exit" commands{eof_str} to exit ODBPlotter\n'
                    )

                except EOFError:
                    self._quit()

            self.postloop()
        finally:
            if self.use_rawinput and self.completekey:
                try:
                    import readline

                    readline.set_completer(self.old_completer)
                except ImportError:
                    pass

    def _confirm(
        self, message: str, confirmation: str, default: "Optional[str]" = None
    ) -> bool:
        yes_vals: Union[Tuple[str, str], Tuple[str, str, str]] = ("yes", "y")
        no_vals: Union[Tuple[str, str], Tuple[str, str, str]] = ("no", "n")
        if isinstance(default, str):
            if default.lower() in yes_vals:
                yes_vals = ("yes", "y", "")
                confirmation += " (Y/n)? "
            elif default.lower() in no_vals:
                no_vals = ("no", "n", "")
                confirmation += " (y/N)? "

        else:
            confirmation += " (y/n)? "

        while True:
            print(f"{message}\n")
            user_input: str = input(confirmation).lower()
            if user_input in yes_vals:
                return True
            elif user_input in no_vals:
                return False
            else:
                print("Error: invalid input\n")

    # Quit and Dispatches
    def _quit(self) -> None:
        print("\nExiting")
        sys.exit(0)

    def do_quit(self, arg: str) -> None:
        """Exit gracefully (same as exit or q)"""
        _ = arg
        self._quit()

    def do_exit(self, arg: str) -> None:
        """Exit gracefully (same as quit or q)"""
        _ = arg
        self._quit()

    def do_q(self, arg: str) -> None:
        """Exit gracefully (same as quit or exit)"""
        _ = arg
        self._quit()

    def _hdf(self) -> None:
        while True:
            try:
                hdf_str: str = input(
                    "Please enter the path of the hdf5 file, or the name of the hdf5 file in the hdfs directory: "
                )
                if not hdf_str.endswith(".hdf5"):
                    hdf_str += ".hdf5"
                hdf_path: pathlib.Path = pathlib.Path(hdf_str)
                if self._confirm(f"You entered {hdf_path}", "Is this correct", "yes"):
                    self.odb.hdf_path = hdf_path
                    return

            except ValueError:
                pass

    def _odb(self) -> None:
        while True:
            try:
                odb_path = input("Please enter the path of the odb file: ")
                if self._confirm(f"You entered {odb_path}", "Is this correct?", "yes"):
                    self.odb.odb_path = odb_path

                if self._confirm(
                    "You many now convert this .odb file to a .hdf5 file."
                    "Would you like to perform this conversion now?",
                    "yes",
                ):
                    self._convert()

                else:
                    print(
                        'You may perform this conversion later with the "convert" command'
                    )

                return

            except FileNotFoundError:
                print(f"Error: {odb_path} does not exist.")

            except ValueError:
                print("Error. Invalid Input")

    def do_odb(self, arg: str) -> None:
        """Select the .odb files to convert or extract from (same as odb)"""
        _ = arg
        self._odb()

    def do_hdf(self, arg: str) -> None:
        """Select the .hdf5 file to load (same as hdf5)"""
        _ = arg
        self._hdf()

    def do_hdf5(self, arg: str) -> None:
        """Select the .hdf5 file to load (same as hdf)"""
        _ = arg
        self._hdf()

    def _convert(self) -> None:
        if not hasattr(self.odb, "odb_path"):
            while True:
                target_odb: str = input(
                    "No .odb file is selected. Please enter the target .odb file: "
                )
                if self._confirm(f"You entered {target_odb}", "Is this correct", "yes"):
                    self.odb.odb_path = target_odb
                    break

        else:
            print(f"{self.odb.odb_path} is set as current .odb file")

        if not hasattr(self.odb, "hdf_path"):
            self._hdf()

        else:
            print(f"{self.odb.hdf_path} is set as the target .hdf5 file")

        self.odb.convert()

    def do_convert(self, arg: str) -> None:
        """Convert a .odb file to a .hdf5 file"""
        _ = arg
        self._convert()

    # Extrema and dispatchers
    def do_extrema(self, arg: str) -> None:
        """Set all spatial, temporal, and thermal extrema (same as range or ranges)"""
        _ = arg
        self._extrema()

    def do_range(self, arg: str) -> None:
        """Set all spatial, temporal, and thermal extrema (same as extrema or ranges)"""
        _ = arg
        self._extrema()

    def do_ranges(self, arg: str) -> None:
        """Set all spatial, temporal, and thermal extrema (same as extrema or range)"""
        _ = arg
        self._extrema()

    def do_x(self, arg: str) -> None:
        """Set x-range (same as xs)"""
        _ = arg
        self._x()

    def do_xs(self, arg: str) -> None:
        """Set x-range (same as x)"""
        _ = arg
        self._x()

    def do_y(self, arg: str) -> None:
        """Set y-range (same as ys)"""
        _ = arg
        self._y()

    def do_ys(self, arg: str) -> None:
        """Set y-range (same as y)"""
        _ = arg
        self._y()

    def do_z(self, arg: str) -> None:
        """Set z-range (same as zs)"""
        _ = arg
        self._z()

    def do_zs(self, arg: str) -> None:
        """Set z-range (same as z)"""
        _ = arg
        self._z()

    def do_time(self, arg: str) -> None:
        """Set time range (same as times)"""
        _ = arg
        self._time()

    def do_times(self, arg: str) -> None:
        """Set time range (same as time)"""
        _ = arg
        self._time()

    def do_temp(self, arg: str) -> None:
        """Set thermal range (same as temps)"""
        _ = arg
        self._temp()

    def do_temps(self, arg: str) -> None:
        """Set thermal range (same as temp)"""
        _ = arg
        self._temp()

    def _get_value(
        self,
        val_name: str,
        default_name: Union[str, None] = None,
        default: Union[float, None] = None,
    ) -> float:
        while True:
            input_str: str = f"Enter the {val_name} you would like to use: "
            if default_name is not None and default is not None:
                input_str = input_str.replace(
                    ":", f" (Leave blank for {default_name}) "
                )
            try:
                val: str = input(input_str)
                if val == "":
                    if default_name is not None and default is not None:
                        return default
                    else:
                        raise ValueError
                else:
                    return float(val)

            except ValueError:
                print("Error, all selected coordinates and time steps must be numbers")

    def _x(self) -> None:
        while True:
            x_low: float = self._get_value("lower x", "negative infinity", -1 * np.inf)
            x_high: float = self._get_value("upper x", "infinity", np.inf)

            if self._confirm(
                f"You entered the lower x value as {x_low} and the upper x value as {x_high}.",
                "Is this correct?",
                "yes",
            ):
                self.odb.x_low = x_low
                self.odb.x_high = x_high
                return

    def _y(self) -> None:
        while True:
            y_low: float = self._get_value("lower y", "negative infinity", -1 * np.inf)
            y_high: float = self._get_value("upper y", "infinity", np.inf)

            if self._confirm(
                f"You entered the lower y value as {y_low} and the upper y value as {y_high}.",
                "Is this correct?",
                "yes",
            ):
                self.odb.y_low = y_low
                self.odb.y_high = y_high
                return

    def _z(self) -> None:
        while True:
            z_low: float = self._get_value("lower z", "negative infinity", -1 * np.inf)
            z_high: float = self._get_value("upper z", "infinity", np.inf)

            if self._confirm(
                f"You entered the lower z value as {z_low} and the upper z value as {z_high}.",
                "Is this correct?",
                "yes",
            ):
                self.odb.z_low = z_low
                self.odb.z_high = z_high
                return

    def _time(self) -> None:
        while True:
            time_low: float = self._get_value("start time", "zero", 0.0)
            time_high: float = self._get_value("end time", "infinity", np.inf)

            if self._confirm(
                f"You entered the start time value as {time_low} and the stop time value as {time_high}.",
                "Is this correct?",
                "yes",
            ):
                self.odb.time_low = time_low
                self.odb.time_high = time_high
                return

    def _temp(self) -> None:
        while True:
            temp_low: float = self._get_value("lower temperature", "zero", 0.0)
            temp_high: float = self._get_value("upper temperature", "infinity", np.inf)

            if self._confirm(
                f"You entered the lower temperature value as {temp_low} and the upper temperature value as {temp_high}.",
                "Is this correct?",
                "yes",
            ):
                self.odb.temp_low = temp_low
                self.odb.temp_high = temp_high
                return

    def _extrema(self) -> None:
        self._x()
        self._y()
        self._z()
        self._time()
        self._temp()

    def do_step(self, arg: str) -> None:
        """Set the time step (jump between frames extracted) (same as time_step)"""
        _ = arg
        self._step()

    def do_time_step(self, arg: str) -> None:
        """Set the tiem step (jump between frames extracted) (same as step)"""
        _ = arg
        self._step()

    def _step(self) -> None:
        while True:
            try:
                step_str: str = input("Enter the time step value: ")
                step: int = int(step_str)
                if step < 1:
                    raise ValueError

                self.odb.time_step = step
                return

            except ValueError:
                print("Time step must be an integer greater than or equal to 1")

    def do_filename(self, arg: str) -> None:
        """Set the filename to save images as (same as file and name)"""
        _ = arg
        self._filename()

    def do_file(self, arg: str) -> None:
        """Set the filename to save images as (same as filename and name)"""
        _ = arg
        self._filename()

    def do_name(self, arg: str) -> None:
        """Set the filename to save images as (same as filename and file)"""
        _ = arg
        self._filename()

    def _filename(self) -> None:
        while True:
            if hasattr(self.odb, "hdf_path"):
                filename: str = input(
                    f"Enter the filename to save images as. The time will be appended (Leave blank for default value {self.odb.hdf_path.stem}): "
                )
                if filename == "":
                    filename = self.odb.hdf_path.stem
            else:
                filename: str = input(
                    "Enter the filename to save images as. The time wiill be appended: "
                )

            if self._confirm(f"You entered {filename}", "Is this correct?", "yes"):
                self.odb.filename = filename
                return

    def do_title(self, arg: str) -> None:
        """Set the title for images (same as label)"""
        _ = arg
        self._title()

    def do_label(self, arg: str) -> None:
        """Set the title for images (same as title)"""
        _ = arg
        self._title()

    def _title(self) -> None:
        while True:
            if hasattr(self.odb, "filename"):
                title: str = input(
                    f"Enter the title for images (Leave blank for default value {self.odb.filename}): "
                )
                if title == "":
                    title = self.odb.filename
            else:
                title: str = input("Enter the title for images: ")

            if self._confirm(f"You entered {title}", "Is this correct?", "yes"):
                self.odb.title = title
                return

    def do_load(self, arg: str) -> None:
        """Load selected .hdf5 file (same as run and process)"""
        _ = arg
        self._load()

    def do_run(self, arg: str) -> None:
        """Load selected .hdf5 file (same as load and process)"""
        _ = arg
        self._load()

    def do_process(self, arg: str) -> None:
        """Load selected .hdf5 file (same as load and run)"""
        _ = arg
        self._load()

    def _load(self) -> None:
        try:
            self.odb.load_hdf()
        except Exception:
            print("Error accessing the .hdf5 file. Make sure it exists.")

    def do_unload(self, arg: str) -> None:
        """Unload seleted .hdf5 file (same as delete)"""
        _ = arg
        self._unload()

    def do_delete(self, arg: str) -> None:
        """Unload selected .hdf5 file (same as unload)"""
        _ = arg
        self._unload()

    def _unload(self) -> None:
        self.odb.unload_hdf()

    def do_abaqus(self, arg: str) -> None:
        """Select the Abaqus Executable (same as abq and abqpy)"""
        _ = arg
        self._abaqus()

    def do_abq(self, arg: str) -> None:
        """Select the Abaqus Executable (same as abaqus and abqpy)"""
        _ = arg
        self._abaqus()

    def do_abqpy(self, arg: str) -> None:
        """Select the Abaqus Executable (same as abaqus and abq)"""
        _ = arg
        self._abaqus()

    def _abaqus(self) -> None:
        while True:
            abaqus_executable: str = input(
                "Please enter the desired Abaqus Executable: "
            )
            if self._confirm(
                f"You entered {abaqus_executable}", "Is this correct?", "yes"
            ):
                self.odb.abaqus_executable = abaqus_executable
                return

    def do_cpu(self, arg: str) -> None:
        """Set the number of CPU cores to use (same as cpus, core, and cores)"""
        _ = arg
        self._cpus()

    def do_cpus(self, arg: str) -> None:
        """Set the number of CPU cores to use (same as cpu, core, and cores)"""
        _ = arg
        self._cpus()

    def do_core(self, arg: str) -> None:
        """Set the number of CPU cores to use (same as cpu, cpus, and cores)"""
        _ = arg
        self._cpus()

    def do_cores(self, arg: str) -> None:
        """Set the number of CPU cores to use (same as cpu, cpus, and core)"""
        _ = arg
        self._cpus()

    def _cpus(self) -> None:
        while True:
            try:
                cpus: int = int(input("Please enter the number of CPU cores to use: "))
                if cpus < 1:
                    raise ValueError

                if self._confirm(f"You entered {cpus}", "Is this correct?", "yes"):
                    self.odb.cpus = cpus
                    return

            except ValueError:
                print("Error. Number of CPU cores must be an Integer greater than 0")

    def do_color(self, arg: str) -> None:
        """Set the colormap to use (same as colors, colormap, cmap)"""
        _ = arg
        self._colormap()

    def do_colors(self, arg: str) -> None:
        """Set the colormap to use (same as color, colormap, cmap)"""
        _ = arg
        self._colormap()

    def do_colormap(self, arg: str) -> None:
        """Set the colormap to use (same as color, colors, cmap)"""
        _ = arg
        self._colormap()

    def do_cmap(self, arg: str) -> None:
        """Set the colormap to use (same as color, colors, colormap)"""
        _ = arg
        self._colormap()

    def _colormap(self) -> None:
        while True:
            colormap: str = input("Please enter the desired colormap: ")
            if self._confirm(f"You entered {colormap}.", "Is this correct?", "yes"):
                self.odb.colormap = colormap
                return

    def do_save(self, arg: str) -> None:
        """Toggle whether images should be saved"""
        _ = arg
        self.odb.save = not self.odb.save

        if self.odb.save:
            sys.stdout.write(f"Images will now be saved in {self.odb.result_dir}")
            sys.stdout.write(
                'Please use the "file", "name", or "filename" commands to format under which names these files are saved'
            )

        else:
            sys.stdout.write("Images will now be shown but not saved.")

    def do_format(self, arg: str) -> None:
        """Select the format for saving images (same as save_format)"""
        _ = arg
        self._save_format()

    def do_save_format(self, arg: str) -> None:
        """Select the format for saving images (same as format)"""
        _ = arg
        self._save_format()

    def _save_format(self):
        while True:
            save_format: str = input(
                "Please enter the format (.png, .jpeg, etc) as which to save images: "
            )
            save_format = save_format.lower()
            if not save_format.startswith("."):
                save_format = "." + save_format

            if self._confirm(f"You entered {save_format}.", "Is this correct?", "yes"):
                self.odb.save_format = save_format
                return

    def do_font(self, arg: str) -> None:
        """Set the font of the images"""
        _ = arg
        while True:
            font: str = input("Please enter the desired font: ")
            if self._confirm(f"You entered {font}.", "Is this correct?", "yes"):
                self.odb.font = font
                return

    def do_font_color(self, arg: str) -> None:
        """Set the color for fonts on images"""
        _ = arg
        while True:
            font_color: str = input("Please enter the desired font color: ")
            if self._confirm(f"You entered {font_color}.", "Is this correct?", "yes"):
                self.odb.font_color = font_color
                return

    def do_font_size(self, arg: str) -> None:
        """Set the size for fonts on iamges"""
        _ = arg
        while True:
            try:
                font_size: float = float(input("Please enter the desired font size: "))
                if font_size <= 0.0:
                    raise ValueError

                if self._confirm(
                    f"You entered {font_size}.", "Is this correct?", "yes"
                ):
                    self.odb.font_size = font_size
                    return

            except ValueError:
                print("Error. Font size must be a positive number")

    def do_background_color(self, arg: str) -> None:
        """Set the background color for images"""
        _ = arg
        while True:
            background_color = input("Please enter the desired background color: ")
            if self._confirm(
                f"You entered {background_color}", "Is this correct?", "yes"
            ):
                self.odb.background_color = background_color
                return

    def do_below_color(self, arg: str) -> None:
        """Set the color for elements below the set range (same as below_range and below_range_color)"""
        _ = arg
        self._below_color()

    def do_below_range(self, arg: str) -> None:
        """Set the color for elements below the set range (same as below_color and below_range_color)"""
        _ = arg
        self._below_color()

    def do_below_range_color(self, arg: str) -> None:
        """Set the color for elements below the set range (same as below_color and below_range)"""
        _ = arg
        self._below_color()

    def _below_color(self) -> None:
        self.odb.below_range_color = self._set_nullable_color("below")

    def do_above_color(self, arg: str) -> None:
        """Set the color for elements above the set range (same as above_range and above_range_color)"""
        _ = arg
        self._above_color()

    def do_above_range(self, arg: str) -> None:
        """Set the color for elements above the set range (same as above_color and above_range_color)"""
        _ = arg
        self._above_color()

    def do_above_range_color(self, arg: str) -> None:
        """Set the color for elements above the set range (same as above_color and above_range)"""
        _ = arg
        self._above_color()

    def _above_color(self) -> None:
        self.odb.above_range_color = self._set_nullable_color("above")

    def _set_nullable_color(self, color_type: str) -> str:
        while True:
            color: str = input(f"Please enter the desired {color_type} color: ")
            if self._confirm(f"You entered {color}.", "Is this correct?", "yes"):
                return color


    def do_axis_color(self, arg: str) -> None:
        """Set the color of the axis text"""
        _ = arg
        while True:
            color: str = input(f'Please enter the desired axis color, either "white" or "black": ')
            if self._confirm(f"You entered {color}.", "Is this correct?", "yes"):
                self.odb.axis_text_color = color
                return

    # def _parse_range(self, range_str: str, extrema: List[int, int]) -> None:
    #    final_list: List[Iterator[int]] = list()

    #    selected_vals: List[str] = list(
    #        filter(
    #            lambda x: len(x) > 0 and not x.isspace(),
    #            re.split(r"[\[\]\(\)\{\}]", range_str)
    #        )
    #    )

    #    val: str
    #    try:
    #        for val in selected_vals:
    #            split_vals = list(map(int, val.split(":")))
    #
    #            if len(split_vals) == 1:
    #                if split_vals[0].isspace():
    #                    pass # No-op in this case
    #                else:
    #                    final_list.append(range(split_vals[0], split_vals[0] + 1))

    #            elif len(split_vals) == 2:
    #                if all(len(x) == 0 for x in split_vals):
    #                    return chain(range(extrema[0], extrema[1] + 1))

    #                else:
    #                    if len(split_vals[0]) == 0:
    #                        final_list.append(range(0, split_vals[1] + 1))
    #                    elif len(split_vals[1] == 0):
    #                        final_list.append(range(split_vals[0], extrema[1] + 1))
    #                    else:
    #                        final_list.append(range(split_vals[0], split_vals[1]))

    #            elif len(split_vals) == 3:
    #                if all(len(x) == 0 for x in split_vals):
    #                    return chain(range(extrema[0], extrema[1] + 1))

    #                else:
    #                    if len(split_vals[0]) == 0:
    #                        if len(split_vals[1]) == 0:
    #                            if split_vals[2] == 1 or split_vals[2] == -1:
    #                                return chain(range(extrema[0], extrema[1] + 1))
    #                            else:
    #                                final_list.append(range(extrema[0], extrema[1] + 1, split_vals[2]))

    #                        else:
    #                            final_list.append(range(extrema[0], split_vals[1], split_vals[2]))

    #                    elif len(split_vals[1] == 0):
    #                        final_list.append(range(split_vals[0], extrema[1] + 1, split_vals[2]))

    #                    else:
    #                        final_list.append(range(split_vals[0], split_vals[1], split_vals[2]))

    #        return chain(*final_list)

    #    except ValueError:
    #        raise ValueError("Error! Values for ranges must be entered like Python slices ([#:#:#]) or single numbers (#)")

    # def do_nodes(self, arg: str) -> None:
    #    """Set a list of nodes from which to extract or to convert. Use one or more Python Slice syntax instances ([#:#:#]) or single numbers (#) """
    #    _ = arg
    #    if not hasattr(self.odb, "_node_range"):
    #        try:
    #            self.odb.get_odb_info()
    #
    #        except AttributeError:
    #            sys.stdout.write('You can only use the "nodes" command once a .odb file has been selected')
    #            return

    #    max_val: int
    #    min_val: int
    #    min_val, max_val = self.odb._node_range
    #    while True:
    #        nodes_str: str = input(f"Please enter the node or range(s) of nodes you would like to process. Possible range is from {min_val} to {max_val}, inclusive. Use one or more Python Slice syntax instances ([#:#:#]) or single numbers (#): ")

    #        try:
    #            nodes: chain = self._parse_range(nodes_str, self.odb._node_range)

    #        except ValueError as e:
    #            print(*e.args)
    #
    #        if self._confirm(f"You entered {nodes_str}", "Is this correct?", "yes"):
    #            self.odb.nodes = nodes
    #            return

    # frames
    # def do_frames(self, arg: str) -> None:
    #    """Set a list of frames from which to extract or to convert. Use one or more Python Slice syntax instances ([#:#:#]) or single numbers (#) """
    #    _ = arg
    #    if not hasattr(self.odb, "_frame_range"):
    #        try:
    #            self.odb.get_odb_info()
    #
    #        except AttributeError:
    #            print('You can only use the "frames" command once a .odb file has been selected')
    #            return

    #    max_val: int
    #    min_val: int
    #    min_val, max_val = self.odb._frame_range
    #    while True:
    #        frames_str: input(f"Please enter the frame or range(s) of frames you would like to process. Possible range is from {min_val} to {max_val}, inclusive. Use one or more Python Slice syntax instances ([#:#:#]) or single numbers (#): ")

    #        try:
    #            frames: chain = self._parse_range(frames_str, self.odb._frame_range)

    #        except ValueError as e:
    #            print(*e.args)
    #
    #        if self._confirm(f"You entered {frames_str}", "Is this correct?", "yes"):
    #            self.odb.frames = frames
    #            return

    # steps
    def do_steps(self, arg: str) -> None:
        """Add to the list of steps from which to extract or which to convert (use clear_steps to empty this list)"""
        _ = arg
        if not hasattr(self.odb, "_step_names"):
            try:
                self.odb.get_odb_info()

            except AttributeError:
                print(
                    "Warning: .odb file with list of steps is not given to compare against"
                )

        step: str = ""
        print(
            'Enter names of steps. Type "quit", "exit", "stop, "done" or use Control-C to finish'
        )
        if hasattr(self.odb, "_step_names"):
            print(f"Valid step names are: {self.odb._step_names}")
        while step.lower() not in ("quit", "exit", "stop", "done"):
            try:
                print(f"Currently selected steps: {self.odb.steps}")
                add_step: bool = True

                step = input("Enter the name of a step to add: ")

                if hasattr(self.odb, "_step_names"):
                    if step not in self.odb._step_names:
                        add_step = False

                if add_step:
                    if self.odb.steps is None:
                        self.odb.steps = list()
                    self.odb.steps.append(step)

            except KeyboardInterrupt:
                break

    def do_clear_steps(self, arg: str) -> None:
        """Clear out current list of steps (add some with steps)"""
        _ = arg

        print(f"Previous list of steps: {self.odb.steps}")
        self.odb.steps = None

        print("Step list has been cleared.")

    # parts
    def do_parts(self, arg: str) -> None:
        """Add to the list of parts from which to extract or which to convert (use clear_parts to empty this list)"""
        _ = arg
        if not hasattr(self.odb, "_part_names"):
            try:
                self.odb.get_odb_info()

            except AttributeError:
                print(
                    "Warning: .odb file with list of parts is not given to compare against"
                )

        part: str = ""
        print(
            'Enter names of parts. Type "quit", "exit", "stop, "done" or use Control-C to finish'
        )
        if hasattr(self.odb, "_part_names"):
            print(f"Valid part names are: {self.odb._part_names}")
        while part.lower() not in ("quit", "exit", "stop", "done"):
            try:
                print(f"Currently selected parts: {self.odb.parts}")
                add_part: bool = True

                part = input("Enter the name of a step to add: ")

                if hasattr(self.odb, "_part_names"):
                    if part not in self.odb._part_names:
                        add_part = False

                if add_part:
                    if self.odb.parts is None:
                        self.odb.parts = list()
                    self.odb.parts.append(part)

            except KeyboardInterrupt:
                break

    def do_clear_parts(self, arg: str) -> None:
        """Clear out current list of parts (add some with parts)"""
        _ = arg

        print(f"Previous list of parts: {self.odb.parts}")
        self.odb.parts = None

        print("Part list has been cleared.")

    # nodesets
    def do_nodesets(self, arg: str) -> None:
        """Add to the list of nodesets from which to extract or which to convert (use clear_nodesets to empty this list)"""
        _ = arg
        if not hasattr(self.odb, "_nodeset_names"):
            try:
                self.odb.get_odb_info()

            except AttributeError:
                print(
                    "Warning: .odb file with list of nodesets is not given to compare against"
                )

        nodeset: str = ""
        print(
            'Enter names of nodesets. Type "quit", "exit", "stop, "done" or use Control-C to finish'
        )
        if hasattr(self.odb, "_nodeset_names"):
            print(f"Valid nodeset names are: {self.odb._nodeset_names}")
        while nodeset.lower() not in ("quit", "exit", "stop", "done"):
            try:
                print(f"Currently selected nodesets: {self.odb.nodesets}")
                add_nodeset: bool = True

                nodeset = input("Enter the name of a nodeset to add: ")

                if hasattr(self.odb, "_nodeset_names"):
                    if nodeset not in self.odb._nodeset_names:
                        add_nodeset = False

                if add_nodeset:
                    if self.odb.nodesets is None:
                        self.odb.nodesets = list()
                    self.odb.nodesets.append(nodeset)

            except KeyboardInterrupt:
                break

    def do_clear_nodesets(self, arg: str) -> None:
        """Clear out current list of nodesets (add some with nodesets)"""
        _ = arg

        print(f"Previous list of nodesets: {self.odb.nodesets}")
        self.odb.nodesets = None

        print("Nodeset list has been cleared.")

    # angle angles view views
    def do_view(self, arg: str) -> None:
        """Set the viewing angle (same as views, angle, angles)"""
        _ = arg
        self._view()

    def do_views(self, arg: str) -> None:
        """Set the viewing angle (same as view, angle, angles)"""
        _ = arg
        self._view()

    def do_angle(self, arg: str) -> None:
        """Set the viewing angle (same as view, views, angles)"""
        _ = arg
        self._view()

    def do_angles(self, arg: str) -> None:
        """Set the viewing angle (same as view, views, angle)"""
        _ = arg
        self._view()

    def _view(self) -> None:
        views = [", ".join(k) + ": " + str(v) for k, v in self.odb._views.items()]
        print(f"Valid views are:")
        for v in views:
            print(v)
        while True:
            view: str = input("Enter desired view: ")

            view_valid = False
            for k in self.odb._views.keys():
                if view in k:
                    view_valid = True
                    if self._confirm(f"You entered {view}.", "Is this correct?", "yes"):
                        self.odb.view = view
                        return
                    else:
                        break
 
            if not view_valid:
                print("Error. Invalid View")

    def do_interactive(self, arg: str) -> None:
        """Toggle between ineractive or non-interactive plotting"""
        self.odb.interactive = not self.odb.interactive
        print(f"Interactive plotting toggled to {self.odb.interactive}")

    def do_axis(self, arg: str) -> None:
        """Show or hide the 3D axes (same as axes)"""
        _ = arg
        self._axes()

    def do_axes(self, arg: str) -> None:
        """Show or hide the 3D axes (same as axis)"""
        _ = arg
        self._axes()

    def _axes(self) -> None:
        self.odb.show_axes = not self.odb.show_axes
        print(f"Show axes toggled to {self.odb.show_axes}")

    # print state setting settings status
    def do_print(self, arg: str) -> None:
        """Output current state (same as state, setting, settings, status)"""
        _ = arg
        self._state()

    def do_state(self, arg: str) -> None:
        """Output current state (same as print, setting, settings, status)"""
        _ = arg
        self._state()

    def do_setting(self, arg: str) -> None:
        """Output current state (same as print, state, settings, status)"""
        _ = arg
        self._state()

    def do_settings(self, arg: str) -> None:
        """Output current state (same as print, state, setting, status)"""
        _ = arg
        self._state()

    def do_status(self, arg: str) -> None:
        """Output current state (same as print, state, setting, settings)"""
        _ = arg
        self._state()

    def _state(self) -> None:
        result: str = self.odb.get_odb_settings_state()
        if hasattr(self.odb, "odb"):
            result += "True"
        else:
            result += "False"

        print(result)

    def do_coord(self, arg: str) -> None:
        """Set the name of the spatial coordinate key (same as coords, coordinate, coordinates)"""
        _ = arg
        self._coord()

    def do_coords(self, arg: str) -> None:
        """Set the name of the spatial coordinate key (same as coord, coordinate, coordinates)"""
        _ = arg
        self._coord()

    def do_coordinate(self, arg: str) -> None:
        """Set the name of the spatial coordinate key (same as coord, coords, coordinates)"""
        _ = arg
        self._coord()

    def do_coordinates(self, arg: str) -> None:
        """Set the name of the spatial coordinate key (same as coord, coords, coordinate)"""
        _ = arg
        self._coord()

    def _coord(self) -> None:
        while True:
            if hasattr(self.odb, "_frame_keys"):
                print(f"Valid potential keys: {self.odb._frame_keys}")

            coord_key: str = input("Please enter the desired coordinate key: ")
            if self.confirm(f"You entered {coord_key}.", "Is this correct?", "yes"):
                self.odb.coord_key = coord_key
                return

    def do_target_outputs(self, arg: str) -> None:
        """Add to the list of target output keys"""
        _ = arg
        if hasattr(self.odb, "_frame_keys"):
            print(f"Valid potential keys: {self.odb._frame_keys}")

        target_output: str = ""
        print(
            'Enter desired keys. Type "quit", "exit", "stop, "done" or use Control-C to finish'
        )
        while target_output.lower() not in ("quit", "exit", "stop", "done"):
            try:
                print(f"Currently selected target outputs: {self.odb.target_outputs}")

                target_output = input("Enter the key to add: ")

                if self.odb.target_outputs is None:
                    self.odb.target_outputs = list()
                self.odb.target_outputs.append(target_output)

            except KeyboardInterrupt:
                break

    def do_clear_target_outputs(self, arg: str) -> None:
        """Clear out current list of target outputs (add some with target_outputs)"""
        _ = arg

        print(f"Previous list of target outputs: {self.odb.target_outputs}")
        self.odb.nodesets = None

        print("Target outputs list has been cleared.")

    def do_get_hdf_status(self, arg: str) -> None:
        """Get the metadata about the loaded .hdf5"""
        _ = arg
        if hasattr(self.odb, "hdf_status"):
            if hasattr(self.odb, "hdf_path"):
                print(f"Metadata for {self.odb.hdf_path}:")
            else:
                print("Metadata for Currently Loaded .hdf5 file:")
            k: str
            v: str
            for k, v in self.odb.hdf_status.items():
                print(f"\t{k}:{v}")

        else:
            print("No .hdf5 file is current loaded.")

    # TODO Other types/scripts
    def do_extract(self, arg: str) -> None:
        """Get extracted minimum, mean, and maximum values from a .odb or a .hdf5"""
        _ = arg
        while True:
            target: str = input(
                'Please enter the file to which you\'d like to save the extracted .csv (or enter "show" to print these values).'
                "If you do not enter an absolute path, the file will be saved in the results directory if it is set"
            )
            if self._confirm(f"You entered{target}.", "Is this correct?", "yes"):
                results_df: DataFrameType = self.extract()
                if target == "show":
                    print(results_df)

                else:
                    target_path: pathlib.Path = pathlib.Path(target_path)
                    if not target_path.absolute() and hasattr(self.odb, "result_dir"):
                        results_df.to_csv(self.odb.result_dir / target_path)

                    else:
                        results_df.to_csv(target_path)

    def do_get_odb_info(self, arg: str) -> None:
        """Get data about a .odb file before extraction or conversion"""
        _ = arg

        if hasattr(self.odb, "odb_path"):
            self.odb.get_odb_info()
            print(f"Data for {self.odb.odb_path}")
            print(f"Frame Range: {self.odb.frame_range}")
            print(f"Frame Keys: {self.odb.frame_keys}")
            print(f"Frame Keys Per Step: {self.odb.frame_keys_per_step}")
            print(f"Step Names: {self.odb.step_names}")
            print(f"Step Lengths: {self.odb.step_lens}")
            print(f"Nodeset Names: {self.odb.nodeset_names}")
            print(f"Part Names: {self.odb.part_names}")
            print(f"Node Range: {self.odb.node_range}")
            print(f"Node Ranges Per Part: {self.odb.node_ranges_per_part}")

        else:
            while True:
                try:
                    odb_path_str: str = input(
                        "Please enter the path of the file from which you would like to extract: "
                    )
                    if self._confirm(
                        f"You entered {odb_path_str}.", "Is this correct?", "yes"
                    ):
                        odb_path: pathlib.Path = pathlib.Path(odb_path_str)
                        final_path: pathlib.Path
                        if odb_path.exists():
                            final_path = odb_path

                        elif (
                            hasattr(self.odb, "odb_source_dir")
                            and (self.odb.odb_source_dir / odb_path).exists()
                        ):
                            final_path = self.odb.odb_source_dir / odb_path

                        elif (pathlib.Path.cwd() / odb_path).exists():
                            final_path = pathlib.Path.cwd() / odb_path

                        else:
                            raise ValueError

                        result: Dict[
                            str,
                            Union[
                                Tuple[int, int], List[str], Dict[str, Tuple[int, int]]
                            ],
                        ]
                        result = self.odb.get_odb_info_from_file(final_path)

                        k: str
                        v: Union[Tuple[int, int], List[str], Dict[str, Tuple[int, int]]]
                        print(f"Data for {final_path}")
                        for k, v in result.items():
                            print(f"\t{k}: {v}")

                        return

                except ValueError:
                    print(f"Error: File {odb_path} could not be found")

    # TODO
    # TODO
    # TODO
    # dir dirs directory directories

    def do_plot_val_v_time(self, arg: str) -> None:
        """2D Plot of mean or max values for a given key (same as plot_val_vs_time)"""
        _ = arg
        self._plot_val_v_time()

    def do_plot_val_vs_time(self, arg: str) -> None:
        """2D Plot of mean or max values for a given key (same as plot_val_v_time)"""
        _ = arg
        self._plot_val_v_time()

    def _plot_val_v_time(self) -> None:
        print(
            f"Potential values to plot: {self.odb.target_outputs if hasattr(self.odb, 'target_outputs') and self.odb.target_outputs is not None else self.odb._frame_keys if hasattr(self.odb, '_frame_keys') and self.odb._frame_keys is not None else 'All Outputs'}"
        )
        while True:
            chosen_output: str = input("Select output: ")
            if (
                not hasattr(self.odb, "target_outputs")
                or self.odb.target_outputs is None
            ) or (
                hasattr(self.odb, "target_outputs")
                and chosen_output in self.odb.target_outputs
            ):
                if self._confirm(
                    f"You entered {chosen_output}.", "Is this correct?", "yes"
                ):
                    chosen_range: str = ""
                    while chosen_range not in ("max", "mean", "both"):
                        chosen_range: str = input(
                            "Would you like to plot 'max', 'mean', or 'both'? "
                        ).lower()
                        if chosen_range in ("max", "mean", "both"):
                            if self._confirm(
                                f"You entered {chosen_range}.",
                                "Is this correct?",
                                "yes",
                            ):
                                result: Optional[str] = self.odb.plot_key_versus_time(
                                    chosen_output, chosen_range
                                )
                                if isinstance(result, str):
                                    print(f"Results saved to {result}")
                                    return

                        else:
                            print("Invalid choice")

            else:
                print("Invalid choice")

    # TODO temp range
    def do_plot_3d(self, arg: str) -> None:
        """3D plot of .odb data over time"""
        _ = arg
        print(
            f"Potential values to plot: {self.odb.target_outputs if hasattr(self.odb, 'target_outputs') and self.odb.target_outputs is not None else self.odb._frame_keys if hasattr(self.odb, '_frame_keys') and self.odb._frame_keys is not None else 'All Outputs'}"
        )
        while True:
            chosen_output: str = input("Select output: ")
            if (
                not hasattr(self.odb, "target_outputs")
                or self.odb.target_outputs is None
                or chosen_output in self.odb.target_outputs
            ):
                if self._confirm(
                    f"You entered {chosen_output}.", "Is this correct?", "yes"
                ):
                    results: List[pathlib.Path] = self.odb.plot_3d_all_times(
                        chosen_output, title=self.odb.title
                    )
                    print(f"Results saved to: {results}")
                    return

            else:
                print("Invalid choice")

    def do_plot_meltpool(self, arg: str) -> None:
        """3D Plot of thermal meltpool over time"""
        _ = arg
        print(
            f"Select one of these as the thermal values: {self.odb.target_outputs if hasattr(self.odb, 'target_outputs') and self.odb.target_outputs is not None else self.odb._frame_keys if hasattr(self.odb, '_frame_keys') and self.odb._frame_keys is not None else 'All Outputs'}"
        )
        while True:
            temp: str = input("Select the name of the temperature values: ")
            if self._confirm(f"You entered {temp}.", "Is this correct?", "yes"):
                target_nodes: DataFrameType = self.odb.odb[
                    self.odb.odb[temp] >= self.odb.temp_high
                ]
                results: List[pathlib.Path] = self.odb.plot_3d_all_times(
                    temp,
                    title=self.odb.title,
                    target_nodes=target_nodes,
                    plot_type="meltpool",
                )
                print(f"Results saved to: {results}")
                return

    def do_plot_node(self, arg: str) -> None:
        """Plot a value over time for one node"""
        _ = arg
        print(
            f"Potential values to plot: {self.odb.target_outputs if hasattr(self.odb, 'target_outputs') and self.odb.target_outputs is not None else self.odb._frame_keys if hasattr(self.odb, '_frame_keys') and self.odb._frame_keys is not None else 'All Outputs'}"
        )
        while True:
            target_output: str = input(
                "Select the name of the value to plot over time: "
            )
            if self._confirm(
                f"You entered {target_output}.", "Is this correct?", "yes"
            ):
                while True:
                    try:
                        target_node: int = int(
                            input(
                                "Select one node by integer index which you would like to plot: "
                            )
                        )
                        if self._confirm(
                            f"You entered {target_node}.", "Is this correct?", "yes"
                        ):
                            result: pathlib.Path = self.odb.plot_single_node(
                                target_output, target_node
                            )
                            print(f"Results saved to {result}")
                            return

                    except ValueError:
                        print("Target node must be an integer")
