#!/usr/bin/env python3

"""
Built-in CLI for ODB Plotter, allowing for interactive system access without writing scripts
"""

import sys
import cmd
import pathlib
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
import numpy as np
import pandas as pd
from typing import Any, Union, TextIO, List, Tuple, Dict, Optional
from .odb import Odb
from .state import process_input, print_state, load_views_dict
from odbp import __version__


class OdbPlotterCLIOptions():
    """Simple struct to store a few options that do not belong directly
    in the Odb object"""

    __slots__ = (
        "filename", # Done
        "title", # Done
        "config_file_path", # TODO!!!
        "run_immediate",
    )

    def __init__(self) -> None:
        self.filename: str = ""
        self.title: str = ""
        self.config_file_path: Optional[str] = None
        self.run_immediate: bool = False


class OdbPlotterCLI(cmd.Cmd):
    
    def __init__(self) -> None:
        super().__init__()
        self.prompt: str = "> "
        self.intro: str = f"ODBPlotter {__version__}"
        self.state: Odb
        self.options: OdbPlotterCLIOptions
        #self.state, self.options = process_input()
        self.odb = Odb()
        self.options = OdbPlotterCLIOptions()


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
                readline.parse_and_bind(self.completekey+": complete")
            except ImportError:
                pass
        try:
            if intro is not None:
                self.intro = intro
            if self.intro:
                self.stdout.write(str(self.intro)+"\n")
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
                                line = 'EOF'
                        else:
                            self.stdout.write(self.prompt)
                            self.stdout.flush()
                            line = self.stdin.readline()
                            if not len(line):
                                line = 'EOF'
                            else:
                                line = line.rstrip('\r\n')
                    line = self.precmd(line)
                    stop = self.onecmd(line)
                    stop = self.postcmd(stop, line)
                except KeyboardInterrupt:
                    self.stdout.write("Caught a Control-C. Returning to main command line")
                    self.stdout.write('Please use the "quit", "q", or "exit" commands (or Control-D) to exit ODBPlotter\n')

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


    def _confirm(self, message: str, confirmation: str, default: "Optional[str]" = None) -> bool:
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
            self.stdout.write(f"{message}\n")
            user_input: str = input(confirmation).lower()
            if user_input in yes_vals:
                return True
            elif user_input in no_vals:
                return False
            else:
                self.stdout.write("Error: invalid input\n")


    # Quit and Dispatches
    def _quit(self) -> None:
        self.stdout.write("\nExiting")
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
                hdf_str: str = input("Please enter the path of the hdf5 file, or the name of the hdf5 file in the hdfs directory: ")
                if not hdf_str.endswith(".hdf5"):
                    hdf_str += ".hdf5"
                hdf_path: pathlib.Path = pathlib.Path(hdf_str)
                if(self._confirm(f"You entered {hdf_path}", "Is this correct", "yes")):
                    self.odb.hdf_path = hdf_path
                    # TODO!!!
                    #pre_process_data(self.odb, user_options)
                    return

            except ValueError:
                pass


    def _odb(self) -> None:
        while True:
            try:
                odb_path = input("Please enter the path of the odb file: ")
                if self._confirm(f"You entered {odb_path}", "Is thsi correct?", "yes"):
                    self.odb.odb_path = odb_path

                    # TODO
                    # handle time step
                    #

                if self._confirm("You many now convert this .odb file to a .hdf5 file." "Would you like to perform this conversion now?", "yes"):
                    self._convert()

                else:
                    self.stdout.write('You may perform this conversion later with the "convert" command')

            except (FileNotFoundError, ValueError):
                pass


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
        if not hasattr(self, "odb_path"):
            while True:
                target_odb: str = input("No .odb file is selected. Please enter the target .odb file: ")
                if self._confirm(f"You entered {target_odb}", "Is this correct", "yes"):
                    self.odb.odb_path = target_odb
                    break

        else:
            self.stdout.write(f"{self.odb.odb_path} is set as current .odb file")

        if not hasattr(self, "hdf_path"):
            self._hdf()

        else:
            self.stdout.write(f"{self.odb.hdf_path} is set as the target .hdf5 file")

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
        default: Union[float, None] = None
        ) -> float:
        while True:
            input_str: str = f"Enter the {val_name} you would like to use: "
            if default_name is not None and default is not None:
                input_str = input_str.replace(":", f" (Leave blank for {default_name}) ")
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
                self.stdout.write("Error, all selected coordinates and time steps must be numbers")


    def _x(self) -> None:
        
        while True:
            x_low: float = self._get_value("lower x", "negative infinity", -1*np.inf)
            x_high: float = self._get_value("upper x", "infinity", np.inf)

            if self._confirm(f"You entered the lower x value as {x_low} and the upper x value as {x_high}.", "Is this correct?", "yes"):
                self.odb.x_low = x_low
                self.odb.x_high = x_high
                return


    def _y(self) -> None:
        
        while True:
            y_low: float = self._get_value("lower y", "negative infinity", -1*np.inf)
            y_high: float = self._get_value("upper y", "infinity", np.inf)

            if self._confirm(f"You entered the lower y value as {y_low} and the upper y value as {y_high}.", "Is this correct?", "yes"):
                self.odb.y_low = y_low
                self.odb.y_high = y_high
                return


    def _z(self) -> None:
        
        while True:
            z_low: float = self._get_value("lower z", "negative infinity", -1*np.inf)
            z_high: float = self._get_value("upper z", "infinity", np.inf)

            if self._confirm(f"You entered the lower z value as {z_low} and the upper z value as {z_high}.", "Is this correct?", "yes"):
                self.odb.z_low = z_low
                self.odb.z_high = z_high
                return


    def _time(self) -> None:

        while True:
            time_low: float = self._get_value("start time", "zero", 0.0)
            time_high: float = self._get_value("end time", "infinity", np.inf)

            if self._confirm(f"You entered the start time value as {time_low} and the stop time value as {time_high}.", "Is this correct?", "yes"):
                self.odb.time_low = time_low
                self.odb.time_high = time_high
                return


    def _temp(self) -> None:

        while True:
            temp_low: float = self._get_value("lower temperature", "zero", 0.0)
            temp_high: float = self._get_value("upper temperature", "infinity", np.inf)

            if self._confirm(f"You entered the lower temperature value as {temp_low} and the upper temperature value as {temp_high}.", "Is this correct?", "yes"):
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
        """Set the time step (jump between frames extracted)"""
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
                self.stdout.write("Time step must be an integer greater than or equal to 1")


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
                filename: str = input(f"Enter the filename to save images as. The time will be appended (Leave blank for default value {self.odb.hdf_path.stem}): ")
                if filename == "":
                    filename = self.odb.hdf_path.stem                
            else:
                filename: str = input("Enter the filename to save images as. The time wiill be appended: ")

            if self._confirm(f"You entered {filename}", "Is this correct?", "yes"):
                self.options.filename = filename
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
            if hasattr(self.options, "filename"):
                title: str = input(f"Enter the title for images (Leave blank for default value {self.options.filename}): ")
                if title == "":
                    title = self.options.filename
            else:
                title: str = input("Enter the title for images: ")

            if self._confirm(f"You entered {title}", "Is this correct?", "yes"):
                self.options.title = title
                return


    # TODO ???
    #def do_dirs(self, arg: str) -> None:
    #    """Set the """
    #self.directory_options: List[str] = ["dir", "dirs", "directory", "directories"]
    #self.directory_help: str = "Set the source and output directories"
    #self.directory_options_formatted: str = ", ".join(self.directory_options)


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
        self.odb.load_hdf()


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


def cli() -> None:

    main_loop: bool = True

    # TODO Process input toml file and/or cli switches here
    state: Odb
    user_options: UserOptions
    result: Union[Tuple[Odb, UserOptions], pd.DataFrame]
    result = process_input()
    print(f"ODBPlotter {__version__}")
    if isinstance(result, pd.DataFrame):
        sys.exit(print(result))

    else:
        state, user_options = result

    cli_options: CLIOptions = CLIOptions()

    if user_options.run_immediate:
        # TODO
        load_hdf(state)
        plot_time_range(state, user_options)

    while main_loop:
        try:
            user_input:str = input("\n> ").strip().lower()

            if user_input in cli_options.directory_options:
                set_directories(user_options)

            elif user_input in cli_options.angle_options:
                set_views(state)

            elif user_input in cli_options.show_all_options:
                state.interactive = not state.interactive
                print(f"Plots will now {'BE' if state.interactive else 'NOT BE'} shown")

            elif user_input in cli_options.abaqus_options:
                set_abaqus(state)

            elif user_input in cli_options.nodeset_options:
                set_nodeset(state)

            elif user_input in cli_options.state_options:
                print_state(state, user_options)

def pre_process_data(state: Odb, user_options: UserOptions):
    meltpoint: Optional[float] = None
    low_temp: Optional[float] = None
    time_sample: Optional[int] = None

    if user_options.config_file_path is not None:
        config_file: TextIO
        with open(user_options.config_file_path, "rb") as config_file:
            config: Dict[str, Any] = tomllib.load(config_file)

        if "hdf_file_path" in config:
            if config["hdf_file_path"] != state.hdf_file_path:
                print("INFO: File name provided and File Name in the config do not match. This could be an issue, or it might be fine")

        if "meltpoint" in config:
            meltpoint = config["meltpoint"]

        if "low_temp" in config:
            low_temp = config["low_temp"]

        if "time_sample" in config:
            time_sample = config["time_sample"]

        # Manage meltpoint
        if meltpoint is not None:
            print(f"Setting Melting Point to stored value of {meltpoint}")
            state.set_meltpoint(meltpoint)

        elif hasattr(state, "meltpoint"):
            print(f"Setting Default Melting Point to given value of {state.meltpoint}")
        
        else: # Neither the stored value nor the given value exist
            print("No Melting Point found. You must set it:")
            set_meltpoint(state)

        # Manage lower temperature bound
        if low_temp is not None:
            print(f"Setting Melting Point to stored value of {low_temp}")
            state.set_low_temp(low_temp)

        elif hasattr(state, "low_temp"):
            print(f"Setting Default Melting Point to given value of {state.low_temp}")
        
        else: # Neither the stored value nor the given value exist
            print("No Lower Temperature Bound found. You must set it:")
            set_low_temp(state)

        # Manage time sample
        if time_sample is not None:
            print(f"Setting Time Sample to stored value of {time_sample}")
            state.set_time_sample(time_sample)

        elif hasattr(state, "time_sample"):
            print(f"Setting Default Time Sample to given value of {state.time_sample}")

        else: # Neither the stored value nor the given value exist
            print("No Time Sample found. You must set it:")
            set_time_sample(state)

    if not all(hasattr(state, "meltpoint"), hasattr(state, "low_temp"), hasattr(state, "time_sample")):

        if meltpoint is None:
            set_meltpoint(state)

        if low_temp is None:
            set_low_temp(state)

        if time_sample is None:
            set_time_sample(state)

        if isinstance(user_options.config_file_path, str):
            state.dump_config_to_toml(user_options.config_file_path)

    state.select_colormap()


def set_directories(user_options: UserOptions) -> None:
    print(f"For setting all of these data directories, Please enter either absolute paths, or paths relative to your present working directory: {os.getcwd()}")
    user_input: str

    gen_hdf_dir: bool = False
    """if hasattr(user_options, "hdf_source_directory"):
        gen_hdf_dir = _confirm(f".hdf5 source directory is currently set to {user_options.hdf_source_directory}.", "Would you like to overwrite it?")
    else:
        gen_hdf_dir = True"""

    """if gen_hdf_dir:
        while True:
            user_input = input("Please enter the directory of your .hdf5 files and associated data: ")
            if os.path.exists(user_input):
                if _confirm(f"You entered {user_input}", "Is this correct", "yes"):
                    # os.path.isabs can be finnicky cross-platform, but, for this purpose, shoudld be fully correct
                    if os.path.isabs(user_input):
                        user_options.hdf_source_directory = user_input
                    else:
                        user_options.hdf_source_directory = os.path.join(os.getcwd(), user_input)
                    break
            else:
                print(f"Error: That directory does not exist. Please enter the absolute path to a directory or the path relative to your present working directory: {os.getcwd()}")"""

    gen_odb_dir: bool = False
    """if hasattr(user_options, "odb_source_directory"):
        gen_odb_dir = _confirm(f".odb source directory is currently set to {user_options.odb_source_directory}.", "Would you like to overwrite it?")
    else:
        gen_odb_dir = True"""

    if gen_odb_dir:
        while True:
            user_input = input("Please enter the directory of your .odb files: ")
            """if os.path.exists(user_input):
                if _confirm(f"You entered {user_input}", "Is this correct", "yes"):
                    # os.path.isabs can be finnicky cross-platform, but, for this purpose, shoudld be fully correct
                    if os.path.isabs(user_input):
                        user_options.odb_source_directory = user_input
                    else:
                        user_options.odb_source_directory = os.path.join(os.getcwd(), user_input)
                    break
            else:
                print(f"Error: That directory does not exist. Please enter the absolute path to a directory or the path relative to your present working directory: {os.getcwd()}")"""

    gen_results_dir: bool = False
    """if hasattr(user_options, "results_directory"):
        gen_results_dir = _nconfirm(f"The results directory is currently set to {user_options.results_directory}.", "Would you like to overwrite it?")
    else:
        gen_results_dir = True"""
    
    if gen_results_dir:
        while True:
            user_input = input("Please enter the directory where you would like your results to be written: ")
            """if os.path.exists(user_input):
                if _confirm(f"You entered {user_input}", "Is this correct", "yes"):
                    # os.path.isabs can be finnicky cross-platform, but, for this purpose, shoudld be fully correct
                    if os.path.isabs(user_input):
                        user_options.results_directory = user_input
                    else:
                        user_options.results_directory = os.path.join(os.getcwd(), user_input)
                    break
            else:
                print(f"Error: That directory does not exist. Please enter the absolute path to a directory or the path relative to your present working directory: {os.getcwd()}")"""


# TODO Fix
def set_views(state: Odb) -> None:
    views_dict: Dict[str, Dict[str, int]] = load_views_dict()
    while True:
        print("Please Select a Preset View for your plots")
        print('To view all default presets, please enter "list"')
        print('Or, to specify your own view angle, please enter "custom"')
        user_input: str = input("> ").strip().lower()
        if user_input == "list":
            print_views(views_dict)
        elif user_input == "custom":
            elev: int
            azim: int
            roll: int
            elev, azim, roll = get_custom_view()

            state.elev = elev
            state.azim = azim
            state.roll = roll
            return

        else:
            try:
                if user_input.isnumeric():
                    chosen_ind: int = int(user_input) - 1
                    state.elev = views_dict[list(views_dict.keys())[chosen_ind]]["elev"]
                    state.azim = views_dict[list(views_dict.keys())[chosen_ind]]["azim"]
                    state.roll = views_dict[list(views_dict.keys())[chosen_ind]]["roll"]
                    return

                else:
                    state.elev = views_dict[user_input]["elev"]
                    state.azim = views_dict[user_input]["azim"]
                    state.roll = views_dict[user_input]["roll"]
                    return

            except:
                print('Error: input must be "list," "custom," or the index or name of a named view as seen from the "list" command.')


def get_custom_view() -> "Tuple[int, int, int]":
    elev: int
    azim: int
    roll: int
    while True:
        while True:
            try:
                elev = int(input("Elevation Angle in Degrees: "))
                break
            except ValueError:
                print("Error: Elevation Angle must be an integer")
        while True:
            try:
                azim = int(input("Azimuth Angle in Degrees: "))
                break
            except ValueError:
                print("Error: Azimuth Angle must be an integer")
        while True:
            try:
                roll = int(input("Roll Angle in Degrees: "))
                break
            except ValueError:
                print("Error: Roll Angle must be an integer")

        """if _confirm(f"X Rotation: {elev}\nY Rotation: {azim}\nZ Rotation: {roll}", "Is this correct?", "yes"):
            break"""

    return (elev, azim, roll)


def print_views(views: "Dict[str, Dict[str, int]]") -> None:
    print("Index | Name | Rotation Values")
    v: Tuple[str, Dict[str, int]]
    view: str
    vals: Dict[str, int]
    key: str
    val: int
    for i, v in enumerate(views.items()):
        view, vals = v
        print(f"{i + 1}: `{view}")
        for key, val in vals.items():
            print(f"\t{key}: {val}")
        print()
    print()


def set_abaqus(state: Odb) -> None:
    while True:
        user_input = input("Please enter the exectuable program to process .odb files: ")
        """if _confirm(f"You entered {user_input}", "Is this correct?", "yes"):
            state.set_abaqus(user_input)
            break"""


def set_nodeset(state: Odb) -> None:
    while True:
        user_input = input("Please enter the name of the target nodeset: ")
        """if _confirm(f"You entered {user_input}", "Is this correct?", "yes"):
            state.set_nodesets(user_input)
            break"""