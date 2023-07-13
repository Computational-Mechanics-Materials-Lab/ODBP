#!/usr/bin/env python3

"""
Built-in CLI for ODB Plotter, allowing for interactive system access without writing scripts
"""

import os
import sys
import cmd
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
import numpy as np
import pandas as pd
from typing import Any, Union, TextIO, List, Tuple, Dict, Optional
from .odb import Odb
from .state import CLIOptions, UserOptions, process_input, print_state, load_views_dict
from odbp import __version__


class OdbPlotterCLI(cmd.Cmd):
    
    def __init__(self) -> None:
        super().__init__()
        self.prompt: str = "> "
        self.intro: str = f"ODBPlotter {__version__}"
        self.state: Odb
        self.options: UserOptions
        #self.state, self.options = process_input()
        self.odb = Odb()
        self.options = UserOptions()


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
                    print("Caught a Control-C. Returning to main command line\n")
                    print('Please use the "quit", "q", or "exit" commands (or Control-D) to exit ODBPlotter\n\n')

            self.postloop()
        finally:
            if self.use_rawinput and self.completekey:
                try:
                    import readline
                    readline.set_completer(self.old_completer)
                except ImportError:
                    pass



    def confirm(self, message: str, confirmation: str, default: "Optional[str]" = None) -> bool:
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
    def quit(self) -> None:
        self.stdout.write("\nExiting")
        sys.exit(0)


    def do_quit(self, arg: str) -> None:
        """Exit gracefully (same as exit or q)"""
        _ = arg
        self.quit()

    
    def do_exit(self, arg: str) -> None:
        """Exit gracefully (same as quit or q)"""
        _ = arg
        self.quit()


    def do_q(self, arg: str) -> None:
        """Exit gracefull (same as quit or exit)"""
        _ = arg
        self.quit()


    # Select and Dispatches
    def select(self) -> None:
        
        #def select_files(state: Odb, user_options: UserOptions) -> None:
        odb_options: Tuple[str, str] = ("odb", ".odb")
        hdf_options: Tuple[str, str, str, str, str ,str] = (".hdf", "hdf", ".hdf5", "hdf5", "hdfs", ".hdfs")
        user_input: str

        # select odb
        while True:
            user_input = input('Please enter either "hdf" if you plan to open .hdf5 file or "odb" if you plan to open a .odb file: ').strip().lower()

            if user_input in odb_options or user_input in hdf_options:
                if(self.confirm(f"You entered {user_input}", "Is this correct", "yes")):
                    break

            else:
                print("Error: invalid input")

        if user_input in odb_options:
            # process odb
            odb_path_valid: bool = False
            while not odb_path_valid:
                user_input = input("Please enter the path of the odb file: ")
                if(self.confirm(f"You entered {user_input}", "Is this correct", "yes")):
                    output: Optional[bool] = self.odb.select_odb(self.options, user_input)
                if isinstance(output, bool):
                    print(f"Error: the file {user_input} could not be found")

                else:
                    odb_path_valid = True

            gen_time_sample: bool = False
            # TODO fix with slots
            if hasattr(self.odb, "time_sample"):
                gen_time_sample = self.confirm(f"Time Sample is already set as {self.odb.time_sample}.", "Would you like to overwrite it?")

            else:
                gen_time_sample = True

            if gen_time_sample:
                # TODO
                set_time_sample(self.odb)

            if self.confirm('You may now convert this .odb file to a .hdf5 file or you may do this later with the "convert" command.', "Would you like to convert now?", "yes"):
                self.convert()

        elif user_input in hdf_options:
            # process hdf
            hdf_path_valid: bool = False
            while not hdf_path_valid:
                user_input = input("Please enter the path of the hdf5 file, or the name of the hdf5 file in the hdfs directory: ")
                if(self.confirm(f"You entered {user_input}", "Is this correct", "yes")):
                    output: Union[UserOptions, bool] = self.odb.select_hdf(user_options, user_input)

                # TODO try/except
                if isinstance(output, UserOptions):
                    user_options = output
                    hdf_path_valid = True

                else:
                    print(f"Error: the file {user_input} could not be found")

            pre_process_data(self.odb, user_options)
            print(f"Target .hdf5 file: {self.odb.hdf_file_path}")




    def do_select(self, arg: str) -> None:
        """Select the .odb or .hdf5 files to convert or read from"""
        _ = arg
        self.select()


    def convert(self) -> None:
        if not hasattr(self, "odb_path"):
            while True:
                target_odb: str = input("No .odb file is selected. Please enter the target .odb file: ")
                if self.confirm(f"You entered {target_odb}", "Is this correct", "yes"):
                    self.odb.odb_path = target_odb
                    break

        else:
            print(f"{self.odb.odb_path} is set as current .odb file")

        if not hasattr(self, "hdf_path"):
            while True:
                target_hdf: str = input("No .hdf5 file is selected. Please enter the new name for the target .hdf5 file: ")
                if self.confirm(f"You entered {target_hdf}", "Is this correct", "yes"):
                    if not target_hdf.endswith(".hdf5"):
                        target_hdf += ".hdf5"
                    self.odb.hdf_path = target_hdf                    
                    break

        else:
            print(f"{self.odb.hdf_path} is set as the target .hdf5 file")

        self.odb.convert()


    def do_convert(self, arg: str) -> None:
        """Convert a .odb file to a .hdf5 file"""
        _ = arg
        self.convert()


    # Extrema and dispatchers
    def do_extrema():


    def do_range():


    def do_ranges():


    def do_x(self, arg: str) -> None:
        """Set x-range (same as xs)"""
        _ = arg
        self.x()


    def do_xs(self, arg: str) -> None:
        """Set x-range (same as x)"""
        _ = arg
        self.x()


    def do_y(self, arg: str) -> None:
        """Set y-range (same as ys)"""
        _ = arg
        self.y()

    
    def do_ys(self, arg: str) -> None:
        """Set y-range (same as y)"""
        _ = arg
        self.y()


    def do_z(self, arg: str) -> None:
        """Set z-range (same as zs)"""
        _ = arg
        self.z()


    def do_zs(self, arg: str) -> None:
        """Set z-range (same as z)"""
        _ = arg
        self.z()


    def do_time():


    def do_times():


    def do_temp():


    def do_temps():


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

            if user_input in cli_options.extrema_options:
                set_extrema(state)

            elif user_input in cli_options.time_options:
                set_time(state)

            elif user_input in cli_options.time_sample_options:
                set_time_sample(state)

            elif user_input in cli_options.meltpoint_options:
                set_meltpoint(state)

            elif user_input in cli_options.low_temp_options:
                set_low_temp(state)

            elif user_input in cli_options.title_label_options:
                set_title_and_label(state, user_options)

            elif user_input in cli_options.directory_options:
                set_directories(user_options)

            elif user_input in cli_options.process_options:
                load_hdf(state)

            elif user_input in cli_options.angle_options:
                set_views(state)

            elif user_input in cli_options.show_all_options:
                state.interactive = not state.interactive
                print(f"Plots will now {'BE' if state.interactive else 'NOT BE'} shown")

            elif user_input in cli_options.plot_options:
                plot_time_range(state, user_options)

            elif user_input in cli_options.abaqus_options:
                set_abaqus(state)

            elif user_input in cli_options.nodeset_options:
                set_nodeset(state)

            elif user_input in cli_options.state_options:
                print_state(state, user_options)

            elif user_input in cli_options.help_options:
                cli_options.print_help()

            else:
                print('Invalid option. Use "help" to see available options')

        except KeyboardInterrupt:
            print("\nKeyboard Interrupt Received, returning to main menu")
            print('(From this menu, use the "exit" command to exit, or Control-D/EOF)')

        except EOFError:
            print("\nExiting")
            main_loop = False


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


def set_title_and_label(state: Odb, user_options: UserOptions) -> None:
    default_title: str = ""

    if hasattr(state, "hdf_file_path"):
        default_title = state.hdf_file_path.split(os.sep)[-1].split(".")[0]
    elif hasattr(state, "odb_file_path"):
        default_title = state.odb_file_path.split(os.sep)[-1].split(".")[0]

    while True:
        user_input: str
        if isinstance(default_title, str):
            user_input = input(f"Please Enter the Title for your Images (Leave blank for the Default value: {default_title}): ")
            if user_input == "":
                user_input = default_title

            """if confirm(f"You entered {user_input}", "Is this correct", "yes"):
                user_options.image_title = user_input
                break"""

        else:
            user_input = input("Please Enter the Title for you Images: ")
            if user_input == "":
                print("Error: You must enter a non-empty value")
            """else:
                if confirm(f"You entered {user_input}", "Is this correct", "yes"):
                    user_options.image_title = user_input
                    break"""

    while True:
        user_input: str
        default_label = user_options.image_title
        user_input = input(f"Please Enter the Label for your Images (Leave blank for the Default value: {default_label}): ")
        if user_input == "":
            user_input = default_label

        """if confirm(f"You entered {user_input}", "Is this correct", "yes"):
            user_options.image_label = user_input
            break"""


def set_directories(user_options: UserOptions) -> None:
    print(f"For setting all of these data directories, Please enter either absolute paths, or paths relative to your present working directory: {os.getcwd()}")
    user_input: str

    gen_hdf_dir: bool = False
    """if hasattr(user_options, "hdf_source_directory"):
        gen_hdf_dir = confirm(f".hdf5 source directory is currently set to {user_options.hdf_source_directory}.", "Would you like to overwrite it?")
    else:
        gen_hdf_dir = True"""

    """if gen_hdf_dir:
        while True:
            user_input = input("Please enter the directory of your .hdf5 files and associated data: ")
            if os.path.exists(user_input):
                if confirm(f"You entered {user_input}", "Is this correct", "yes"):
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
        gen_odb_dir = confirm(f".odb source directory is currently set to {user_options.odb_source_directory}.", "Would you like to overwrite it?")
    else:
        gen_odb_dir = True"""

    if gen_odb_dir:
        while True:
            user_input = input("Please enter the directory of your .odb files: ")
            """if os.path.exists(user_input):
                if confirm(f"You entered {user_input}", "Is this correct", "yes"):
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
        gen_results_dir = confirm(f"The results directory is currently set to {user_options.results_directory}.", "Would you like to overwrite it?")
    else:
        gen_results_dir = True"""
    
    if gen_results_dir:
        while True:
            user_input = input("Please enter the directory where you would like your results to be written: ")
            """if os.path.exists(user_input):
                if confirm(f"You entered {user_input}", "Is this correct", "yes"):
                    # os.path.isabs can be finnicky cross-platform, but, for this purpose, shoudld be fully correct
                    if os.path.isabs(user_input):
                        user_options.results_directory = user_input
                    else:
                        user_options.results_directory = os.path.join(os.getcwd(), user_input)
                    break
            else:
                print(f"Error: That directory does not exist. Please enter the absolute path to a directory or the path relative to your present working directory: {os.getcwd()}")"""


def set_extrema(state: Odb) -> None:
    x_low: float
    x_high: float
    y_low: float
    y_high: float
    z_low: float
    z_high: float
    while True:
        # Get the desired coordinates and time steps to plot
        extrema: Dict[Tuple[str, str], Tuple[float, float]] = {
                ("lower X", "upper X"): tuple(),
                ("lower Y", "upper Y"): tuple(),
                ("lower Z", "upper Z"): tuple(),
                }
        extremum: Tuple[str, str]
        for extremum in extrema.keys():
            extrema[extremum] = process_extrema(extremum)

        x_low, x_high = extrema[("lower X", "upper X")]
        y_low, y_high = extrema[("lower Y", "upper Y")]
        z_low, z_high = extrema[("lower Z", "upper Z")]
        print()
        """if confirm(f"SELECTED VALUES:\nX from {x_low} to {x_high}\nY from {y_low} to {y_high}\nZ from {z_low} to {z_high}", "Is this correct", "yes"):
            state.set_x_low(x_low)
            state.set_x_high(x_high)
            state.set_y_low(y_low)
            state.set_y_high(y_high)
            state.set_z_low(z_low)
            state.set_z_high(z_high)
            print(f"Spatial Dimensions Updated to:\nX from {state.x.low} to {state.x.high}\nY from {state.y.low} to {state.y.high}\nZ from {state.z.low} to {state.z.high}")
            break"""


def set_meltpoint(state: Odb) -> None:
    while True:
        try:
            meltpoint = float(input("Enter the meltpoint of the Mesh: "))

            """if confirm(f"Meltng Point: {meltpoint}", "Is this correct", "yes"):
                state.set_meltpoint(meltpoint)
                print(f"Melting Point set to: {state.meltpoint}")
                break"""

        except ValueError:
            print("Error, Melting Point must be a number")


def set_low_temp(state: Odb) -> None:
    while True:
        try:
            low_temp = float(input("Enter the lower temperature bound of the Mesh: "))

            """if confirm(f"Lower Temperature Bound: {low_temp}", "Is this correct", "yes"):
                state.set_low_temp(low_temp)
                print(f"Lower Temperature Bound set to: {state.low_temp}")
                break"""

        except ValueError:
            print("Error, Lower Temperature Bound must be a number")


def set_time_sample(state: Odb) -> None:
    while True:
        try:
            time_sample: int = int(input("Enter the Time Sample: "))

            """if confirm(f"Time Sample: {time_sample}", "Is this correct", "yes"):
                state.set_time_sample(time_sample)
                break"""

        except ValueError:
            print("Error, Time Sample must be an integer greater than or equal than 1")


def set_time(state: Odb) -> None:
    lower_time: Union[int, float] = 0
    upper_time: Union[int, float] = float("inf")
    while True:
        values: List[Tuple[str, Union[int, float], str]] = [("lower time", 0, "0"), ("upper time", float("inf"), "infinity")]
        i: int
        v: Tuple[str, Union[int, float], str]
        for i, v in enumerate(values): 
            key: str
            default: Union[int, float]
            default_text: str
            key, default, default_text = v
            while True:
                try:
                    user_input: str = input(f"Enter the {key} value you would like to plot (Leave blank for {default_text}): ")
                    result: Union[int, float]
                    if user_input == "":
                        result = default
                    else:
                        result = float(user_input)
                    
                    if i == 0:
                        lower_time = result
                    else:
                        upper_time = result
                    break

                except ValueError:
                    print("Error, all selected time values must be positive numbers")

        """if confirm(f"You entered {lower_time} as the starting time and {upper_time} as the ending time.", "Is this correct", "yes"):
            state.set_time_low(lower_time)
            state.set_time_high(upper_time)
            print(f"Time Range: from {state.time_low} to {state.time_high if state.time_high != float('inf') else 'infinity'}")
            break"""


def process_extrema(keys: "Tuple[str, str]") -> "Tuple[float, float]":
    results: List[float] = list()
    i: int
    key: str
    inf_addon: str
    inf_val: float
    for i, key in enumerate(keys):
        if i % 2 == 1:
            inf_addon = ""
            inf_val = np.inf
        else:
            inf_addon = "negative "
            inf_val = -1 * np.inf
        while True:
            try:
                user_input: str = input(f"Enter the {key} value you would like to plot (Leave blank for {inf_addon}infinity): ")
                if user_input == "":
                    results.append(inf_val)
                else:
                    results.append(float(user_input))
                break

            except ValueError:
                print("Error, all selected coordinates and time steps must be numbers")

    return tuple(results)


def load_hdf(state: Odb) -> None:
    state.process_hdf()


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

        """if confirm(f"X Rotation: {elev}\nY Rotation: {azim}\nZ Rotation: {roll}", "Is this correct?", "yes"):
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
        """if confirm(f"You entered {user_input}", "Is this correct?", "yes"):
            state.set_abaqus(user_input)
            break"""


def set_nodeset(state: Odb) -> None:
    while True:
        user_input = input("Please enter the name of the target nodeset: ")
        """if confirm(f"You entered {user_input}", "Is this correct?", "yes"):
            state.set_nodesets(user_input)
            break"""


def plot_time_range(state: Odb, user_options: UserOptions) -> None:

    if not state.loaded:
        print('Error, you must load the contents of a .hdf5 file into memory with the "run" or "process" commands in order to plot')
        return

    """if user_options.image_label == "" or user_options.image_title == "":
        if not confirm("Warning: Either the image label or image title is unset. Consider setting them with the \"title\" or \"label\" commands.", "Do you want to continue", "no"):
            return"""

    # out_nodes["Time"] has the time values for each node, we only need one
    # Divide length by len(bounded_nodes), go up to that
    times: Any = state.out_nodes["Time"]
    final_time_idx: int = int(len(times) / len(state.bounded_nodes))

    if not state.interactive:
        print("Please wait while the plotter prepares your images...")
    for time in times[:final_time_idx]:
        plot_time_slice(time, state, user_options)


def plot_time_slice(time: float, state: Odb, user_options: UserOptions) -> None:
    formatted_time: str = format(round(time, 2), ".2f")

    if state.interactive:
        print(f"Plotting time step {formatted_time}")

    save_str: str = os.path.join(user_options.results_directory, f"{user_options.image_title}-{formatted_time}.png")
    plot: Any = state.plot_time_3d(time, user_options.image_label, state.interactive)

    if state.interactive:
        try:
            plot.show()
            plot.screenshot(save_str)
        except RuntimeError:
            print('Error: The plotter could not save a screenshot. Please close the viewing window by hitting the "q" key instead.')

    else:
        plot.screenshot(save_str)
        # with plot.window_size_context((1920, 1080)):
        #     plot.screenshot(save_str)

    del plot
