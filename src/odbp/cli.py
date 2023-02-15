#!/usr/bin/env python3

"""
Built-in CLI for ODB Plotter, allowing for interactive system access without writing scripts
"""

import os
import sys
import json
import argparse
import PIL
import numpy as np
import matplotlib.pyplot as plt
#from mpt_toolkits.mplot3d import Axes3D
from typing import Any, Union, TypeAlias
from .odb_visualizer import OdbVisualizer
from .util import confirm
from odbp import __version__


ConfigFiletype: TypeAlias = Union[str, None]


class UserOptions():
    """
    Struct to store user's input
    """
    def __init__(self, hdf_source_directory: str, odb_source_directory: str, results_directory: str, image_title: str, image_label: str, json_config_file: ConfigFiletype, run_immediate: bool = False) -> None:
        """
        Default values for user Options:
        hdf_source_directory: user's present working directory
        odb_source_directory: user's present working direcotry
        results_directory: user's present working directory
        image_title: name of the .hdf5 file + ".png"
        image_label: image_title
        run_immediate: False
        """
        self.hdf_source_directory: str = hdf_source_directory
        self.odb_source_directory: str = odb_source_directory
        self.results_directory: str = results_directory
        self.image_title: str = image_title
        self.image_label: str = image_label
        self.json_config_file: ConfigFiletype = json_config_file
        self.run_immediate: bool = run_immediate


class CLIOptions():
    """
    Struct to store cli optoins without repeating
    """
    def __init__(self) -> None:
        self.quit_options: list[str] = ["exit", "quit", "q"]
        self.quit_help: str = "Exit ODBPlotter"
        self.quit_options_formatted: str = ", ".join(self.quit_options)

        self.select_options: list[str] = ["select"]
        self.select_help: str = "Select an hdf5 file (or generate an hdf5 file from a .odb file)"
        self.select_options_formatted: str = ", ".join(self.select_options)

        self.seed_options: list[str] = ["seed", "mesh"]
        self.seed_help: str = "Set the Mesh Seed Size"
        self.seed_options_formatted: str = ", ".join(self.seed_options)

        self.extrema_options: list[str] = ["extrema", "range"]
        self.extrema_help: str = "Set the upper and lower x, y, and z bounds for plotting"
        self.extrema_options_formatted: str = ", ".join(self.extrema_options)

        self.time_options: list[str] = ["time"]
        self.time_help: str = "Set the upper and lower time bounds"
        self.time_options_formatted: str = ", ".join(self.time_options)

        self.time_sample_options: list[str] = ["sample"]
        self.time_sample_help: str = "Set the Time Sample for the hdf5 file"
        self.time_sample_options_formatted: str = ", ".join(self.time_sample_options)

        self.meltpoint_options: list[str] = ["meltpoint", "melt", "point"]
        self.meltpoint_help: str = "Set the Melting Point for the hdf5 file"
        self.meltpoint_options_formatted: str = ", ".join(self.meltpoint_options)

        self.title_label_options: list[str] = ["title", "label"]
        self.title_label_help: str = "Set the title and label of the output plots"
        self.title_label_options_formatted: str = ", ".join(self.title_label_options)

        self.directory_options: list[str] = ["dir", "dir", "directory", "directories"]
        self.directory_help: str = "Set the source and output directories"
        self.directory_options_formatted: str = ", ".join(self.directory_options)

        self.process_options: list[str] = ["process", "run", "load"]
        self.process_help: str = "Actually load the selected data from the file set in select"
        self.process_options_formatted: str = ", ".join(self.process_options)

        self.angle_options: list[str] = ["angle", "elev", "elevation", "azim", "azimuth", "roll"]
        self.angle_help: str = "Update the viewing angle"
        self.angle_options_formatted: str = ", ".join(self.angle_options)

        self.show_all_options: list[str] = ["show-all", "plot-all"]
        self.show_all_help: str = "Toggle if each time step will be shown in te matplotlib interactive viewer"
        self.show_all_options_formatted: str = ", ".join(self.show_all_options)

        self.plot_options: list[str] = ["plot", "show"]
        self.plot_help: str = "Plot each selected timestep"
        self.plot_options_formatted: str = ", ".join(self.plot_options)
        
        self.state_options: list[str] = ["state", "status", "settings"]
        self.state_help: str = "Show the current state of the settings of the plotter"
        self.state_options_formatted: str = ", ".join(self.state_options)

        self.help_options: list[str] = ["help", "use", "usage"]
        self.help_help: str = "Show this menu"
        self.help_options_formatted: str = ", ".join(self.help_options)

        self.longest_len: int = max(
                len(self.quit_options_formatted),
                len(self.select_options_formatted),
                len(self.seed_options_formatted),
                len(self.extrema_options_formatted),
                len(self.time_options_formatted),
                len(self.time_sample_options_formatted),
                len(self.meltpoint_options_formatted),
                len(self.title_label_options_formatted),
                len(self.directory_options_formatted),
                len(self.process_options_formatted),
                len(self.angle_options_formatted)
                )

    def print_help(self) -> None:
        print(
    f"""ODBPlotter Help:
{self.help_options_formatted.ljust(self.longest_len)} -- {self.help_help}
{self.quit_options_formatted.ljust(self.longest_len) } -- {self.quit_help}
{self.select_options_formatted.ljust(self.longest_len)} -- {self.select_help}
{self.seed_options_formatted.ljust(self.longest_len)} -- {self.seed_help}
{self.extrema_options_formatted.ljust(self.longest_len)} -- {self.extrema_help}
{self.time_options_formatted.ljust(self.longest_len)} -- {self.time_help}
{self.time_sample_options_formatted.ljust(self.longest_len)} -- {self.time_sample_help}
{self.meltpoint_options_formatted.ljust(self.longest_len)} -- {self.meltpoint_help}
{self.title_label_options_formatted.ljust(self.longest_len)} -- {self.title_label_help}
{self.directory_options_formatted.ljust(self.longest_len)} -- {self.directory_help}
{self.process_options_formatted.ljust(self.longest_len)} -- {self.process_help}
{self.angle_options_formatted.ljust(self.longest_len)} -- {self.angle_help}
{self.show_all_options_formatted.ljust(self.longest_len)} -- {self.show_all_help}
{self.plot_options_formatted.ljust(self.longest_len)} -- {self.plot_help}
{self.state_options_formatted.ljust(self.longest_len)} -- {self.state_help}"""
    )




def cli():

    state: OdbVisualizer = OdbVisualizer()
    main_loop: bool = True

    # TODO Process input json file and/or cli switches here
    user_options: UserOptions = process_input(state)
    cli_options: CLIOptions = CLIOptions()

    if user_options.run_immediate:
        # TODO
        load_hdf(state)
        plot_time_range(state, user_options)

    print(f"ODBPlotter {__version__}")
    while main_loop:
        try:
            user_input:str = input("\n> ").strip().lower()

            if user_input in cli_options.quit_options:
                print("\nExiting")
                main_loop = False

            elif user_input in cli_options.select_options:
                select_files(state, user_options)

            elif user_input in cli_options.seed_options:
                set_seed_size(state)

            elif user_input in cli_options.extrema_options:
                set_extrema(state)

            elif user_input in cli_options.time_options:
                set_time(state)

            elif user_input in cli_options.time_sample_options:
                set_time_sample(state)

            elif user_input in cli_options.meltpoint_options:
                set_meltpoint(state)

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


def select_files(state: OdbVisualizer, options: UserOptions) -> None:
    odb_options: tuple[str, str] = ("odb", ".odb")
    hdf_options: tuple[str, str, str, str, str ,str] = (".hdf", "hdf", ".hdf5", "hdf5", "hdfs", ".hdfs")
    user_input: str

    while True:
        user_input = input('Please enter either "hdf" if you plan to open .hdf5 file or "odb" if you plan to open a .odb file: ').strip().lower()

        if user_input in odb_options or user_input in hdf_options:
            if(confirm(f"You entered {user_input}", "yes")):
                break

        else:
            print("Error: invalid input")

    if user_input in odb_options:
        # process odb
        while True:
            user_input = input("Please enter the path of the odb file: ")
            if(confirm(f"You entered {user_input}", "yes")):
                odb = user_input
                break

        if hasattr(state, "time_sample"):
            if(confirm(f"Time Sample is already set as {state.time_sample}. Would you like to overwrite it?")):
                while True:
                        
                    user_input = input("The Program will extract every Nth frame where N = ")
                    if(confirm(f"You entered {user_input}", "yes")):
                        state.time_sample = int(user_input)
                        break

        state.odb_file = odb
        print("Converting this .odb file to a .hdf file")
        default: str = state.odb_file.split(".")[0] + ".hdf5"
        hdf_file_path: str
        while True:
            user_input = input(f"Please enter the name for this generated hdf file, or leave blank for the default: {default}")
            if user_input == "":
                user_input = default
            
            if(confirm(f"You entered {user_input}", "yes")):
                hdf_file_path = os.path.join(options.hdf_source_directory, user_input)
                break

        state.odb_to_hdf(hdf_file_path)


    elif user_input in hdf_options:
        # process hdf
        while True:
            user_input = input("Please enter the path of the hdf5 file, or the name of the hdf5 file in the hdfs directory: ")
            if(confirm(f"You entered {user_input}", "yes")):
                if not os.path.exists(os.path.join(options.hdf_source_directory, user_input)):
                    if not os.path.exists(os.path.join(os.getcwd(), user_input)):
                        if not os.path.exists(user_input):
                            print(f"Error: {user_input} file could not be found")
                        else:
                            state.hdf_file = user_input
                    else:
                        state.hdf_file = os.path.join(os.getcwd(), user_input)
                else:
                    state.hdf_file = os.path.join(options.hdf_source_directory, user_input)

            if hasattr(state, "hdf_file") and state.hdf_file is not None and state.hdf_file != "":
                break

    pre_process_data(state, options)
    print(f"Target .hdf5 file: {state.hdf_file}")


def pre_process_data(state: OdbVisualizer, options: UserOptions):
    mesh_seed_size: Union[float, None] = None
    meltpoint: Union[float, None] = None
    time_sample: Union[int, None] = None

    if options.json_config_file is not None and os.path.exists(options.json_config_file):
        with open(options.json_config_file, "r") as j:
            json_config: dict[str, Any] = json.load(j)

        if "hdf_file" in json_config:
            if json_config["hdf_file"] != state.hdf_file:
                print("INFO: File name provided and File Name in the config do not match. This could be an issue, or it might be fine")

        if "mesh_seed_size" in json_config:
            mesh_seed_size = json_config["mesh_seed_size"]

        if "meltpoint" in json_config:
            meltpoint = json_config["meltpoint"]

        if "time_sample" in json_config:
            time_sample = json_config["time_sample"]

        if hasattr(state, "mesh_seed_size") and mesh_seed_size is not None and state.mesh_seed_size != mesh_seed_size:
            print(f"INFO: Overwriting stored config file value for Mesh Seed Size of {mesh_seed_size} with value given by command line: {state.mesh_seed_size}")

        elif not hasattr(state, "mesh_seed_size") and mesh_seed_size is not None:
            print(f"Setting Default Seed Size of the Mesh to stored value of {state.mesh_seed_size}")
            state.mesh_seed_size = mesh_seed_size

        if hasattr(state, "meltpoint") and meltpoint is not None and state.meltpoint != meltpoint:
            print(f"INFO: Overwriting stored config file value for Melting Point of {meltpoint} with value given by command line: {state.meltpoint}")

        elif not hasattr(state, "meltpoint") and meltpoint is not None:
            print(f"Setting Default Melting Point to stored value of {state.meltpoint}")
            state.meltpoint = meltpoint

        if hasattr(state, "time_sample") and time_sample is not None and state.time_sample != time_sample:
            print(f"INFO: Overwriting stored config file value for Time Sample of {time_sample} with value given by command line: {state.time_sample}")

        elif not hasattr(state, "time_sample") and time_sample is not None:
            print(f"Setting Default Time Sample to stored value of {state.time_sample}")
            state.time_sample = time_sample

    if None in (mesh_seed_size, meltpoint, time_sample):

        # here, we need to set at least one of the things
        if mesh_seed_size is None:
            set_seed_size(state)

        if meltpoint is None:
            set_meltpoint(state)

        if time_sample is None:
            set_time_sample(state)

        if isinstance(options.json_config_file, str):
            state.dump_config_to_json(options.json_config_file)

    state.select_colormap()


def set_title_and_label(state: OdbVisualizer, user_options: UserOptions):
    default_title: str = ""

    if hasattr(state, "hdf_file"):
        default_title = state.hdf_file.split(os.sep)[-1].split(".")[0]
    elif hasattr(state, "odb_file"):
        default_title = state.odb_file.split(os.sep)[-1].split(".")[0]

    while True:
        user_input: str
        if isinstance(default_title, str):
            user_input = input(f"Please Enter the Title for your Images (Leave blank for the Default value: {default_title}): ")
            if user_input == "":
                user_input = default_title

            if confirm(f"You entered {user_input}", "yes"):
                user_options.image_title = user_input
                break

        else:
            user_input = input("Please Enter the Title for you Images: ")
            if user_input == "":
                print("Error: You must enter a non-empty value")
            else:
                if confirm(f"You entered {user_input}", "yes"):
                    user_options.image_title = user_input
                    break

    while True:
        user_input: str
        default_label = user_options.image_title
        user_input = input(f"Please Enter the Label for your Images (Leave blank for the Default value: {default_label}): ")
        if user_input == "":
            user_input = default_label

        if confirm(f"You entered {user_input}", "yes"):
            user_options.image_label = user_input
            break


def set_directories(user_options: UserOptions):
    print(f"For setting all of these data directories, Please enter either absolute paths, or paths relative to your present working directory: {os.getcwd()}")
    user_input: str

    while True:
        user_input = input("Please enter the directory of your .hdf5 files and associated data: ")
        if os.path.exists(user_input):
            if confirm(f"You entered {user_input}", "yes"):
                user_options.hdf_source_directory = user_input
                break
        else:
            print(f"Error: That directory does not exist. Please enter the absolute path to a directory or the path relative to your present working directory: {os.getcwd()}")

    while True:
        user_input = input("Please enter the directory of your .odb files: ")
        if os.path.exists(user_input):
            if confirm(f"You entered {user_input}", "yes"):
                user_options.odb_source_directory = user_input
                break
        else:
            print(f"Error: That directory does not exist. Please enter the absolute path to a directory or the path relative to your present working directory: {os.getcwd()}")

    while True:
        user_input = input("Please enter the directory where you would like your results to be written: ")
        if os.path.exists(user_input):
            if confirm(f"You entered {user_input}", "yes"):
                user_options.results_directory = user_input
                break
        else:
            print(f"Error: That directory does not exist. Please enter the absolute path to a directory or the path relative to your present working directory: {os.getcwd()}")


def set_extrema(state: OdbVisualizer):
    x_low: float
    x_high: float
    y_low: float
    y_high: float
    z_low: float
    z_high: float
    while True:
        # Get the desired coordinates and time steps to plot
        extrema: dict[tuple[str, str], tuple[float, float]] = {
                ("lower X", "upper X"): tuple(),
                ("lower Y", "upper Y"): tuple(),
                ("lower Z", "upper Z"): tuple(),
                }
        extremum: tuple[str, str]
        for extremum in extrema.keys():
            extrema[extremum] = process_extrema(extremum)

        x_low, x_high = extrema[("lower X", "upper X")]
        y_low, y_high = extrema[("lower Y", "upper Y")]
        z_low, z_high = extrema[("lower Z", "upper Z")]
        print()
        if confirm(f"SELECTED VALUES:\nX from {x_low} to {x_high}\nY from {y_low} to {y_high}\nZ from {z_low} to {z_high}", "yes"):
            state.set_x_low(x_low)
            state.set_x_high(x_high)
            state.set_y_low(y_low)
            state.set_y_high(y_high)
            state.set_z_low(z_low)
            state.set_z_high(z_high)
            print(f"Spatial Dimensions Updated to:\nX from {state.x.low} to {state.x.high}\nY from {state.y.low} to {state.y.high}\nZ from {state.z.low} to {state.z.high}")
            break


def set_seed_size(state: OdbVisualizer) -> None:
    print("INFO: You must enter the Mesh Seed Size with which the .hdf5 was generated")
    while True:
        try:
            seed: float = float(input("Enter the Default Seed Size of the Mesh: "))

            if confirm(f"Mesh Seed Size: {seed}", "yes"):
                state.set_mesh_seed_size(seed)
                print(f"Mesh Seed Size set to: {state.mesh_seed_size}")
                break

        except ValueError:
            print("Error, Default Seed Size must be a number")


def set_meltpoint(state: OdbVisualizer) -> None:
    while True:
        try:
            meltpoint = float(input("Enter the meltpoint of the Mesh: "))

            if confirm(f"Meltng Point: {meltpoint}", "yes"):
                state.set_meltpoint(meltpoint)
                print(f"Melting Point set to: {state.meltpoint}")
                break

        except ValueError:
            print("Error, melpoint must be a number")


def set_time_sample(state: OdbVisualizer) -> None:
    print("INFO: You must enter the Time Sample with which the .hdf5 was generated")
    while True:
        try:
            time_sample: int = int(input("Enter the Time Sample: "))

            if confirm(f"Time Sample: {time_sample}", "yes"):
                state.set_time_sample(time_sample)
                break

        except ValueError:
            print("Error, Time Sample must be an integer greater than or equal than 1")


def set_time(state: OdbVisualizer) -> None:
    lower_time: Union[int, float] = 0
    upper_time: Union[int, float] = float("inf")
    while True:
        values: list[tuple[str, Union[int, float], str]] = [("lower time", 0, "0"), ("upper time", float("inf"), "infinity")]
        i: int
        v: tuple[str, Union[int, float], str]
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

        if confirm(f"You entered {lower_time} as the starting time and {upper_time} as the ending time.", "yes"):
            state.time_low = lower_time
            state.time_high = upper_time
            print(f"Time Range: from {state.time_low} to {state.time_high if state.time_high != float('inf') else 'infinity'}")
            break


def process_extrema(keys: tuple[str, str]) -> tuple[float, float]:
    results: list[float] = list()
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


def process_input(state: OdbVisualizer) -> UserOptions:
    """
    The goal is to have a hierarchy of options. If a user passes in an option via a command-line switch, that option is set.
    If an option is not set by a switch, then the json input file is used.
    If an option is not set by the input file (if any), then prompt for it
    Possibly also include a default config file, like in $HOME/.config? Not sure
    """

    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="ODBPlotter")
    parser.add_argument("config_file", nargs="?")

    parser.add_argument("-s", "--hdf-source-directory")
    parser.add_argument("-b", "--odb-source-directory")
    parser.add_argument("-r", "--results-directory")

    parser.add_argument("-o", "--odb")
    parser.add_argument("-m", "--meltpoint")
    parser.add_argument("-S", "--seed-size")
    parser.add_argument("-t", "--time-sample", default=1)

    parser.add_argument("-H", "--hdf")

    parser.add_argument("-T", "--title")
    parser.add_argument("-l", "--label")

    parser.add_argument("--low-x")
    parser.add_argument("--high-x")
    parser.add_argument("--low-y")
    parser.add_argument("--high-y")
    parser.add_argument("--low-z")
    parser.add_argument("--high-z")

    parser.add_argument("--low-time", default=0)
    parser.add_argument("--high-time", default=float("inf"))

    # TODO Default
    #parser.add_argument("-v", "--view")
    
    parser.add_argument("-R", "--run", action="store_true")

    args: argparse.Namespace = parser.parse_args()

    # TODO
    """
    # Check if the default configuration works/is valid
    # TODO method to restore default config file for user
    if os.path.exists(CONFIG_FILE):
        # parse the config file, check that it's valid
        # if it's not, either error out, or just ignore it? (Probably error out)
    """
    if args.config_file:
        # This needs to handle everything in the config file, and should be overwritten by any cli switches, so it's fine for this to just go first
        # Also parse this file. If there is an error in the json, fail
        pass

    hdf_source_dir: str
    if args.hdf_source_directory:
        hdf_source_dir = os.path.abspath(args.hdf_source_directory)

        if not os.path.exists(hdf_source_dir):
            print(f"Error: The directory {args.hdf_source_directory} does not exist")
            sys.exit(1)

    else:
        hdf_source_dir = os.getcwd()

    odb_source_dir: str
    if args.odb_source_directory:
        odb_source_dir = os.path.abspath(args.odb_source_directory)

        if not os.path.exists(odb_source_dir):
            print(f"Error: The directory {args.odb_source_directory} does not exist")
            sys.exit(1)

    else:
        odb_source_dir = os.getcwd()

    results_dir: str = os.path.join(os.getcwd(), "results")
    if args.results_directory:
        results_dir = os.path.abspath(args.results_directory)

    if not os.path.exists(results_dir):
        print(f"Directory {results_dir} does not exist. Creating it now.")
        os.makedirs(results_dir)

    json_file: ConfigFiletype = None

    if args.odb:
        if args.hdf:
            # Don't allow both an odb and an hdf to be passed in
            print("Error: You cannot pass in both an .odb file and a .hdf5 file. Please enter one at a time")
            sys.exit(1)

        else:
            # Ensure the file exists
            if not os.path.exists(os.path.join(odb_source_dir, args.odb)):
                if not os.path.exists(os.path.join(os.getcwd(), args.odb)):
                    if not os.path.exists(args.odb):
                        print(f"Error: The file {args.odb} could not be found")
                        sys.exit(1)

                    else:
                        state.odb_file = args.odb
                else:
                    state.odb_file = os.path.join(os.getcwd(), args.odb)
            else:
                state.odb_file = os.path.join(odb_source_dir, args.odb)

        if args.meltpoint:
            state.meltpoint = args.meltpoint

        if args.seed_size:
            state.mesh_seed_size = args.seed_size

        if args.time_sample:
            state.time_sample = args.time_sample

        print("Converting this .odb file to a .hdf file")
        default: str = state.odb_file.split(".")[0] + ".hdf5"
        hdf_file_path: str
        while True:
            user_input = input(f"Please enter the name for this generated hdf file, or leave blank for the default: {default}")
            if user_input == "":
                user_input = default
            
            if(confirm(f"You entered {user_input}", "yes")):
                hdf_file_path = os.path.join(hdf_source_dir, user_input)
                break

        state.odb_to_hdf(hdf_file_path)
        json_file = os.path.join(hdf_source_dir, state.hdf_file.split(".")[0] + ".json")

    elif args.hdf:
        # ensure the file exists
        if not os.path.exists(os.path.join(hdf_source_dir, args.hdf)):
            if not os.path.exists(os.path.join(os.getcwd(), args.hdf)):
                if not os.path.exists(args.hdf):
                    print(f"Error: the file {args.hdf} could not be found")
                    sys.exit(1)

                else:
                    state.hdf_file = args.hdf
            else:
                state.hdf_file = os.path.join(os.getcwd(), args.hdf)
        else:
            state.hdf_file = os.path.join(hdf_source_dir, args.hdf)

        # Search for the stored json values for this hdf
        json_file = os.path.join(hdf_source_dir, state.hdf_file.split(".")[0] + ".json")
        if not os.path.exists(json_file):
            print(f".json config file for {state.hdf_file} could not be found")

        if args.meltpoint:
            state.meltpoint = args.meltpoint

        if args.seed_size:
            state.mesh_seed_size = args.seed_size

        if args.time_sample:
            state.time_sample = args.time_sample
 
    else:
        # Only if these other switches are used
        if args.meltpoint or args.seed_size or args.time_sample:
            print("INFO: neither a .odb file or a .hdf5 file were provided. You must provide one manually")
            if args.meltpoint:
                state.meltpoint = args.meltpoint

            if args.seed_size:
                state.mesh_seed_size = args.seed_size

            if args.time_sample:
                state.time_sample = args.time_sample

    # The 4 dimensions and the two booleans are always set here if they exist
    if args.low_x:
        state.set_x_low(args.low_x)
    if args.high_x:
        state.set_x_high(args.high_x)

    if args.low_y:
        state.set_y_low(args.low_y)
    if args.high_y:
        state.set_y_high(args.high_y)

    if args.low_z:
        state.set_z_low(args.low_z)
    if args.high_z:
        state.set_z_high(args.high_z)

    if args.low_time:
        state.set_time_low(args.low_time)
    if args.high_time:
        state.set_time_high(args.high_time)

    image_title: str
    image_label: str
    if args.title:
        image_title = args.title
    else:
        if hasattr(state, "hdf_file"):
            image_title = state.hdf_file.split(os.sep)[-1].split(".")[0]

        elif hasattr(state, "odb_file"):
            image_title = state.odb_file.split(os.sep)[-1].split(".")[0]

        else:
            image_title = ""

    if args.label:
        image_label = args.label
    else:
        image_label = image_title


    # Manage the final user state and return it
    user_options: UserOptions
    if args.run:
        user_options = UserOptions(hdf_source_dir, odb_source_dir, results_dir, image_title, image_label, json_file, args.run)

    else:
        user_options = UserOptions(hdf_source_dir, odb_source_dir, results_dir, image_title, image_label, json_file)

    return user_options


def load_hdf(state: OdbVisualizer):
    state.process_hdf()

def print_state(state: OdbVisualizer, user_options: UserOptions) -> None:
    print(
        f"""X Range:                 {state.x.low if hasattr(state.x, "low") else "not set"} to {state.x.high - state.mesh_seed_size if hasattr(state.x, "high") and hasattr(state, "mesh_seed_size") else "not set"}
Y Range:                 {state.y.low if hasattr(state.y, "low") else "not set"} to {state.y.high - state.mesh_seed_size if hasattr(state.y, "high") and hasattr(state, "mesh_seed_size") else "not set"}
Z Range:                 {state.z.low if hasattr(state.z, "low") else "not set"} to {state.z.high - state.mesh_seed_size if hasattr(state.z, "high") and hasattr(state, "mesh_seed_size") else "not set"}
Time Range:              {state.time_low if hasattr(state, "time_low") else "not set"} to {state.time_high if hasattr(state, "time_high") else "not set"}
Seed Size of the Mesh:   {state.mesh_seed_size if hasattr(state, "mesh_seed_size") else "not set"}
View Angle:              {state.angle if hasattr(state, "angle") else "not set"}
View Elevation:          {state.elev if hasattr(state, "elev") else "not set"}
View Azimuth:            {state.azim if hasattr(state, "azim") else "not set"}
View Roll:               {state.roll if hasattr(state, "roll") else "not set"}

Data loaded into memory: {'Yes' if state.loaded else 'No'}

Is each time-step being shown in the matplotlib interactive viewer? {'Yes' if state.interactive else 'No'}
Image Title:             {user_options.image_title}
Image Label:             {user_options.image_label}"""
    )


def set_views(state):
    while True:
        print("Please Select a Preset View for your plots: ")
        print('To view all default presets, please enter "list"')
        print ('Or, to specify your own view angle, please enter "custom"')
        print("Important Defaults: Top Face: 4, Right Face: 14, Front Face: 18, Top/Right/Front Isometric: 50")
        user_input = input("> ")
        if user_input.lower() == "list":
            print_views(state.views_list)
        elif user_input.lower() == "custom":
            state.select_views(get_custom_view())
            return

        else:
            try:
                user_input = int(user_input)
                if 0 > user_input > (len(state.views_list) + 1):
                    raise ValueError

                state.select_views(user_input)
                return

            except ValueError:
                print(f'Error: input must be "list," "custom," or an integer between 1 and {len(state.views_list) + 1}')


def get_custom_view():
    while True:
        while True:
            try:
                user_input = input("Elevation Value to view the plot (Leave Blank for the default): ")
                if user_input == "":
                    elev = "default"
                else:
                    elev = float(user_input)
                break
            except ValueError:
                print("Error, Elevation Value must be a number or left blank")
        while True:
            try:
                user_input = input("Azimuth Value to view the plot (Leave Blank for the default): ")
                if user_input == "":
                    azim = "default"
                else:
                    azim = float(user_input)
                break
            except ValueError:
                print("Error, Azimuth Value must be a number or left blank")
        while True:
            try:
                user_input = input("Roll Value to view the plot (Leave Blank for the default): ")
                if user_input == "":
                    roll = "default"
                else:
                    roll = float(user_input)
                break
            except ValueError:
                print("Error, Roll Value must be a number or left blank")

        if confirm(f"Elevation: {elev}\nAzimuth:   {azim}\nRoll:      {roll}", "yes"):
            break

    if elev == "default":
        elev = 30
    if azim == "default":
        azim = -60
    if roll == "default":
        roll = 0
    
    return elev, azim, roll


def print_views(views):
    print("View Index | View Angle: Face on Top")
    for idx, view in enumerate(views):
        print(f"{idx + 1}: {view}")
    print()


def plot_time_range(state: OdbVisualizer, user_options: UserOptions):

    if not state.loaded:
        print('Error, you must load the contents of a .hdf5 file into memory with the "run" or "process" commands in order to plot')
        return

    if user_options.image_label == "" or user_options.image_title == "":
        if not confirm(f"Warning: Either the image label or image title is unset. Consider setting them with the \"title\" or \"label\" commands. do you want to continue?", "no"):
            return

    # out_nodes["Time"] has the time values for each node, we only need one
    # Divide length by len(bounded_nodes), go up to that
    times: Any = state.out_nodes["Time"]
    final_time_idx: int = int(len(times) / len(state.bounded_nodes))

    if state.show_plots:
        print("Please wait while the plotter prepares your images...")
    for time in times[:final_time_idx]:
        plot_time_slice(time, state, user_options)


def plot_time_slice(time: float, state: OdbVisualizer, user_options: UserOptions) -> None:
    formatted_time: str = format(round(time, 2), ".2f")

    if state.interactive:
        print(f"Plotting time step {formatted_time}")

    save_str: str = os.path.join(user_options.results_directory, f"{user_options.image_title}-{formatted_time}.png")
    plot: Any = state.plot_time_3d(time, user_options.image_label, state.interactive)

    if state.interactive:
        plot.show()
        plot.screenshot(save_str)

    else:
        plot.screenshot(save_str)
        # with plot.window_size_context((1920, 1080)):
        #     plot.screenshot(save_str)

    del plot

