#!/usr/bin/env python3

"""
Built-in CLI for ODB Plotter, allowing for interactive system access without writing scripts
"""

import os
import toml
import numpy as np
from typing import Any, Union
from .odb_visualizer import OdbVisualizer
from .util import confirm
from .state import CLIOptions, UserOptions, process_input, print_state, load_views_dict
from odbp import __version__


def cli() -> None:

    main_loop: bool = True

    # TODO Process input toml file and/or cli switches here
    state: OdbVisualizer
    user_options: UserOptions
    state, user_options = process_input()
    cli_options: CLIOptions = CLIOptions()

    if user_options.run_immediate:
        # TODO
        load_hdf(state)
        plot_time_range(state, user_options)
        return

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


# TODO Rewrite
def select_files(state: OdbVisualizer, user_options: UserOptions) -> None:
    odb_options: tuple[str, str] = ("odb", ".odb")
    hdf_options: tuple[str, str, str, str, str ,str] = (".hdf", "hdf", ".hdf5", "hdf5", "hdfs", ".hdfs")
    user_input: str

    # select odb
    while True:
        user_input = input('Please enter either "hdf" if you plan to open .hdf5 file or "odb" if you plan to open a .odb file: ').strip().lower()

        if user_input in odb_options or user_input in hdf_options:
            if(confirm(f"You entered {user_input}", "Is this correct", "yes")):
                break

        else:
            print("Error: invalid input")

    if user_input in odb_options:
        # process odb
        while True:
            user_input = input("Please enter the path of the odb file: ")
            if(confirm(f"You entered {user_input}", "Is this correct", "yes")):
                if not os.path.exists(os.path.join(options.odb_source_directory, user_input)):
                    if not os.path.exists(os.path.join(os.getcwd(), user_input)):
                        if not os.path.exists(user_input):
                            print(f"Error: {user_input} file could not be found")
                        else:
                            state.odb_file_path = user_input
                    else:
                        state.odb_file_path = os.path.join(os.getcwd(), user_input)
                else:
                    state.odb_file_path = os.path.join(options.odb_source_directory, user_input)

            if hasattr(state, "odb_file_path") and state.odb_file_path is not None and state.odb_file_path != "":
                break

        gen_time_sample: bool = False
        if hasattr(state, "time_sample"):
            gen_time_sample = confirm(f"Time Sample is already set as {state.time_sample}.", "Would you like to overwrite it?")

        else:
            gen_time_sample = True

        if gen_time_sample:
            while True:
                user_input = input("The Program will extract every Nth frame where N = ")
                if(confirm(f"You entered {user_input}", "Is this correct", "yes")):
                    try:
                        state.time_sample = int(user_input)
                        break
                    except ValueError:
                        print("Error: time sample must be an integer")

        print("Converting this .odb file to a .hdf file")
        default: str = os.path.join(options.hdf_source_directory, state.odb_file_path.split(os.sep)[-1].split(".")[0] + ".hdf5")
        hdf_file_path: str
        while True:
            user_input = input(f"Please enter the name for this generated hdf file (leave blank for the default {default}): ")
            if user_input == "":
                user_input = default
            
            if(confirm(f"You entered {user_input}", "Is this correct", "yes")):
                hdf_file_path = os.path.join(options.hdf_source_directory, user_input)
                break

        state.odb_to_hdf(hdf_file_path)

    elif user_input in hdf_options:
        # process hdf
        while True:
            user_input = input("Please enter the path of the hdf5 file, or the name of the hdf5 file in the hdfs directory: ")
            if(confirm(f"You entered {user_input}", "Is this correct", "yes")):
                # If the config file exists, it will be under this name
                toml_config_file: str = user_input.split(".")[0] + ".toml"
                if not os.path.exists(os.path.join(options.hdf_source_directory, user_input)):
                    if not os.path.exists(os.path.join(os.getcwd(), user_input)):
                        if not os.path.exists(user_input):
                            print(f"Error: {user_input} file could not be found")
                        else:
                            state.hdf_file_path = user_input
                            options.toml_config_file = toml_config_file
                    else:
                        state.hdf_file_path = os.path.join(os.getcwd(), user_input)
                        options.toml_config_file = os.path.join(os.getcwd(), toml_config_file)
                else:
                    state.hdf_file_path = os.path.join(options.hdf_source_directory, user_input)
                    options.toml_config_file = os.path.join(options.hdf_source_directory, toml_config_file)

            if hasattr(state, "hdf_file_path") and state.hdf_file_path is not None and state.hdf_file_path != "":
                break

    pre_process_data(state, options)
    print(f"Target .hdf5 file: {state.hdf_file_path}")


# TODO on toml_config_file
def pre_process_data(state: OdbVisualizer, options: UserOptions):
    mesh_seed_size: Union[float, None] = None
    meltpoint: Union[float, None] = None
    low_temp: Union[float, None] = None
    time_sample: Union[int, None] = None

    if options.toml_config_file is not None and os.path.exists(options.toml_config_file):
        with open(options.toml_config_file, "r") as j:
            toml_config: dict[str, Any] = toml.load(j)

        if "hdf_file_path" in toml_config:
            if toml_config["hdf_file_path"] != state.hdf_file_path:
                print("INFO: File name provided and File Name in the config do not match. This could be an issue, or it might be fine")

        if "mesh_seed_size" in toml_config:
            mesh_seed_size = toml_config["mesh_seed_size"]

        if "meltpoint" in toml_config:
            meltpoint = toml_config["meltpoint"]

        if "low_temp" in toml_config:
            low_temp = toml_config["low_temp"]

        if "time_sample" in toml_config:
            time_sample = toml_config["time_sample"]

        # Manage mesh_seed_size
        if mesh_seed_size is not None:
            print(f"Setting Mesh Seed Size to stored value of {mesh_seed_size}")
            state.mesh_seed_size = mesh_seed_size

        elif hasattr(state, "mesh_seed_size"):
            print(f"Setting Default Seed Size of the Mesh to given value of {state.mesh_seed_size}")

        else: # Neither the stored value or the given value exist
            print("No Mesh Seed Size found. You must set it:")
            set_seed_size(state)

        # Manage meltpoint
        if meltpoint is not None:
            print(f"Setting Melting Point to stored value of {meltpoint}")
            state.meltpoint = meltpoint

        elif hasattr(state, "meltpoint"):
            print(f"Setting Default Melting Point to given value of {state.meltpoint}")
        
        else: # Neither the stored value nor the given value exist
            print("No Melting Point found. You must set it:")
            set_meltpoint(state)

        # Manage lower temperature bound
        if low_temp is not None:
            print(f"Setting Melting Point to stored value of {low_temp}")
            state.low_temp = low_temp

        elif hasattr(state, "low_temp"):
            print(f"Setting Default Melting Point to given value of {state.low_temp}")
        
        else: # Neither the stored value nor the given value exist
            print("No Lower Temperature Bound found. You must set it:")
            set_low_temp(state)

        # Manage time sample
        if time_sample is not None:
            print(f"Setting Time Sample to stored value of {time_sample}")
            state.time_sample = time_sample

        elif hasattr(state, "time_sample"):
            print(f"Setting Default Time Sample to given value of {state.time_sample}")

        else: # Neither the stored value nor the given value exist
            print("No Time Sample found. You must set it:")
            set_time_sample(state)

    if not all((hasattr(state, "mesh_seed_size"), hasattr(state, "meltpoint"), hasattr(state, "low_temp"), hasattr(state, "time_sample"))):

        # here, we need to set at least one of the things
        if mesh_seed_size is None:
            set_seed_size(state)

        if meltpoint is None:
            set_meltpoint(state)

        if low_temp is None:
            set_low_temp(state)

        if time_sample is None:
            set_time_sample(state)

        if isinstance(options.toml_config_file, str):
            state.dump_config_to_toml(options.toml_config_file)

    state.select_colormap()


def set_title_and_label(state: OdbVisualizer, user_options: UserOptions):
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

            if confirm(f"You entered {user_input}", "Is this correct", "yes"):
                user_options.image_title = user_input
                break

        else:
            user_input = input("Please Enter the Title for you Images: ")
            if user_input == "":
                print("Error: You must enter a non-empty value")
            else:
                if confirm(f"You entered {user_input}", "Is this correct", "yes"):
                    user_options.image_title = user_input
                    break

    while True:
        user_input: str
        default_label = user_options.image_title
        user_input = input(f"Please Enter the Label for your Images (Leave blank for the Default value: {default_label}): ")
        if user_input == "":
            user_input = default_label

        if confirm(f"You entered {user_input}", "Is this correct", "yes"):
            user_options.image_label = user_input
            break


def set_directories(user_options: UserOptions):
    print(f"For setting all of these data directories, Please enter either absolute paths, or paths relative to your present working directory: {os.getcwd()}")
    user_input: str

    gen_hdf_dir: bool = False
    if hasattr(user_options, "hdf_source_directory"):
        gen_hdf_dir = confirm(f".hdf5 source directory is currently set to {user_options.hdf_source_directory}.", "Would you like to overwrite it?")
    else:
        gen_hdf_dir = True

    if gen_hdf_dir:
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
                print(f"Error: That directory does not exist. Please enter the absolute path to a directory or the path relative to your present working directory: {os.getcwd()}")

    gen_odb_dir: bool = False
    if hasattr(user_options, "odb_source_directory"):
        gen_odb_dir = confirm(f".odb source directory is currently set to {user_options.odb_source_directory}.", "Would you like to overwrite it?")
    else:
        gen_odb_dir = True

    if gen_odb_dir:
        while True:
            user_input = input("Please enter the directory of your .odb files: ")
            if os.path.exists(user_input):
                if confirm(f"You entered {user_input}", "Is this correct", "yes"):
                    # os.path.isabs can be finnicky cross-platform, but, for this purpose, shoudld be fully correct
                    if os.path.isabs(user_input):
                        user_options.odb_source_directory = user_input
                    else:
                        user_options.odb_source_directory = os.path.join(os.getcwd(), user_input)
                    break
            else:
                print(f"Error: That directory does not exist. Please enter the absolute path to a directory or the path relative to your present working directory: {os.getcwd()}")

    gen_results_dir: bool = False
    if hasattr(user_options, "results_directory"):
        gen_results_dir = confirm(f"The results directory is currently set to {user_options.results_directory}.", "Would you like to overwrite it?")
    else:
        gen_results_dir = True
    
    if gen_results_dir:
        while True:
            user_input = input("Please enter the directory where you would like your results to be written: ")
            if os.path.exists(user_input):
                if confirm(f"You entered {user_input}", "Is this correct", "yes"):
                    # os.path.isabs can be finnicky cross-platform, but, for this purpose, shoudld be fully correct
                    if os.path.isabs(user_input):
                        user_options.results_directory = user_input
                    else:
                        user_options.results_directory = os.path.join(os.getcwd(), user_input)
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
        if confirm(f"SELECTED VALUES:\nX from {x_low} to {x_high}\nY from {y_low} to {y_high}\nZ from {z_low} to {z_high}", "Is this correct", "yes"):
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

            if confirm(f"Mesh Seed Size: {seed}", "Is this correct", "yes"):
                state.set_mesh_seed_size(seed)
                print(f"Mesh Seed Size set to: {state.mesh_seed_size}")
                break

        except ValueError:
            print("Error, Default Seed Size must be a number")


def set_meltpoint(state: OdbVisualizer) -> None:
    while True:
        try:
            meltpoint = float(input("Enter the meltpoint of the Mesh: "))

            if confirm(f"Meltng Point: {meltpoint}", "Is this correct", "yes"):
                state.set_meltpoint(meltpoint)
                print(f"Melting Point set to: {state.meltpoint}")
                break

        except ValueError:
            print("Error, Melting Point must be a number")


def set_low_temp(state: OdbVisualizer) -> None:
    while True:
        try:
            low_temp = float(input("Enter the lower temperature bound of the Mesh: "))

            if confirm(f"Lower Temperature Bound: {low_temp}", "Is this correct", "yes"):
                state.set_low_temp(low_temp)
                print(f"Lower Temperature Bound set to: {state.low_temp}")
                break

        except ValueError:
            print("Error, Lower Temperature Bound must be a number")


def set_time_sample(state: OdbVisualizer) -> None:
    print("INFO: You must enter the Time Sample with which the .hdf5 was generated")
    while True:
        try:
            time_sample: int = int(input("Enter the Time Sample: "))

            if confirm(f"Time Sample: {time_sample}", "Is this correct", "yes"):
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

        if confirm(f"You entered {lower_time} as the starting time and {upper_time} as the ending time.", "Is this correct", "yes"):
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


def load_hdf(state: OdbVisualizer):
    state.process_hdf()


# TODO Fix
def set_views(state: OdbVisualizer):
    views_dict: dict[str, dict[str, int]] = load_views_dict()
    while True:
        print("Please Select a Preset View for your plots")
        print('To view all default presets, please enter "list"')
        print('Or, to specify your own view angle, please enter "custom"')
        user_input: str = input("> ").strip().lower()
        if user_input == "list":
            print_views(views_dict)
        elif user_input == "custom":
            x_rot: int
            y_rot: int
            z_rot: int
            x_rot, y_rot, z_rot = get_custom_view()

            state.x_rot = x_rot
            state.y_rot = y_rot
            state.z_rot = z_rot
            return

        else:
            try:
                state.x_rot = views_dict[user_input]["x_rot"]
                state.y_rot = views_dict[user_input]["y_rot"]
                state.z_rot = views_dict[user_input]["z_rot"]
                return

            except KeyError:
                print('Error: input must be "list," "custom," or a named view as seen from the "list" command.')


def get_custom_view() -> tuple[int, int, int]:
    x_rot: int
    y_rot: int
    z_rot: int
    while True:
        while True:
            try:
                x_rot = int(input("Rotation around the X-Axis in Degrees: "))
                break
            except ValueError:
                print("Error: Rotation around the X-Axis must be an integer")
        while True:
            try:
                y_rot = int(input("Rotation around the Y-Axis in Degrees: "))
                break
            except ValueError:
                print("Error: Rotation around the Y-Axis must be an integer")
        while True:
            try:
                z_rot = int(input("Rotation around the Z-Axis in Degrees: "))
                break
            except ValueError:
                print("Error: Rotation around the Z-Axis must be an integer")

        if confirm(f"X Rotation: {x_rot}\nY Rotation: {y_rot}\nZ Rotation: {z_rot}", "Is this correct?", "yes"):
            break

    return (x_rot, y_rot, z_rot)


def print_views(views: dict[str, dict[str, int]]) -> None:
    print("Name | Rotation Values")
    view: str
    vals: dict[str, int]
    key: str
    val: int
    for view, vals in views.items():
        print(view)
        for key, val in vals.items():
            print(f"\t{key}: {val}")
        print()
    print()


def plot_time_range(state: OdbVisualizer, user_options: UserOptions):

    if not state.loaded:
        print('Error, you must load the contents of a .hdf5 file into memory with the "run" or "process" commands in order to plot')
        return

    if user_options.image_label == "" or user_options.image_title == "":
        if not confirm(f"Warning: Either the image label or image title is unset. Consider setting them with the \"title\" or \"label\" commands.", "Do you want to continue", "no"):
            return

    # out_nodes["Time"] has the time values for each node, we only need one
    # Divide length by len(bounded_nodes), go up to that
    times: Any = state.out_nodes["Time"]
    final_time_idx: int = int(len(times) / len(state.bounded_nodes))

    if not state.interactive:
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

