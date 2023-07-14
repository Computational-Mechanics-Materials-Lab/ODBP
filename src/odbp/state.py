""""""
import argparse
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
import sys
import os
import platformdirs
import subprocess
import pickle
import pandas as pd
from typing import Union, Any, TextIO, Tuple, List, Dict, Optional
from .odb import Odb
from odbp import __version__

        """
        angle
        abaqus/abq/abqpy
        cpus/cores
        nodesets
        frames
        nodes
        parts
        steps
        coord_key
        temp_key
        colormap
        save
        formate/save_format

        plot/show
        state/status/settings
        """

def print_state(state: Odb, user_options: UserOptions) -> None:
    lines: List[Dict[str, str]] = [
        {
            ".hdf5 file": f"{state.hdf_file_path if hasattr(state, 'hdf_file_path') else 'not set'}",
            ".odb file": f"{state.odb_file_path if hasattr(state, 'odb_file_path') else 'not set'}",
        },
        {
            "Source directory of .hdf5 files": f"{user_options.hdf_source_directory}",
            "Source directory of .odb files": f"{user_options.odb_source_directory}",
            "Directory to store results": f"{user_options.results_directory}",
        },
        {
            "X Range": f"{state.x.low if hasattr(state.x, 'low') else 'not set'} to {state.x.high if hasattr(state.x, 'high') else 'not set'}",
            "Y Range": f"{state.y.low if hasattr(state.y, 'low') else 'not set'} to {state.y.high if hasattr(state.y, 'high') else 'not set'}",
            "Z Range": f"{state.z.low if hasattr(state.z, 'low') else 'not set'} to {state.z.high if hasattr(state.z, 'high') else 'not set'}",
            "Time Range": f"{state.time_low if hasattr(state, 'time_low') else 'not set'} to {state.time_high if hasattr(state, 'time_high') else 'not set'}",
            "Temperature Range": f"{state.low_temp if hasattr(state, 'low_temp') else 'not set'} to {state.meltpoint if hasattr(state, 'meltpoint') else 'not set'}",
        },
        {
            "Time Sample of the Mesh": f"{state.time_sample if hasattr(state, 'time_sample') else 'not set'}",
        },
        {
            "Is each time-step being shown in the PyVista interactive Viewer": f"{'Yes' if state.interactive else 'No'}",
            "View Angle": f"{state.angle if hasattr(state, 'angle') else 'not set'}",
            "Rotation around the X Axis": f"{state.elev if hasattr(state, 'elev') else 'not set'}",
            "Rotation around the Y Axis": f"{state.azim if hasattr(state, 'azim') else 'not set'}",
            "Rotation around the Z Axis": f"{state.roll if hasattr(state, 'roll') else 'not set'}",
        },
        {
            "Image Title": f"{user_options.image_title if hasattr(user_options, 'image_title') else 'not set'}",
            "Image Label": f"{user_options.image_label if hasattr(user_options, 'image_label') else 'not set'}",
        },
        {
            "Target Parts": ",".join(state.parts) if hasattr(state, "parts") else "None Set",
            "Target Nodeset": (state.nodesets[0] if len(state.nodesets) == 1 else "Error: Only specify multiple nodesets in extract mode") if hasattr(state, "nodesets") else "None Set"
        },
        {
            "Data loaded into memory": f"{'Yes' if state.loaded else 'No'}",
            "Abaqus Program": f"{state.abaqus_program if hasattr(state, 'abaqus_program') else 'not set'}",
        },
    ]

    key: str
    val: str
    sub_dict: Dict[str, str]
    max_len = 0
    for sub_dict in lines:
        for key in sub_dict.keys():
            max_len = max(max_len, len(key))

    final_state_output: str = ""
    for sub_dict in lines:
        for key, val in sub_dict.items():
            final_state_output += key.ljust(max_len) + ": " + val + "\n"
        final_state_output += "\n" 

    print(final_state_output, end="") # No ending newline because we added it above


def process_input() -> "Union[Tuple[Odb, UserOptions], pd.DataFrame]": # Returns UserOptions or Pandas Dataframe
    """
    The goal is to have a hierarchy of options. If a user passes in an option via a command-line switch, that option is set.
    If an option is not set by a switch, then the toml input file is used.
    If an option is not set by the input file (if any), then prompt for it
    Possibly also include a default config file, like in $HOME/.config? Not sure
    """

    parser: argparse.ArgumentParser = argparse.ArgumentParser(prog="python -m odbp", description="ODB Plotter")

    parser.add_argument("-v", "--version", action="store_true", help="Show the version of ODB Plotter and exit")

    # Extract mode needs the extract keyword, and then the file to extract from (odb or hdf), 
    parser.add_argument("-e", "--extract", action="store_true", help="Pass this flag to extract directly from the .odb file, rather than reading into a .hdf5 file")
    parser.add_argument("input_file", nargs="?", help=".toml file used to give input values to ODBPlotter")

    parser.add_argument("-s", "--hdf-source-directory", help="Directory from which to source .hdf5 files")
    parser.add_argument("-b", "--odb-source-directory", help="Directory from which to source .odb files")
    parser.add_argument("-r", "--results-directory", help="Directory in which to store results")

    parser.add_argument("-o", "--odb", help="Path to the desired .odb file")
    parser.add_argument("-m", "--meltpoint", help="Melting Point of the Mesh")
    parser.add_argument("-l", "--low-temp", help="Temperature lower bound, defaults to 300 K")
    parser.add_argument("-t", "--time-sample", help="Time-sample value (N for every Nth frame you extracted). Defaults to 1")

    parser.add_argument("-H", "--hdf", help="Path to desired .hdf5 file")

    parser.add_argument("-T", "--title", help="Title to save each generated file under")
    parser.add_argument("-L", "--label", help="Label to put on each generated image")

    parser.add_argument("--low-x", help="Lower X-Axis Bound")
    parser.add_argument("--high-x", help="Upper X-Axis Bound")
    parser.add_argument("--low-y", help="Lower Y-Axis Bound")
    parser.add_argument("--high-y", help="Upper Y-Axis Bound")
    parser.add_argument("--low-z", help="Lower Z-Axis Bound")
    parser.add_argument("--high-z", help="Upper Z-Axis Bound")

    parser.add_argument("--low-time", help="Lower time limit, defaults to 0 (minimum possible)")
    parser.add_argument("--high-time", help="Upper time limit, defaults to infinity (max possible)")

    parser.add_argument("-n", "--nodesets", help="Nodesets from which to Extract. Enter only one via CLI, use the .toml input file for a list")

    parser.add_argument("-a", "--abaqus", help="Abaqus executable program to extract from .odb files")

    parser.add_argument("-V", "--view", type=view, help="Viewing Angle to show the plot in. Must either be a 3-tuple of integers or the name of a pre-defined view")
    
    parser.add_argument("-R", "--run", action="store_true", help="Run the plotter immediately, fail if all required parameters are not specified")

    args: argparse.Namespace = parser.parse_args()

    # Manage the options
    # Help is handled by the library
    if args.version:
        print(f"ODB Plotter version {__version__}")
        sys.exit(0)
    
    elif args.extract:
        #return extract_from_file(args)
        return extract_from_file(args)

    else:
        return generate_cli_settings(args)


def generate_cli_settings(args: argparse.Namespace) -> "Tuple[Odb, UserOptions]":

    state: Odb = Odb()
    user_options: UserOptions = UserOptions()

    # Stage 1: User platformdirs to read base-line settings
    odbp_config_dir: str = platformdirs.user_config_dir("odbp")
    if not os.path.exists(odbp_config_dir):
        os.makedirs(odbp_config_dir)
    config_file_path: str = os.path.join(odbp_config_dir, "config.toml")
    if not os.path.exists(config_file_path):
        print(f"Generating default config file at {config_file_path}")
        base_config_file: TextIO
        config_data: str
        new_config_file: TextIO
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "config.toml"), "r") as base_config_file:
            config_data = base_config_file.read()

        with open(config_file_path, "w") as new_config_file:
            new_config_file.write(config_data)

    config_file: TextIO
    config_settings: Dict[str, Any]
    with open(config_file_path, "rb") as config_file:
        config_settings = tomllib.load(config_file)
    state, user_options = read_setting_dict(state, user_options, config_settings)

    # Stage 2: Use the input_file if it exists
    if args.input_file:
        input_file_path: str = args.input_file
        if not os.path.exists(input_file_path):
            if not os.path.exists(os.path.join(os.getcwd(), input_file_path)):
                print(F"Error: Input File {input_file_path} could not be found")
                sys.exit(1)
            else:
                input_file_path = os.path.join(os.getcwd(), input_file_path)

        input_file: TextIO
        input_settings: Dict[str, Any]
        with open(input_file_path, "rb") as input_file:
            input_settings = tomllib.load(input_file)

        state, user_options = read_setting_dict(state, user_options, input_settings)

    # Stage 3: pass in the (other) dict values from args
    cli_flags: Dict[str, any] = vars(args)
    cli_flags = {k: v for k, v in cli_flags.items() if v is not None}
    state, user_options = read_setting_dict(state, user_options, cli_flags)

    if not os.path.exists(user_options.results_directory):
        print(f"Directory {user_options.results_directory} does not exist. Creating it now.")
        os.makedirs(user_options.results_directory)

    """if not state.hdf_processed:
        if hasattr(state, "odb_file_path") and hasattr(state, "hdf_file_path"):
            if confirm(f"{state.odb_file_path} can be automatically converted to {state.hdf_file_path} with time sample {state.time_sample}", "Would you like to perfrom this conversion?", "yes"):
                print(f"Converting {state.odb_file_path} file to .hdf5 file with name: {state.hdf_file_path}")
                state.odb_to_hdf(state.hdf_file_path)
            else:
                print("You may perform this conversion later with the \"convert\" command")"""

    return (state, user_options)


def extract_from_file(args: argparse.Namespace) -> pd.DataFrame:
    pass


def read_setting_dict(state: Odb, user_options: UserOptions, settings_dict: "Dict[str, Any]") -> "Tuple[Odb, UserOptions]":
    hdf_source_dir: str
    if "hdf_source_directory" in settings_dict:
        hdf_source_dir = os.path.abspath(settings_dict["hdf_source_directory"])

        if not os.path.exists(hdf_source_dir):
            print(f"Error: The directory {hdf_source_dir} does not exist")
            sys.exit(1)

    else:
        if not hasattr(user_options, "hdf_source_directory"):
            hdf_source_dir = os.getcwd()
        else:
            hdf_source_dir = user_options.hdf_source_directory

    user_options.hdf_source_directory = hdf_source_dir

    odb_source_dir: str
    if "odb_source_directory" in settings_dict:
        odb_source_dir = os.path.abspath(settings_dict["odb_source_directory"])

        if not os.path.exists(odb_source_dir):
            print(f"Error: The directory {odb_source_dir} does not exist")
            sys.exit(1)

    else:
        if not hasattr(user_options, "odb_source_directory"):
            odb_source_dir = os.getcwd()
        else:
            odb_source_dir = user_options.odb_source_directory

    user_options.odb_source_directory = odb_source_dir

    results_dir: str
    if "results_directory" in settings_dict:
        results_dir = os.path.abspath(settings_dict["results_directory"])

    else:
        if not hasattr(user_options, "results_directory"):
            results_dir = os.path.join(os.getcwd(), "results")
        else:
            results_dir = user_options.results_directory

    user_options.results_directory = results_dir

    if "odb" in settings_dict:
        # Ensure the file exists
        given_odb_file_path: str = settings_dict["odb"]
        if isinstance(state.select_odb(user_options, given_odb_file_path), bool):
            print(f"Error: The file {given_odb_file_path} could not be found")
            sys.exit(1)


        if "meltpoint" in settings_dict:
            state.set_meltpoint(settings_dict["meltpoint"])

        if "low_temp" in settings_dict:
            state.set_low_temp(settings_dict["low_temp"])

        if "time_sample" in settings_dict:
            state.set_time_sample(settings_dict["time_sample"])

        if "hdf" in settings_dict:
            state.hdf_file_path = os.path.join(user_options.hdf_source_directory, settings_dict["hdf"])

    elif "hdf" in settings_dict:
        # ensure the file exists
        given_hdf_file_path: str = settings_dict["hdf"]
        output: Union[UserOptions, bool] = state.select_hdf(user_options, given_hdf_file_path)
        if isinstance(output, bool):
            print(f"Error: the file {given_hdf_file_path} could not be found")
            sys.exit(1)

        else:
            user_options = output

        # If none of these values are set, read as many as are available out of the .toml config file
        # Otherwise, the file must have already been read

        # Search for the stored toml values for this hdf
        config: Optional[Dict[str, Any]] = None
        if user_options.config_file_path is None:
            print(f".toml config file for {state.hdf_file_path} could not be found")
        else:
            config_file: TextIO
            with open(user_options.config_file_path, "rb") as config_file:
                config = tomllib.load(config_file)

        if config is not None and "meltpoint" in config:
            state.set_meltpoint(config["meltpoint"])
        elif "meltpoint" in settings_dict:
            state.set_meltpoint(settings_dict["meltpoint"])

        if config is not None and "low_temp" in config:
            state.set_low_temp(config["low_temp"])
        elif "low_temp" in settings_dict:
            state.set_low_temp(settings_dict["low_temp"])

        if config is not None and "time_sample" in config:
            state.set_time_sample(config["time_sample"])
        elif "time_sample" in settings_dict:
            state.set_time_sample(settings_dict["time_sample"])

    else: # The case where a .odb file and a .hdf5 file were not provided

        if "meltpoint" in settings_dict:
            state.set_meltpoint(settings_dict["meltpoint"])

        if "low_temp" in settings_dict:
            state.set_low_temp(settings_dict["low_temp"])

        if "time_sample" in settings_dict:
            state.set_time_sample(settings_dict["time_sample"])

    # The 4 dimensions and the two booleans are always set here if they exist
    # The 5th dimension, temperature, is not set here because it shouldn't
    # change based on the file, even if the spatial
    # or temporal dimensions change

    if "low_x" in settings_dict:
        state.set_x_low(settings_dict["low_x"])
    if "high_x" in settings_dict:
        state.set_x_high(settings_dict["high_x"])

    if "low_y" in settings_dict:
        state.set_y_low(settings_dict["low_y"])
    if "high_y" in settings_dict:
        state.set_y_high(settings_dict["high_y"])

    if "low_z" in settings_dict:
        state.set_z_low(settings_dict["low_z"])
    if "high_z" in settings_dict:
        state.set_z_high(settings_dict["high_z"])

    if "low_time" in settings_dict:
        state.set_time_low(settings_dict["low_time"])
    if "high_time" in settings_dict:
        state.set_time_high(settings_dict["high_time"])

    image_title: str
    image_label: str
    if "title" in settings_dict:
        image_title = settings_dict["title"]
    else:
        if user_options.image_title == "":
            if hasattr(state, "hdf_file_path"):
                image_title = state.hdf_file_path.split(os.sep)[-1].split(".")[0]

            elif hasattr(state, "odb_file_path"):
                image_title = state.odb_file_path.split(os.sep)[-1].split(".")[0]

            else:
                image_title = ""
        else:
            image_title = user_options.image_title

    if "label" in settings_dict:
        image_label = settings_dict["label"]
    else:
        if user_options.image_label == "":
            image_label = image_title
        else:
            image_label = user_options.image_label

    user_options.image_title = image_title
    user_options.image_label = image_label

    if "colormap_name" in settings_dict:
        state.colormap_name = settings_dict["colormap_name"]

    # Run Immediate
    if "run" in settings_dict:
        if not user_options.run_immediate:
            user_options.run_immediate = settings_dict["run"]

    if "interactive" in settings_dict:
        state.interactive = settings_dict["interactive"]
    else:
        if not hasattr(state, "interactive"):
            state.interactive = True

    # Views
    if "view" in settings_dict:
        # Views can either be a string or a dict
        if isinstance(settings_dict["view"], dict):
            state.angle = "custom"
            state.elev = settings_dict["view"]["elev"]
            state.azim = settings_dict["view"]["azim"]
            state.roll = settings_dict["view"]["roll"]
        
        else:
            given_view: Dict[str, int] = view(settings_dict["view"])
            elev: int = given_view["elev"]
            azim: int = given_view["azim"]
            roll: int = given_view["roll"]
            state.angle=settings_dict["view"]
            state.elev = elev
            state.azim = azim
            state.roll = roll

    if "parts" in settings_dict:
        state.set_parts(settings_dict["parts"])

    if "nodes" in settings_dict:
        state.set_nodes(settings_dict["nodes"])

    if hasattr(state, "nodesets"):
        if "nodesets" in settings_dict:
            nodesets = settings_dict["nodesets"]
            if type(nodesets) is list:
                state.set_nodesets(nodesets)
            elif type(nodesets) is str:
                state.set_nodesets([nodesets])
    else:
        nodesets = settings_dict.get("nodesets", ["ALL NODES"])
        if type(nodesets) is list:
            state.set_nodesets(nodesets)
        elif type(nodesets) is str:
            state.set_nodesets([nodesets])

    if hasattr(state, "abaqus_program"):
        if "abaqus" in settings_dict:
            state.set_abaqus(settings_dict["abaqus"])
    else:
        state.set_abaqus(settings_dict.get("abaqus", "abaqus"))

    return (state, user_options)


def load_views_dict() -> "Dict[str, Dict[str, int]]":
    views_file: TextIO
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "views.toml"), "rb") as views_file:
        return tomllib.load(views_file)


# Used to define the "view" type for argparse
def view(string: str) -> "Dict[str, int]":
    views_dict: Dict[str, Dict[str, int]] = load_views_dict()
    if string in views_dict:
        return views_dict[string]

    # Otherwise, try to parse the user's input as a 3-tuple
    formatted_string: str = string.replace("(", "")
    formatted_string = formatted_string.replace(")", "")
    formatted_string = formatted_string.replace("[", "")
    formatted_string = formatted_string.repalce("]", "")
    try:
        elev: int
        azim: int
        roll: int
        elev, azim, roll = [int(f) for f in formatted_string.split(",")]
        return {"elev": elev, "azim": azim, "roll": roll}

    except:
        raise argparse.ArgumentTypeError("View must be one of the default views or a 3-tuple of integer angles (a, b, c)")
