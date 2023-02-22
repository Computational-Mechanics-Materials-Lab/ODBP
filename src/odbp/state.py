import argparse
import toml
import sys
import os
import pandas as pd
from typing import Union, Any, TypeAlias, TextIO
from .odb_visualizer import OdbVisualizer
from odbp import __version__


ConfigFileType: TypeAlias = Union[str, None]
ViewsDict: TypeAlias = dict[str, dict[str, int]]


class UserOptions():
    """
    Struct to store user's input
    """
    def __init__(self) -> None:
        """
        Default values for user Options:
        hdf_source_directory: user's present working directory
        odb_source_directory: user's present working direcotry
        results_directory: user's present working directory
        image_title: name of the .hdf5 file + ".png"
        image_label: name of the .hdf5 file
        run_immediate: False
        """
        self.hdf_source_directory: str = "" 
        self.odb_source_directory: str = "" 
        self.results_directory: str = ""
        self.image_title: str = ""
        self.image_label: str = ""
        self.config_file_path: ConfigFileType = None
        self.run_immediate: bool = False


SettingType: TypeAlias = tuple[OdbVisualizer, UserOptions]


class CLIOptions():
    """
    Struct to store cli options without repeating
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

        self.low_temp_options: list[str] = ["low", "low-temp"]
        self.low_temp_help: str = "Set the Lower Temperate Bound for the hdf5 file"
        self.low_temp_options_formatted: str = ", ".join(self.low_temp_options)

        self.title_label_options: list[str] = ["title", "label"]
        self.title_label_help: str = "Set the title and label of the output plots"
        self.title_label_options_formatted: str = ", ".join(self.title_label_options)

        self.directory_options: list[str] = ["dir", "dirs", "directory", "directories"]
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
                len(self.low_temp_options_formatted),
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
{self.low_temp_options_formatted.ljust(self.longest_len)} -- {self.low_temp_help}
{self.title_label_options_formatted.ljust(self.longest_len)} -- {self.title_label_help}
{self.directory_options_formatted.ljust(self.longest_len)} -- {self.directory_help}
{self.process_options_formatted.ljust(self.longest_len)} -- {self.process_help}
{self.angle_options_formatted.ljust(self.longest_len)} -- {self.angle_help}
{self.show_all_options_formatted.ljust(self.longest_len)} -- {self.show_all_help}
{self.plot_options_formatted.ljust(self.longest_len)} -- {self.plot_help}
{self.state_options_formatted.ljust(self.longest_len)} -- {self.state_help}"""
    )


def print_state(state: OdbVisualizer, user_options: UserOptions) -> None:
    lines: list[dict[str, str]] = [
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
            "X Range": f"{state.x.low if hasattr(state.x, 'low') else 'not set'} to {state.x.high - state.mesh_seed_size if hasattr(state.x, 'high') and hasattr(state, 'mesh_seed_size') else 'not set'}",
            "Y Range": f"{state.y.low if hasattr(state.y, 'low') else 'not set'} to {state.y.high - state.mesh_seed_size if hasattr(state.y, 'high') and hasattr(state, 'mesh_seed_size') else 'not set'}",
            "Z Range": f"{state.z.low if hasattr(state.z, 'low') else 'not set'} to {state.z.high - state.mesh_seed_size if hasattr(state.z, 'high') and hasattr(state, 'mesh_seed_size') else 'not set'}",
            "Time Range": f"{state.time_low if hasattr(state, 'time_low') else 'not set'} to {state.time_high if hasattr(state, 'time_high') else 'not set'}",
            "Temperature Range": f"{state.low_temp if hasattr(state, 'low_temp') else 'not set'} to {state.meltpoint if hasattr(state, 'meltpoint') else 'not set'}",
        },
        {
            "Seed Size of the Mesh": f"{state.mesh_seed_size if hasattr(state, 'mesh_seed_size') else 'not set'}",
            "Time Sample of the Mesh": f"{state.time_sample if hasattr(state, 'time_sample') else 'not set'}",
        },
        {
            "Is each time-step being shown in the PyVista interactive Viewer": f"{'Yes' if state.interactive else 'No'}",
            "View Angle": f"{state.angle if hasattr(state, 'angle') else 'not set'}",
            "View Elevation": f"{state.elev if hasattr(state, 'elev') else 'not set'}",
            "View Azimuth": f"{state.azim if hasattr(state, 'azim') else 'not set'}",
            "View Roll": f"{state.roll if hasattr(state, 'roll') else 'not set'}",
        },
        {
            "Image Title": f"{user_options.image_title if hasattr(state, 'image_title') else 'not set'}",
            "Image Label": f"{user_options.image_label if hasattr(state, 'image_label') else 'not set'}",
        },
        {
            "Data loaded into memory": f"{'Yes' if state.loaded else 'No'}",
        },
    ]

    key: str
    val: str
    sub_dict: dict[str, str]
    max_len = 0
    for sub_dict in lines:
        for key, _ in sub_dict:
            max_len = max(max_len, len(key))

    final_state_output: str = ""
    for sub_dict in lines:
        for key, val in sub_dict:
            final_state_output += key.ljust(max_len) + ": " + val + "\n"
        final_state_output += "\n" 

    print(final_state_output, end="") # No ending newline because we added it above


def process_input() -> Union[SettingType, Any]: # Returns UserOptions or Pandas Dataframe
    """
    The goal is to have a hierarchy of options. If a user passes in an option via a command-line switch, that option is set.
    If an option is not set by a switch, then the toml input file is used.
    If an option is not set by the input file (if any), then prompt for it
    Possibly also include a default config file, like in $HOME/.config? Not sure
    """

    parser: argparse.ArgumentParser = argparse.ArgumentParser(prog="python -m odbp", description="ODB Plotter")

    parser.add_argument("-v", "--version", help="Show the version of ODB Plotter and exit")

    subparsers: argparse.ArgumentParser = parser.add_subparsers(help='use either the "extract" command to pull data directly from a .odb file or the "plot" command to create visual plots from data stored in .odbs or .hdf5s')

    # Extract mode needs the extract keyword, and then the file to extract from (odb or hdf), 
    extract_parser: argparse.ArgumentParser = subparsers.add_parser("extract")
    extract_parser.add_argument("extract", action="store_true", help="Flag to denote extract action in lieu of parse action")
    extract_parser.add_argument("input-file", nargs="?", help=".toml file for which nodesets to extract from")
    extract_parser.add_argument("-o", "--odb", help="Path to the desired .odb file")
    #extract_parser.add_argument("-H", "--hdf", help="Path to the desired .hdf5 file")

    plot_parser: argparse.ArgumentParser = subparsers.add_parser("plot")
    plot_parser.add_argument("input-file", nargs="?", help=".toml file used to give input values to ODBPlotter")

    plot_parser.add_argument("-s", "--hdf-source-directory", help="Directory from which to source .hdf5 files")
    plot_parser.add_argument("-b", "--odb-source-directory", help="Directory from which to source .odb files")
    plot_parser.add_argument("-r", "--results-directory", help="Directory in which to store results")

    plot_parser.add_argument("-o", "--odb", help="Path to the desired .odb file")
    plot_parser.add_argument("-m", "--meltpoint", help="Melting Point of the Mesh")
    plot_parser.add_argument("-l", "--low-temp", help="Temperature lower bound, defaults to 300 K")
    plot_parser.add_argument("-S", "--mesh-seed-size", help="Mesh seed size of the .odb file")
    plot_parser.add_argument("-t", "--time-sample", help="Time-sample value (N for every Nth frame you extracted). Defaults to 1")

    plot_parser.add_argument("-H", "--hdf", help="Path to desired .hdf5 file")

    plot_parser.add_argument("-T", "--title", help="Title to save each generated file under")
    plot_parser.add_argument("-L", "--label", help="Label to put on each generated image")

    plot_parser.add_argument("--low-x", help="Lower X-Axis Bound")
    plot_parser.add_argument("--high-x", help="Upper X-Axis Bound")
    plot_parser.add_argument("--low-y", help="Lower Y-Axis Bound")
    plot_parser.add_argument("--high-y", help="Upper Y-Axis Bound")
    plot_parser.add_argument("--low-z", help="Lower Z-Axis Bound")
    plot_parser.add_argument("--high-z", help="Upper Z-Axis Bound")

    plot_parser.add_argument("--low-time", help="Lower time limit, defaults to 0 (minimum possible)")
    plot_parser.add_argument("--high-time", help="Upper time limit, defaults to infinity (max possible)")

    plot_parser.add_argument("-V", "--view", type=view, help="Viewing Angle to show the plot in. Must either be a 3-tuple of ")
    
    plot_parser.add_argument("-R", "--run", action="store_true", help="Run the plotter immediately, fail if all required parameters are not specified")

    args: argparse.Namespace = parser.parse_args()

    # Manage the options
    # Help is handled by the library
    if args.version:
        print(f"ODB Plotter version {__version__}")
        sys.exit(0)
    
    elif args.extract:
        return extract_from_file(args)

    else:
        return generate_cli_settings(args)


def generate_cli_settings(args: argparse.Namespace) -> SettingType:

    state: OdbVisualizer = OdbVisualizer()
    user_options: UserOptions = UserOptions()

    # Stage 1: User platformdirs to read base-line settings
    # config_file_path: str = <config_file_path>
    # if not os.path.exists(<config_file_path>):
    #    print("Generating default config file at <config_file_path>")
    #    generate_config_file()
    # config_file: TextIO
    # config_settings: dict[str, Any]
    # with open(config_file_path, "r") as config_File:
    #    config_settings = toml.read(config_file)
    # state, user_options = read_setting_dict(state, user_options, config_settings)

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
        input_settings: dict[str, Any]
        with open(input_file_path, "r") as input_file:
            input_settings = toml.read(input_file)

        state, user_options = read_setting_dict(state, user_options, input_settings)

    # Stage 3: pass in the (other) dict values from args
    state, user_options = read_setting_dict(state, user_options, vars(args))

    return (state, user_options)


def extract_from_file(args: argparse.Namespace) -> pd.DataFrame:
    pass


def read_setting_dict(state: OdbVisualizer, user_options: UserOptions, settings_dict: dict[str, Any]) -> SettingType:

    hdf_source_dir: str
    if "hdf_source_directory" in settings_dict:
        hdf_source_dir = os.path.abspath(settings_dict["hdf_source_directory"])

        if not os.path.exists(hdf_source_dir):
            print(f"Error: The directory {hdf_source_dir} does not exist")
            sys.exit(1)

    else:
        hdf_source_dir = os.getcwd()

    user_options.hdf_source_directory = hdf_source_dir

    odb_source_dir: str
    if "odb_source_directory" in settings_dict:
        odb_source_dir = os.path.abspath(settings_dict["odb_source_directory"])

        if not os.path.exists(odb_source_dir):
            print(f"Error: The directory {odb_source_dir} does not exist")
            sys.exit(1)

    else:
        odb_source_dir = os.getcwd()

    user_options.odb_source_directory = odb_source_dir

    results_dir: str
    if "results_directory" in settings_dict:
        results_dir = os.path.abspath(settings_dict["results_directory"])

    if not os.path.exists(results_dir):
        print(f"Directory {results_dir} does not exist. Creating it now.")
        os.makedirs(results_dir)

    user_options.results_directory = results_dir

    if "odb" in settings_dict:
        # Ensure the file exists
        given_odb_file_path: str = settings_dict["odb"]
        odb_file_path: str
        if not os.path.exists(os.path.join(user_options.odb_source_directory, given_odb_file_path)):
            if not os.path.exists(os.path.join(os.getcwd(), given_odb_file_path)):
                if not os.path.exists(given_odb_file_path):
                    print(f"Error: The file {given_odb_file_path} could not be found")
                    sys.exit(1)

                else:
                    odb_file_path = given_odb_file_path
            else:
                odb_file_path = os.path.join(os.getcwd(), given_odb_file_path)
        else:
            odb_file_path = os.path.join(user_options.odb_source_directory, given_odb_file_path)

        state.odb_file_path = odb_file_path

        if "meltpoint" in settings_dict:
            state.meltpoint = settings_dict["meltpoint"]

        if "low_temp" in settings_dict:
            state.low_temp = settings_dict["low_temp"]

        if "mesh_seed_size" in settings_dict:
            state.mesh_seed_size = settings_dict["mesh_seed_size"]

        if "time_sample" in settings_dict:
            state.time_sample = settings_dict["time_sample"]

        if "hdf" in settings_dict:
            new_hdf_file_name: str = os.path.join(user_options.hdf_source_directory, settings_dict["hdf"])
            print(f"Converting .odb file to .hdf5 file with name: {new_hdf_file_name}")
            state.odb_to_hdf(new_hdf_file_name)

        else:
            print('You passed in a .odb file. You must manually convert it to a .hdf5 file using the "convert" command, or by passing in the name of the target .hdf5 file to convert automatically.')

    elif "hdf" in settings_dict:
        # ensure the file exists
        given_hdf_file_path: str = settings_dict["hdf"]
        hdf_file_path: str
        if not os.path.exists(os.path.join(user_options.hdf_source_directory, given_hdf_file_path)):
            if not os.path.exists(os.path.join(os.getcwd(), given_hdf_file_path)):
                if not os.path.exists(given_hdf_file_path):
                    print(f"Error: the file {given_hdf_file_path} could not be found")
                    sys.exit(1)

                else:
                    hdf_file_path = given_hdf_file_path
            else:
                hdf_file_path = os.path.join(os.getcwd(), given_hdf_file_path)
        else:
            hdf_file_path = os.path.join(user_options.hdf_source_directory, given_hdf_file_path)

        state.hdf_file_path = hdf_file_path

        # If none of these values are set, read as many as are available out of the .toml config file
        # Otherwise, the file must have already been read
        if not hasattr(state, "meltpoint") and not hasattr(state, "low_temp") and not hasattr(state, "mesh_seed_size") and not hasattr(state, "time_sample"):

            # Search for the stored toml values for this hdf
            config_file_path: ConfigFileType = os.path.join(user_options.hdf_source_directory, state.hdf_file_path.split(".")[0] + ".toml")
            config: Union[dict[str, Any], None] = None
            if not os.path.exists(config_file_path):
                print(f".toml config file for {state.hdf_file_path} could not be found")
            else:
                user_options.config_file_path = config_file_path
                config_file: TextIO
                with open(config_file_path, "r") as config_file:
                    config = toml.load(config_file)

            if "meltpoint" in config:
                state.meltpoint = config["meltpoint"]
            elif "meltpoint" in settings_dict:
                state.meltpoint = settings_dict["meltpoint"]

            if "low_temp" in config:
                state.low_temp = config["low_temp"]
            elif "low_temp" in settings_dict:
                state.low_temp = settings_dict["low_temp"]

            if "mesh_seed_size" in config:
                state.mesh_seed_size = config["mesh_seed_size"]
            elif "mesh_seed_size" in settings_dict:
                state.mesh_seed_size = settings_dict["mesh_seed_size"]

            if "time_sample" in config:
                state.time_sample = config["time_sample"]
            elif "time_sample" in settings_dict:
                state.time_sample = settings_dict["time_sample"]

    else: # The case where a .odb file and a .hdf5 file were not provided

        if "meltpoint" in settings_dict:
            state.meltpoint = settings_dict["meltpoint"]

        if "low_temp" in settings_dict:
            state.low_temp = settings_dict["low_temp"]

        if "mesh_seed_size" in settings_dict:
            state.mesh_seed_size = settings_dict["mesh_seed_size"]

        if "time_sample" in settings_dict:
            state.time_sample = settings_dict["time_sample"]

    # The 4 dimensions and the two booleans are always set here if they exist
    # The 5th dimension, temperature, is not set here because it shouldn't change based on
    # the file, even if the spatial or temporal dimensions change
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
    if "high_time":
        state.set_time_high(settings_dict["high_time"])

    image_title: str
    image_label: str
    if "title" in settings_dict:
        image_title = settings_dict["title"]
    else:
        if hasattr(state, "hdf_file_path"):
            image_title = state.hdf_file_path.split(os.sep)[-1].split(".")[0]

        elif hasattr(state, "odb_file_path"):
            image_title = state.odb_file_path.split(os.sep)[-1].split(".")[0]

        else:
            image_title = ""

    if "label" in settings_dict:
        image_label = settings_dict["label"]
    else:
        image_label = image_title

    user_options.image_title = image_title
    user_options.image_label = image_label

    # Run Immediate
    if "run" in settings_dict:
        user_options.run_immediate = settings_dict["run"]

    # Views
    if "view" in settings_dict:
        state.x_rot = settings_dict["view"]["x_rot"]
        state.y_rot = settings_dict["view"]["y_rot"]
        state.z_rot = settings_dict["view"]["z_rot"]

    return (state, user_options)


def load_views_dict() -> ViewsDict:
    views_file: TextIO
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "views.toml"), "r") as views_file:
        return toml.load(views_file)


# Used to define the "view" type for argparse
def view(string: str) -> dict[str, int]:
    views_dict: ViewsDict = load_views_dict()
    if string in views_dict:
        return views_dict[string]

    # Otherwise, try to parse the user's input as a 3-tuple
    formatted_string: str = string.replace("(", "")
    formatted_string = formatted_string.replace(")", "")
    formatted_string = formatted_string.replace("[", "")
    formatted_string = formatted_string.repalce("]", "")
    try:
        x_rot: int
        y_rot: int
        z_rot: int
        x_rot, y_rot, z_rot = map(int, formatted_string.split(","))
        return {"x_rot": x_rot, "y_rot": y_rot, "z_rot": z_rot}

    except:
        raise argparse.ArgumentTypeError("View must be one of the default views or a 3-tuple of integer angles (a, b, c)")