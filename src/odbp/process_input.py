""""""
import argparse
import pathlib

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
import sys
import platformdirs
import shutil
from typing import Any, BinaryIO, Dict
from .odb import Odb
from odbp import __version__

"""
plot/show
"""


def process_input() -> Odb:
    """
    The goal is to have a hierarchy of options. If a user passes in an option via a command-line switch, that option is set.
    If an option is not set by a switch, then the toml input file is used.
    If an option is not set by the input file (if any), then prompt for it
    Possibly also include a default config file, like in $HOME/.config? Not sure
    """

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="python -m odbp", description="ODB Plotter"
    )

    parser.add_argument(
        "-v",
        "--version",
        action="store_true",
        help="Show the version of ODB Plotter and exit",
    )

    parser.add_argument(
        "input_file",
        nargs="?",
        help=".toml file used to give input values to ODBPlotter",
    )

    parser.add_argument(
        "-s", "--hdf-source-dir", help="Directory from which to source .hdf5 files"
    )
    parser.add_argument(
        "-b", "--odb-source-dir", help="Directory from which to source .odb files"
    )
    parser.add_argument(
        "-r", "--result-dir", help="Directory in which to store results"
    )

    parser.add_argument("-o", "--odb_path", help="Path to the desired .odb file")

    parser.add_argument("-H", "--hdf_path", help="Path to desired .hdf5 file")

    parser.add_argument("--x-low", help="Lower X-Axis Bound")
    parser.add_argument("--x-high", help="Upper X-Axis Bound")
    parser.add_argument("--y-low", help="Lower Y-Axis Bound")
    parser.add_argument("--y-high", help="Upper Y-Axis Bound")
    parser.add_argument("--z-low", help="Lower Z-Axis Bound")
    parser.add_argument("--z-high", help="Upper Z-Axis Bound")

    parser.add_argument(
        "--time-low", help="Lower time limit, defaults to 0 (minimum possible)"
    )
    parser.add_argument(
        "--time-high", help="Upper time limit, defaults to infinity (max possible)"
    )

    parser.add_argument("--temp-low", help="Temperature lower bound, defaults to 300 K")
    parser.add_argument("--temp-high", help="Melting Point of the Mesh")
    parser.add_argument(
        "-t",
        "--time-step",
        help="Time-step value (N for every Nth frame you extracted). Defaults to 1",
    )

    # TODO Lists
    # ???
    # parser.add_argument("-n", "--nodesets", help="Nodesets from which to Extract. Enter only one via CLI, use the .toml input file for a list")
    # nodes, parts, steps

    parser.add_argument(
        "-c", "--cpus", help="Number of cpu cores to use for this process"
    )

    # Coord Key
    parser.add_argument(
        "-k",
        "--coord-key",
        help="Value by which coordinates are keyed in the .odb or .hdf5 files, defaults to 'COORD'",
    )

    # TODO Lists
    # target_outputs ???

    parser.add_argument(
        "-a",
        "--abaqus-executable",
        help="Abaqus executable program to extract from .odb files",
    )

    parser.add_argument(
        "-m", "--colormap", help="Which colormap to use for plots. Defaults to 'turbo'"
    )

    parser.add_argument(
        "-S",
        "--save",
        help="Boolean for whether to save generated images to hard drive. Defaults to True",
    )

    parser.add_argument(
        "-f",
        "--save-format",
        help="What format should images be saved as. Default to '.png'",
    )

    parser.add_argument(
        "-F",
        "--filename",
        help="Format for the names of the saved images. Defaults to <name of the .hdf5 file>",
    )

    parser.add_argument("-T", "--title", help="Format for title shown on images.")

    parser.add_argument(
        "-n", "--font", help="Font to use on plots. Defaults to 'courier'"
    )
    parser.add_argument(
        "-R", "--font-color", help="The color for fonts. Defaults to '#000000'"
    )
    parser.add_argument("-z", "--font-size", help="The size of fonts. Defaults to 14.0")

    parser.add_argument(
        "-g",
        "--background",
        dest="background_color",
        help="Color for the backgrounds of images. Defaults to '#FFFFFF'",
    )
    parser.add_argument(
        "-A",
        "--above-range",
        dest="above_range_color",
        help="Color for above-ranges on the colormap. Defaults to '#C0C0C0'",
    )
    parser.add_argument(
        "-B",
        "--below-range",
        dest="below_range_color",
        help="Color for below-ranges on the colormap.",
    )

    parser.add_argument(
        "-V",
        "--view",
        help="Viewing Angle to show the plot in. Defaults to 'UFR-U'",
    )

    args: argparse.Namespace = parser.parse_args()

    # Manage the options
    # Help is handled by the library
    if args.version:
        print(f"ODBPlotter {__version__}")
        sys.exit(0)

    else:
        final_settings_dict: Dict[str, Any] = dict()
        odb_config_dir: pathlib.Path = pathlib.Path(
            platformdirs.user_config_dir("odbp")
        )
        if not odb_config_dir.exists():
            odb_config_dir.mkdir(parents=True)
        config_file_path: pathlib.Path = odb_config_dir / "config.toml"
        if not config_file_path.exists():
            print(f"Generating default config file at {config_file_path}")
            base_config_path: pathlib.Path = (
                pathlib.Path(__file__).parent / "data"
            ) / "config.toml"

            shutil.copyfile(base_config_path, config_file_path)

        config_file: BinaryIO
        with open(config_file_path, "rb") as config_file:
            config_file_data: Dict[str, Any] = tomllib.load(config_file)

        final_settings_dict.update(config_file_data)

        input_file_data: Dict[str, Any] = dict()
        input_file: BinaryIO
        if args.input_file:
            with open(args.input_file, "rb") as input_file:
                input_file_data = tomllib.load(input_file)

        final_settings_dict.update(input_file_data)

        cli_flags_data: Dict[str, Any] = vars(args)
        key: str
        val: Any
        cli_flags_data = {
            key: val for key, val in cli_flags_data.items() if val is not None
        }

        final_settings_dict.update(cli_flags_data)

        return generate_cli_settings(final_settings_dict)


def generate_cli_settings(settings_dict: "Dict[str, Any]") -> Odb:
    odb: Odb = Odb()

    if "hdf_source_dir" in settings_dict:
        odb.hdf_source_dir = pathlib.Path(settings_dict["hdf_source_dir"])

    if "odb_source_dir" in settings_dict:
        odb.odb_source_dir = pathlib.Path(settings_dict["odb_source_dir"])

    if "result_dir" in settings_dict:
        odb.result_dir = pathlib.Path(settings_dict["result_dir"])
        if not odb.result_dir.exists():
            print(f"Directory {odb.result_dir} does not exist. Creating it now.")
            odb.result_dir.mkdir()

    if "odb_path" in settings_dict:
        odb.odb_path = pathlib.Path(settings_dict["odb_path"])

    if "hdf_path" in settings_dict:
        odb.hdf_path = pathlib.Path(settings_dict["hdf_path"])

    if "steps" in settings_dict:
        odb.steps = settings_dict["steps"]

    if "parts" in settings_dict:
        odb.parts = settings_dict["parts"]

    if "nodes" in settings_dict:
        odb.nodes = settings_dict["nodes"]

    if "nodesets" in settings_dict:
        odb.nodesets = settings_dict["nodesets"]

    if "x_low" in settings_dict:
        odb.x_low = settings_dict["x_low"]

    if "x_high" in settings_dict:
        odb.x_high = settings_dict["x_high"]

    if "y_low" in settings_dict:
        odb.y_low = settings_dict["y_low"]

    if "y_high" in settings_dict:
        odb.y_high = settings_dict["y_high"]

    if "z_low" in settings_dict:
        odb.z_low = settings_dict["z_low"]

    if "z_high" in settings_dict:
        odb.z_high = settings_dict["z_high"]

    if "time_low" in settings_dict:
        odb.time_low = settings_dict["time_low"]

    if "time_high" in settings_dict:
        odb.time_high = settings_dict["time_high"]

    if "temp_low" in settings_dict:
        odb.temp_low = settings_dict["temp_low"]

    if "temp_high" in settings_dict:
        odb.temp_high = settings_dict["temp_high"]

    if "time_step" in settings_dict:
        odb.time_step = settings_dict["time_step"]

    if "cpus" in settings_dict:
        odb.cpus = settings_dict["cpus"]

    if "abaqus_executable" in settings_dict:
        odb.abaqus_executable = settings_dict["abaqus_executable"]

    if "colormap" in settings_dict:
        odb.colormap = settings_dict["colormap"]

    if "save" in settings_dict:
        odb.save = settings_dict["save"]

    if "save_format" in settings_dict:
        odb.save_format = settings_dict["save_format"]

    if "filename" in settings_dict:
        odb.filename = settings_dict["filename"]

    if "title" in settings_dict:
        odb.title = settings_dict["title"]

    if "font" in settings_dict:
        odb.font = settings_dict["font"]

    if "font_color" in settings_dict:
        odb.font_color = settings_dict["font_color"]

    if "font_size" in settings_dict:
        odb.font_size = settings_dict["font_size"]

    if "background_color" in settings_dict:
        odb.background_color = settings_dict["background_color"]

    if "above_range_color" in settings_dict:
        odb.above_range_color = settings_dict["above_range_color"]

    if "below_range_color" in settings_dict:
        odb.below_range_color = settings_dict["below_range_color"]

    if "view" in settings_dict:
        odb.view = settings_dict["view"]

    # TODO prompt to convert from here

    return odb
