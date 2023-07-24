# ODB Plotter

## Constraints:
ODB Plotter is being developed by [CMML](https://www.cmml.me.msstate.edu), and as such has a focus on Additive Manufacturing and Temperature Data.

## Install with pip
```shell
pip install odb-plotter["all"]
```

## Install in headless mode (data processing only)
```shell
pip install odb-plotter
```

## Run the cli with python
```shell
python -m odbp
```

## Or import to use the api
```python
from odbp import Odb
...
```

## ODB Plotter Design Goals

### I intend for this project to serve two purposes:
- First, implement an extensible, flexible api for accessing data within .odb files or .hdf5 files with odb data
- Second, implement a user-friendly, sane-defaults cli to allow for quick data extraction, manipulation, and visualization with no hassle

## Changelog
* Before 0.5.0: Did not have the Changelog here.
* 0.5.0: API Updates and better dataframe filtering
    * 0.5.1: Implement new system information (pypi tags, this changelog)
    * 0.5.2: Returning support to Python 3.8+ (type hinting)
    * 0.5.3: Patching conversion bugs
    * 0.5.4: Parametrize number of cpus for testing
* 0.6.0: Extractor improvements, ODB interface tools (iteration, receiving ODB data), re-implementation of basic 3D plots over time (including melt-pool plots). Created two-dimensional plotting capabilities
    * 0.6.1: Update notices if pyvista isn't installed
    * 0.6.2: Improve data extraction for plotting. Ensure that plotting doesn't fork-bomb
    * 0.6.3: Actually filtering 3D plots.
    * 0.6.4: Fixing Python2 Error Reporting
    * 0.6.5: Fixing Python2 Error Reporting in more places
    * 0.6.6: Fixing conversion problems
    * 0.6.7: Implementing tools for .odbs with coords in only some steps or tools for frame steps with different sizes
* 0.7.0: Improve user settings, parameterization, metadata. Let users select plotting colors, keep metadata of nodesets or spatial, thermal, temporal bounds within the .hdf5. Reqwrite CLI to use Python's cmd module and pyreadline/GNU readline.
* Upcoming:
    * 1.0.0: Final bug-fixing, type checking, bounds checking, etc. Hopefully coinciding with (or following) a publication.
    * 1.1.0: Improve pyvista functionality: views, gifs, non-interactive image saving, leave the viewer.
