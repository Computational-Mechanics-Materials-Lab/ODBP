# ODB Plotter

## Constraints:
ODB Plotter is being developed by [CMML](https://www.cmml.me.msstate.edu), and as such has a focus on Additive Manufacturing and Temperature Data.

## Install with pip
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
    * 0.5.1 Implement new system information (pypi tags, this changelog)
    * 0.5.2 Returning support to Python 3.8+ (type hinting)
* Upcoming:
    * 0.6.0: Improved extractor across all file types. Improved Odb object iteration.
    * 0.7.0: Rewrite CLI to use python's cmd module and pyreadline/GNU readline
    * 0.8.0: Parametrize input values such as nodes, nodesets, frames, steps, parts, and colors (both in the API and CLI).
    * 0.9.0: Create two-dimensional plotting capabilities and sane defaults.
    * 0.10.0: Improve PyVista: views, gifs, non-interactive image saving, leaving viewer, etc. Ensure functionality of Abaqus 2019
    * 1.0.0: Final bug-fixing, type checking, bounds checking, etc. Hopefully coinciding (or following) a publication.
