# ODB Plotter

## Constraints:
ODB Plotter is being developed by [CMML](https://www.cmml.me.msstate.edu), and as such has a focus on Additive Manufacturing and Temperature Data.

## Install with pip
```shell
pip install odb-plotter
```

## ODB Plotter is a Plotter. It is also a Data Science Tool
## Install with plotting capabilities
```shell
pip instlal odb-plotter["plot"]
```

## OR
```shell
pip instlal odb-plotter["all"]
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
* Upcoming:
    * 0.7.0: Improve pyvista functionality: views, gifs, non-interactive image saving, leave the viewer.
    * 0.8.0: Rewrite CLI to use python's cmd module and pyreadline/GNU readline. Also improve user input parametrization (nodes, nodesets, frames, steps, parts, colors, image sizes, etc.)
    * 1.0.0: Final bug-fixing, type checking, bounds checking, etc. Hopefully coinciding (or following) a publication.
