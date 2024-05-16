    Before 0.5.0: Did not have the Changelog here.
    0.5.0: API Updates and better dataframe filtering
        0.5.1: Implement new system information (pypi tags, this changelog)
        0.5.2: Returning support to Python 3.8+ (type hinting)
        0.5.3: Patching conversion bugs
        0.5.4: Parametrize number of cpus for testing
    0.6.0: Extractor improvements, ODB interface tools (iteration, receiving ODB data), re-implementation of basic 3D plots over time (including melt-pool plots). Created two-dimensional plotting capabilities
        0.6.1: Update notices if pyvista isn't installed
        0.6.2: Improve data extraction for plotting. Ensure that plotting doesn't fork-bomb
        0.6.3: Actually filtering 3D plots.
        0.6.4: Fixing Python2 Error Reporting
        0.6.5: Fixing Python2 Error Reporting in more places
        0.6.6: Fixing conversion problems
        0.6.7: Implementing tools for .odbs with coords in only some steps or tools for frame steps with different sizes
    0.7.0: Improve user settings, parameterization, metadata. Let users select plotting colors, keep metadata of nodesets or spatial, thermal, temporal bounds within the .hdf5. Reqwrite CLI to use Python's cmd module and pyreadline/GNU readline.
        0.7.1: Data Output fixes, sane defaults, fixing typos, improved plotting, including 2D and better regard to time ranges.
    0.8.0: Updates to how coordinates are stored in .hdf5 files.
        0.8.1: Updating color backgrounds for 3D images

    1.0.0: Minor bugfixes for release aloginside publication, in addition to rudimentary documentation and corresponding .odb files in this git repository
        1.0.1: Minor bugfix when populating default config file
        1.0.2: Implementing default plotting views and non-interactive plotting

    1.1.0: Changed name to ODBP to reflect greater general capabilities. Switched to Polyscope for Plotting. Now storing per-frame positional data and re-emulating Abaqus meshing. Improved filtering and putting bounds on outputs. Added more viewing angles.
        1.1.1: Ensuring everything is using the new ODBP name
        1.1.2: Fixing a minor crash and dependencies
        1.1.3: More dependency changes, updating documentation