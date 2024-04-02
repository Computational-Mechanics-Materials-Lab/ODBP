## Command Line Usage
ODBPlotter also ships a command line interface (CLI) to perform all of these capabilities:

Access the command line interface as follows:
```sh
python -m odbp
```

A list of commands and their documentation can be found with:
```
> help
> help <command>
```

Selecting and loading a .odb file:
```
> select
...
"<path/to/example.odb>"

> process
# Enter .hdf5 files name to create when provided
```

Select ranges:
```
> ranges
...
Enter the lower x you would like to use (Leave blank for negative infinity)
Enter the upper x you would like to use (Leave blank for infinity)  1.0
You entered the lower x value as -inf and the upper x value as 1.0.

Is this correct? (Y/n)? y
Enter the lower y you would like to use (Leave blank for negative infinity)  -1
Enter the upper y you would like to use (Leave blank for infinity)  25.5
You entered the lower y value as -1.0 and the upper y value as 25.5.

Is this correct? (Y/n)? y
Enter the lower z you would like to use (Leave blank for negative infinity)
Enter the upper z you would like to use (Leave blank for infinity)  2.0
You entered the lower z value as -inf and the upper z value as 2.0.

Is this correct? (Y/n)?
Enter the start time you would like to use (Leave blank for zero)
Enter the end time you would like to use (Leave blank for infinity)  1.0
You entered the start time value as 0.0 and the stop time value as 1.0.

Is this correct? (Y/n)?
Enter the lower temperature you would like to use (Leave blank for zero)  300.0
Enter the upper temperature you would like to use (Leave blank for infinity)  1727.0
You entered the lower temperature value as 300.0 and the upper temperature value as 1727.0.

Is this correct? (Y/n)? y
```

Set Plotting Criteria:
```
> views # view and select views
> interactive # toggle interactive plotting
``` 

Plot 3D:
```
> plot_3d
...
> plot_meltpool
```

Plot 2D:
```
> plot_node
...
> plot_val_v_time
...
```

View the current settings of the CLI Plotter
```
> status
```