# MR Pypeline
This is a multimedia retrieval pipeline, written in python, for a university course.

## Usage
The program can be called by using the command `python main.py` on the command line, when in the project's root directory.

The program also has different modes, and mode specific arguments as well as two common arguments.
Please observe the following layout for calling the program:
```
python main.py [common arguments] [mode] [mode arguments]
```
The modes and arguments are explained in the following subsections.

### Common arguments

#### Help
The first common argument is the Help argument.
It is called using the `--help` flag, or its shorthand `-h`.
When present, the program will print a short overview of the available arguments and exit without doing anything further.

#### Debug
The other common argument is the Debug argument.
It is called using the ``--debug` flag, or its shorthand `-d`.
It sets the the output verbosity of the program on the command line.
It does so in three ways:
* When omitted, the output level is set to `INFO`
* When included without an argument, the output level is set to `DEBUG` (this only works when followed by another argument flag)
* When included with an argument, the output level is set to the argument

The output levels available are, in order of least to most severe:
* `NOTSET`
* `DEBUG`
* `INFO`
* `WARNING`
* `ERROR`
* `CRITICAL`
Please note that if the output is set to a certain level, only messages with a lower severity level will be muted.
So, when setting the output level to WARNING, messages with level WARNING, ERROR, or CRITICAL will be printed to the command line.

### Program modes
The program consists of two modes: view and preprocess.
The view mode opens a user interface that allows users to view `.off` and `.ply` files as 3D meshes, wireframes, or point clouds.
The preprocess mode preprocesses `.off` and `.ply` files for multimedia retrieval.

The mode is selected by specifying it on the command line, like so:
```
python main.py [view OR preprocess]
```
Please note that when, omitted the program automatically selects view mode.

#### View mode
The view mode is the default mode for the program.
It is selected by either omitting the mode entirely or by calling:
```python main.py [common arguments] view```
The view mode takes no additional arguments.

#### Preprocess Mode
The preprocess mode is selected by calling:
```python main.py [common arguments] preprocess [input] [output]```
The preprocess mode takes two positional arguments: input and output.
Both the argument should be paths. Input can either be a file or a folder, but output has to be a folder.
If the output path does point to an existing location, the program will attempt to create it.
If the input path is an `.off` or `.ply` file, the file will be preprocessed and placed at the output location.
The output path dictates where the preprocessed files go.
If the input path is a folder, the program will read search for any `.off` and `.ply` files in that folder and any subfolders.
Please note that if the input path is a folder, the output path cannot be a file, because the program expects multiple files to be found.
