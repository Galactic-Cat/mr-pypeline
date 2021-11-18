# MR Pypeline
This is a multimedia retrieval pipeline, written in python, for a university course.

## Installation
1. Clone the repository.
As this repository is rather large, you may need to perform an extra step when cloning it for the first time.
Instead of cloning the repository as you normally would, follow the following steps:
```
git clone git@github.com:Galactic-Cat/mr-pypeline.git --depth 1
cd mr-pypeline
git fetch --unshallow
```
2. Setup the python environment.
The environment was generated using [Conda](https://docs.conda.io/en/latest/miniconda.html), a multi-environment package management tool for Python.
With Conda installed (and accessible on the command line), run this command from the root of the directory:
```
conda env create -f environment.yml
```
3. This will create the "mr" environment including all packages you'll need to run the program.
Activate the environment by using:
```
conda activate mr
```
4. Now you can run the program like you would any other python script.

## Usage
The program can be called by using the command `python main.py` on the command line, when in the project's root directory.

> **Please note** that the repository already comes with a fully processed dataset in the `/output` folder.
> Generating this dataset takes a long time as the distribution features observe 100.000 samples per feature per shape.

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
python main.py [view/preprocess/collect]
```
Please note that when, omitted the program automatically selects view mode.

#### View mode
The view mode is the default mode for the program.
It is selected by either omitting the mode entirely or by calling:
```
python main.py [common arguments] view
```
The view mode takes no additional arguments.

#### Preprocess Mode
The preprocess mode is selected by calling:
```
python main.py [common arguments] preprocess <input> <output>
```
The preprocess mode takes two positional arguments: input and output.
Both the argument should be paths. Input can either be a file or a folder, but output has to be a folder.
If the output path does point to an existing location, the program will attempt to create it.
If the input path is an `.off` or `.ply` file, the file will be preprocessed and placed at the output location.
The output path dictates where the preprocessed files go.
If the input path is a folder, the program will read search for any `.off` and `.ply` files in that folder and any subfolders.
Please note that if the input path is a folder, the output path cannot be a file, because the program expects multiple files to be found.

#### Data Collection Mode
Data collection mode is called by:
```
python main.py [common arguments] collect <input> <output>
```
Here the input is the database of the preprocessed files and the output is a path to print feature graphs.
Which graphs are printed and how is done programmatically, and can't really be changed without editing the code itself.
This mode is mainly used for generating the figures in our report.