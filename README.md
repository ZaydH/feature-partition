# Feature Partition Aggregation

[![docs](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/ZaydH/certified-sparse/blob/main/LICENSE)

This repository contains the source code for reproducing the paper "Feature Partition Aggregation: A Fast Certified Defense Against a Union of Sparse Adversarial Attacks".

* Authors: [Zayd Hammoudeh](https://zaydh.github.io/) and [Daniel Lowd](https://ix.cs.uoregon.edu/~lowd/)
* Link to Paper: [Arxiv](https://arxiv.org/abs/2302.11628)

## Running the Program

To run the program, enter the `src` directory and call:

`python driver.py ConfigFile`

where `ConfigFile` is one of the `yaml` configuration files in folder [`src/configs`](src/configs). 

### First Time Running the Program

The first time each configuration runs, the program automatically downloads any necessary dataset(s).  Please note that this process can be time-consuming -- in particular for the `weather` dataset.

These downloaded files are stored in a folder `.data` that is in the same directory as `driver.py`.  If the program crashes while running a configuration for the first time, we recommend deleting or moving the `.data` to allow the program to re-download and reinitialize the source data.

### Requirements

Our implementation was tested in Python&nbsp;3.7.13.  For the full requirements, see `requirements.txt` in the `src` directory.  If a different version of Python is used, some package settings in `requirements.txt` may need to change.

We recommend running our program in a [virtual environment](https://docs.python.org/3/tutorial/venv.html).  Once your virtual environment is created and *active*, run the following in the `src` directory:

```
pip install --user --upgrade pip
pip install -r requirements.txt
```

## License

[MIT](https://github.com/ZaydH/certified-sparse/blob/main/LICENSE)

## Citation

```
@misc{Hammoudeh:2023:FeaturePartition,
    author = {Hammoudeh, Zayd and
              Lowd, Daniel},
    title = {Feature Partition Aggregation: A Fast Certified Defense Against a Union of Sparse Adversarial Attacks},
    year  = {2023},
    eprint = {2302.11628},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
}
```
