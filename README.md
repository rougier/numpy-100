# 100 NumPy Exercises

[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org:/repo/rougier/numpy-100/notebooks/100%20Numpy%20exercises.ipynb)
[![DOI](https://zenodo.org/badge/10173/rougier/numpy-100.svg)](https://zenodo.org/badge/latestdoi/10173/rougier/numpy-100)

## Table of Contents

- [About this repository](#about-this-repository)
- [How to Use](#how-to-use)
  - [Using Binder (online)](#using-binder-online)
  - [Using a local installation](#using-a-local-installation)
- [How to Contribute](#how-to-contribute)
- [Credits](#credits)
- [License](#license)

## About this repository

This repository offers a collection of 100 NumPy exercises. They are gathered from the NumPy mailing list, Stack Overflow, and the NumPy documentation. Some have also been created by the original author to reach the 100-exercise limit.

The goal of this collection is to provide a quick reference for both new and experienced users and to offer a set of exercises for those who teach. For more extensive exercises, please read [From Python to NumPy](http://www.labri.fr/perso/nrougier/from-python-to-numpy/).

The exercises are available in several formats:

- [Jupyter Notebook](100_Numpy_exercises.ipynb)
- [Markdown file](100_Numpy_exercises.md)
- [Markdown file with hints](100_Numpy_exercises_with_hints.md)
- [Markdown file with solutions](100_Numpy_exercises_with_solutions.md)
- [Markdown file with hints and solutions](100_Numpy_exercises_with_hints_with_solutions.md)

## How to Use

You can run the exercises in your browser using Binder, or you can run them on your local machine.

### Using Binder (online)

Click the "Launch Binder" button below to open the Jupyter Notebook with the exercises in your browser:

[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org:/repo/rougier/numpy-100/notebooks/100%20Numpy%20exercises.ipynb)

### Using a local installation

To run the exercises on your local machine, you need to have Python and the following libraries installed:

- NumPy
- Pandas
- Jupyter
- Jupyter Themes
- MdUtils

You can install them using pip:

```bash
pip install -r requirements.txt
```

Once the dependencies are installed, you can open the Jupyter Notebook:

```bash
jupyter notebook 100_Numpy_exercises.ipynb
```

## How to Contribute

Contributions are welcome! If you find an error or have a better way to solve an exercise, feel free to open an issue or a pull request.

The exercises, hints, and solutions are stored in the `source/exercises100.ktx` file. This file uses a simple key-value format, where each entry is identified by a key starting with `<`. For example:

```
<q1
Import the numpy package under the name `np` (★☆☆)

<h1
hint: import … as

<a1
import numpy as np
```

To modify the exercises, you need to:

1.  Edit the `source/exercises100.ktx` file.
2.  Run the `generators.py` script to update the Jupyter notebooks and markdown files:

    ```bash
    python generators.py
    ```

## Credits

These exercises were collected by [Nicolas P. Rougier](https://www.labri.fr/perso/nrougier/). The original repository can be found at [rougier/numpy-100](https://github.com/rougier/numpy-100).

## License

This work is licensed under the MIT license.
