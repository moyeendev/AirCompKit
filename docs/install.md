# Installation

AirCompKit needs Python 3.10 or newer. The core depends only on NumPy and PyYAML.

```bash
git clone https://github.com/moyeendev/AirCompKit.git
cd AirCompKit
python -m pip install -e .            # core
python -m pip install -e ".[plot]"    # + matplotlib, for figures from `reproduce`
python -m pip install -e ".[dev]"     # + pytest, ruff, mypy
python -m pip install -e ".[docs]"    # + MkDocs and Jupyter, to build these pages
```

Check the installation:

```bash
python -m aircompkit --version
python -m aircompkit list
pytest -q
```

!!! note "Blocked launchers"
    Some managed Windows machines block the `aircompkit.exe` launcher that pip generates.
    `python -m aircompkit ...` works everywhere and is used throughout these pages.

## Datasets

AirCompKit never downloads data. The `synthetic` dataset is generated from a seed and needs
nothing. For MNIST or Fashion-MNIST, put either

- `mnist.npz` / `fashion_mnist.npz` with the Keras keys `x_train, y_train, x_test, y_test`, or
- the four original IDX files (`train-images-idx3-ubyte[.gz]`, ...) in `<root>/mnist/`

in a folder, and pass `data_root=` in Python, `--data-root` on the command line, or set the
`AIRCOMPKIT_DATA` environment variable. `~/.aircompkit/data` is searched last.

## Building the documentation

```bash
python docs/execute_notebooks.py --write   # re-run the tutorials and store their outputs
mkdocs serve                               # http://127.0.0.1:8000
```
