# Build Instructions for the Docs

To build the docs you need to start by creating a virtual environment to house the theme and build dependencies

```bash
python -m venv .docs-env
```

Afterwards we need to activate it, and install the dependencies into the virtual env

```bash
source .docs-env/bin/activate
pip install -r requirements.txt
```

Upon successful setup of the virtual env, one can now create the webpage using the Makefile, i.e.

```bash
make html
```