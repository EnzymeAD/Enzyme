.. _installation:

Installation Instructions
=========================

.. _package-manager:

Package Manager
---------------

The easiest way to install Enzyme is by using a package manager to handle the dependency
on LLVM for you. A number of package managers contain instructions for the installation
of Enzyme 

Homebrew
^^^^^^^^

To install using `Homebrew <https://brew.sh/>`_, run

.. code-block:: bash

    brew install enzyme

Currently, Homebrew has a pre-built binary (bottle) on macOS, but will build from source
on Linux.

Spack
^^^^^

To install using `Spack <https://spack.io/>`_, run

.. code-block:: bash

    spack install enzyme

Which will build you the entire required compiler toolchain together with Enzyme itself.

Conda
^^^^^

To install using `Conda <https://docs.conda.io/en/latest/>`_, run

.. code-block:: bash

    conda create -n test-enzyme -c conda-forge -c ivanyashchuk numba llvmdev=11.* libenzyme

The conda installation is still in the process of being merged into conda-forge, and hence requires
the pulling from a custom conda channel.

The rest of these instructions will focus on building Enzyme from source.

.. _building-from-source:

Building from Source
--------------------

Enzyme is a plugin for LLVM, and hence needs an existing build of LLVM to function. Enzyme itself is designed
to work with a wide range of LLVM version, and is currently tests against LLVM 7, 8, 9, 10, 11, 12, 13, and mainline.
LLVM's plugin infrastructure can sometimes be flakey, or not be build by default. If loading Enzyme into an existing
LLVM installation, such as by your distribution's package manager, or a shipped LLVM distro such as the ones by
Intel, AMD, or IBM, we recommend building LLVM from source.

C/C++
^^^^^


Fortran
^^^^^^^


Rust
^^^^


Verification of Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^



.. _container-development:

Development with Containers
---------------------------