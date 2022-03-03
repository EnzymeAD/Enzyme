.. Enzyme documentation master file. Always need to contain the
   root `toctree` directive.

:github_url: https://github.com/EnzymeAD/Enzyme

Enzyme documentation
===================================

Enzyme is a LLVM compiler-plugin for automatic differentation on the level of the LLVM intermediate representation.
Currently, Enzyme is able to compute reverse-mode derivatives (backprop), and forward-mode derivative. Taking in
arbitrary existing code in a language compiling to the LLVM IR, such as C/C++, Julia, FORTRAN, Swift, Rust, Python,
Haskell, etc. it computes the derivative and gradient of that function. This allows developers to use Enzyme to
automatically create gradients of their source code without much additional work, other than plugging Enzyme into
their compilation process. Utilizing LLVM's optimization pipeline, Enzyme is able to access and modify the program
at a variety of levels hence providing an unprecedented amount of flexibility culminating in optimizations not possible
in other automatic differentiation frameworks.


.. toctree::
    :glob:
    :maxdepth: 1
    :caption: Getting Started

    starting/*

.. toctree::
    :glob:
    :maxdepth: 1
    :caption: Language Bindings

    C/C++ <https://enzyme.mit.edu/cpp/>
    Fortran <https://enzyme.mit.edu/fortran/>
    Julia <https://enzyme.mit.edu/julia/>
    Rust <https://enzyme.mit.edu/rust/>
    Swift <https://enzyme.mit.edu/swift/>

.. toctree::
    :glob:
    :maxdepth: 1
    :caption: User Manual

    calling_convention
    troubleshooting_and_tips
    faq
    glossary
    examples
    talks_and_tutorials

.. toctree::
    :glob:
    :maxdepth: 1
    :caption: Advanced Features

    advanced/*

.. toctree::
    :glob:
    :maxdepth: 2
    :caption: API Documentation
    

.. toctree::
   :maxdepth: 2

   about
   api/library_root