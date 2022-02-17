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
in other automatic differenation frameworks.

Features and approaches described in this documentation are classified by their maturity status:

  *Stable:* Features which are at the core of Enzyme and are stably supported.

  *Beta:* Features which have seen active use, but yet remain in active development to e.g.
          achieve better feature coverage.
  
  *Alpha:* Features which are in the early development cycle, and are neither expressly stable,
           nor have any guarantee of API-continuity.

.. toctree::
    :glob:
    :maxdepth: 1
    :caption: Getting Started

    starting/*

.. toctree::
    :glob:
    :maxdepth: 1
    :caption: Language Bindings

    Julia <https://enzyme.mit.edu/julia/>
    Rust <https://enzyme.mit.edu/rust/>

.. toctree::
    :glob:
    :maxdepth: 1
    :caption: Advanced Features

    advanced_options