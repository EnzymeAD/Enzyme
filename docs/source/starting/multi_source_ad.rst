.. _multi-source-ad:

Link-time AD of Multi-Source Programs
=====================================
Enzyme needs access to the IR of any function that is active and being differentiated.
A function might use code from another compilation unit (a file for C/C++, a crate for Rust, ...),
so we want to present a few solutions here.




.. _merging-with-llvm-link:

Using llvm-link
--------------------
This approach is currently used by the Rust frontend. Using the "emit=llvm-bc" flag we can
ask the rust compiler to emit each crate as a single bitcode file. Using llvm-link we can merge 
those into a single file, which assures that enzyme has access to all the code it might differentiate.
It is possible to use this approach with other compilers like clang manually, although simpler 
alternatives exist. 




.. _merging-with-lto:

Using LTO 
----------
This approach is often used for C/C++ code. Enabling link-time-optimizations will force clang to 
embed the llvm bitcode into compilation artefacts. By using lld as a linker, we are able to pass 
enzyme as a linker plugin, which will give it access to all the required code.




.. _dynamic-loading:

Dynamic loading
---------------
This approach is used by the Julia frontend. 
