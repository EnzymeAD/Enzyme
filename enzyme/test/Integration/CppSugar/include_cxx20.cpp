// The plugin defines its version macros after the preprocessor is initialized.
// Check that doing so keeps the -include inputs, which in C++20 mode clang
// holds outside the predefines buffer (CUDA force-includes its runtime wrapper
// this way).

// RUN: echo "#define FORCE_INCLUDED" > %t.h
// RUN: %clang++ -std=c++17 -fsyntax-only -include %t.h %s %loadClangEnzyme
// RUN: %clang++ -std=c++20 -fsyntax-only -include %t.h %s %loadClangEnzyme

#ifndef FORCE_INCLUDED
#error "-include input was dropped"
#endif

#if !defined(ENZYME_VERSION_MAJOR) || !defined(ENZYME_VERSION_MINOR) ||        \
    !defined(ENZYME_VERSION_PATCH)
#error "Enzyme version macros are not defined"
#endif
