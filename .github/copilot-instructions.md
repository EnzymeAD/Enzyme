# Enzyme GitHub Copilot Instructions

## Project Overview

Enzyme is a high-performance automatic differentiation (AD) plugin for LLVM and MLIR. It performs AD of statically analyzable LLVM IR and MLIR code through compiler transformations, achieving state-of-the-art performance by working on optimized code.

### Key Concepts

- **Automatic Differentiation**: Enzyme transforms functions to compute derivatives automatically
- **LLVM Plugin**: Works as an LLVM transformation pass on intermediate representation (IR)
- **Activity Analysis**: Determines which values are relevant for differentiation
- **Type Analysis**: Tracks and propagates type information through the IR
- **Gradient Utilities**: Core infrastructure for computing forward and reverse-mode derivatives

## Repository Structure

```
enzyme/
├── Enzyme/          # Core AD transformation logic
│   ├── ActivityAnalysis.cpp/h      # Determines active/inactive values
│   ├── TypeAnalysis/               # Type tracking and propagation
│   ├── EnzymeLogic.cpp/h           # Main AD transformation logic
│   ├── GradientUtils.cpp/h         # Gradient computation utilities
│   ├── AdjointGenerator.h          # Reverse-mode AD code generation
│   ├── CacheUtility.cpp/h          # Caching mechanism for AD
│   ├── FunctionUtils.cpp/h         # Function manipulation utilities
│   ├── MLIR/                       # MLIR dialect and passes
│   └── Clang/                      # Clang plugin integration
├── BCLoad/          # Bitcode loader for runtime library support
├── test/            # Test suite using LLVM lit
│   ├── Enzyme/                     # Core Enzyme tests
│   ├── ActivityAnalysis/           # Activity analysis tests
│   ├── TypeAnalysis/               # Type analysis tests
│   ├── Integration/                # Integration tests (ReverseMode, ForwardMode, etc.)
│   └── MLIR/                       # MLIR-specific tests
├── tools/           # Additional tools
│   └── enzyme-tblgen/              # TableGen tool for derivative rules
└── cmake/           # CMake configuration files
```

## Building the Project

### Prerequisites

- CMake 3.13+
- LLVM development libraries (check `.github/workflows/enzyme-ci.yml` for supported versions)
- Ninja or Make build system
- Python 3 with lit (for testing)

### Standard Build

```bash
cd enzyme
mkdir build && cd build
cmake -G Ninja .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm \
  -DLLVM_EXTERNAL_LIT=$(which lit)
ninja
```

### Build Types

- `Release`: Optimized build (default)
- `Debug`: Debug symbols, no optimization
- `RelWithDebInfo`: Optimized with debug symbols

### Build Options

- `ENZYME_ENABLE_PLUGINS`: Enable Clang/LLD/Opt plugins (default: ON)
- `ENZYME_BC_LOADER`: Enable bitcode loader (default: ON)
- `ENZYME_CLANG`: Build enzyme clang plugin (default: ON)
- `ENZYME_MLIR`: Build enzyme MLIR plugin (default: OFF)
- `ENZYME_STATIC_LIB`: Build static library (default: OFF)

## Testing

Enzyme uses LLVM's lit test framework. Tests are written as LLVM IR (.ll), C/C++ (.c, .cpp), MLIR (.mlir), or Fortran (.f90) files with RUN directives.

### Running Tests

```bash
cd build
make check-enzyme           # Run all Enzyme tests
make check-typeanalysis     # Run Type Analysis tests
make check-activityanalysis # Run Activity Analysis tests
```

### Test File Structure

Tests use lit directives:
```llvm
; RUN: %opt < %s %loadEnzyme -enzyme -S | FileCheck %s
```

Tests should verify both correctness and performance properties where applicable.

## Code Style and Formatting

### C++ Style

- Follow **LLVM Coding Standards**: https://llvm.org/docs/CodingStandards.html
- Use `clang-format` with LLVM style (configured in `enzyme/Enzyme/.clang-format`)
- No RTTI (`-fno-rtti` is required)
- C++17 standard

### Formatting Commands

```bash
# Format changed files in last commit
git clang-format HEAD~1

# Format specific file
clang-format -i path/to/file.cpp
```

### Code Organization

- Header guards: Use `#ifndef HEADER_NAME_H` format
- Include order: Local headers, then LLVM headers, then system headers
- Prefer forward declarations to reduce header dependencies
- Use anonymous namespaces for file-local helpers

## Development Guidelines

### Adding New Features

1. **Understand the IR**: Enzyme operates on LLVM IR and/or MLIR. Familiarize yourself with the IR being transformed.
2. **Activity Analysis First**: Ensure values are properly marked as active/inactive
3. **Type Analysis**: Use Type Analysis to understand data flow and pointer types
4. **Test-Driven Development**: Write tests before implementing features
5. **Check Existing Derivatives**: Look at `CallDerivatives.cpp` and `BlasDerivatives.td` for examples

### Common Patterns

#### Pass Registration
New transformation passes should use `AnalysisInfoMixin` pattern (even though they modify IR):
```cpp
class MyPass final : public AnalysisInfoMixin<MyPass> {
  friend struct AnalysisInfoMixin<MyPass>;
private:
  static AnalysisKey Key;
public:
  using Result = PreservedAnalyses;
  Result run(Function &F, FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }
};
```

#### Type Analysis
- When GEP has null pointer operand, don't propagate pointer type to result
- Use `TypeTree` for tracking type information
- Always validate type propagation results

#### Activity Analysis
- Values are "active" if they depend on differentiated inputs
- Use `isConstantValue` and `isConstantInstruction` carefully
- Consider both data and control flow dependencies

### Memory Safety

- Avoid unnecessary allocations in hot paths
- Be careful with pointer ownership and lifetime
- Use LLVM's memory management patterns (e.g., `SmallVector`, `DenseMap`)
- Never assume pointer validity without checking

### Error Handling

- **Prefer `EmitFailure`** over compile-time crashes when possible for better error diagnostics:
  ```cpp
  EmitFailure("RemarkName", Loc, CodeRegion, "Error message: ", value);
  ```
- **Use `CustomErrorHandler`** to provide user-customizable error handling (especially useful for language bindings)
- Use LLVM's error reporting: `llvm::errs()`, `dbgs()`, assertions
- Provide meaningful diagnostic messages
- Use `llvm_unreachable()` for impossible cases

## Pull Request Guidelines

### Before Submitting

1. **Run clang-format** on all changed code
2. **Add tests** that cover your changes (required)
3. **Run relevant test suites** to ensure no regressions
4. **Keep changes focused**: One logical change per PR
5. **Update documentation** if adding new features or changing behavior

### PR Requirements

- Include a small unit test demonstrating the change
- Conform to LLVM Coding Standards
- No unrelated changes
- Be an isolated change (split independent changes into separate PRs)

### Commit Messages

- First line: Brief summary (50 chars or less)
- Blank line
- Detailed description if needed
- Reference issues: "Fixes #123" or "Closes #456"

## Debugging Tips

### Using Compiler Explorer

The easiest way to explore and debug Enzyme: https://enzyme.mit.edu/explorer

### Debug Builds

Build with `CMAKE_BUILD_TYPE=Debug` for better debugging:
- Full debug symbols
- No optimization (easier to follow execution)
- Additional assertions enabled

### Useful LLVM Flags

```bash
opt -load-pass-plugin=path/to/LLVMEnzyme-<version>.so \
    -enzyme -enzyme-print -debug -debug-only=enzyme \
    input.ll -S -o output.ll
```

### Print Debugging

```cpp
llvm::errs() << "Debug message: " << value << "\n";
dbgs() << "Debug-only message\n";  // Only with -debug flag
```

## Common Issues

### Build Failures

- **Missing LLVM**: Ensure `LLVM_DIR` points to the correct LLVM installation
- **Version support**: Enzyme supports LLVM 15 through mainline. However, lit tests currently only pass on LLVM 15 and 16, which support both typed and opaque pointers. LLVM 17+ only supports opaque pointers. Help migrating lit tests to work with both typed and opaque pointers (depending on LLVM version) is appreciated.
- **lit not found**: Install with `pip install lit` and set `LLVM_EXTERNAL_LIT`

### Test Failures

- Run tests locally before submitting PRs
- Check if tests are flaky or environment-dependent
- Ensure test expectations match the actual output format

### Type Analysis Issues

- Null pointer GEP: Don't propagate types from null pointers
- Conflicting types: Review updateAnalysis calls carefully
- Missing type information: Check if analysis is run before usage

## Additional Resources

- **Website**: https://enzyme.mit.edu
- **Documentation**: https://enzyme.mit.edu/Installation/
- **Mailing List**: https://groups.google.com/d/forum/enzyme-dev
- **LLVM Discourse**: https://discourse.llvm.org/c/projects-that-want-to-become-official-llvm-projects/enzyme/45
- **Discord**: #enzyme channel on https://discord.gg/xS7Z362
- **Julia Integration**: https://github.com/EnzymeAD/Enzyme.jl
- **Rust Integration**: https://github.com/EnzymeAD/rust

## Language Integrations

Enzyme can be integrated with any language that compiles to LLVM IR:
- **C/C++**: Via Clang plugin
- **Fortran**: Via Flang support
- **Julia**: Via Enzyme.jl package
- **Rust**: Via rust-enzyme bindings

When adding language-specific features, ensure they work correctly with the core AD transformation.
