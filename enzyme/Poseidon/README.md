# Poseidon

Poseidon is a profile-guided numerical optimizer built as a pass plugin next to
Enzyme. It profiles a program's floating-point instructions, with Enzyme
differentiating the probes, and rewrites the program to run as fast as a user
accuracy target allows, on CPUs, NVIDIA GPUs (CUDA) and AMD GPUs (HIP).

Project website: <https://prontolab.github.io/numerics/>

## Build

Poseidon needs LLVM 20 or newer and MPFR. It builds Herbie from source, which
takes Rust and Racket, unless `-DPOSEIDON_HERBIE_BINARY=<herbie>` names a
prebuilt one.

```sh
cmake -G Ninja -S enzyme -B build \
    -DLLVM_DIR=<llvm-build>/lib/cmake/llvm \
    -DENZYME_POSEIDON=ON
ninja -C build
ninja -C build check-poseidon
```

## License

Apache License v2.0 with LLVM Exceptions, the repository's `LICENSE`.
