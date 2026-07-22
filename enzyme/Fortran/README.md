# Fortran bindings for Enzyme

Source files in this subdirectory provides Fortran bindings for Enzyme, as
detailed in the following.

## Note on compilers

Before providing details on the Fortran bindings, it is worth noting that Enzyme
only supports the `2023.0.0`, `2023.1.0`, and `2023.2.4` versions of the Intel
IFX Fortran compiler. We strongly recommend using the
[Flang](https://flang.llvm.org) compiler, which is available as part of the
[LLVM project](https://github.com/llvm/llvm-project).

## Function hooks

We provide bindings for the `__enzyme_fwddiff` and `__enzyme_autodiff` function
hooks using implicit interfaces. Some Fortran compilers disallow procedure names
starting with an underscore so we rename the function hooks to remove the
leading double underscore.

To make use of the `enzyme_autodiff` function hook in your code, import via
```fortran
use enzyme, only: enzyme_autodiff
```
and call it as a subroutine or function as appropriate. For example, if you have
a function
```fortran
  real function square(x)
    real, intent(in) :: x
    square = x**2
  end function
```
then you can compute its derivative with reverse mode with the call
```fortran
  call enzyme_autodiff(square, x, dx)
```

Similarly for
`enzyme_fwddiff`. Thanks to the implicit interface, arbitrary signatures are
supported, with the following caveats.

> [!NOTE]
> A limitation of the implicit interfacing is that it only works for arguments
> that are passed by reference - the default in Fortran. If you want to pass any
> arguments by value using the `value` attribute then you will need to write an
> explicit interface block to the function hook yourself.

> [!WARNING]
> The implicit interfacing approach is not supported by the Intel Fortran
> compiler ifx when running without optimizations, i.e., running with `-O0`. If
> you want to use ifx with `-O0` then you will need to write an explicit
> interface block, even if you are only passing arguments by reference.

> [!WARNING]
> Differentiation with respect to procedures with assumed shape arrays is not
> currently supported when compiling with Flang. It should work with ifx,
> however.

## Activity descriptors

We provide bindings for the activity descriptors `enzyme_const`, `enzyme_dup`,
`enzyme_dupnoneed`, and `enzyme_out`, as well as the descriptors
`enzyme_scalar`, `enzyme_width`, and `enzyme_vector`. To make use of these in
your code, import via
```fortran
use enzyme, only: enzyme_const, enzyme_dup
```
and then include them in calls to function hooks as you would in C or C++. For
example, if you have a subroutine
```fortran
  subroutine my_subroutine(n, x, y)
    integer, intent(in) :: n
    real, dimension(n), intent(in) :: x
    real, dimension(n), intent(out) :: y
    ! ...
  end subroutine my_subroutine
```
then you can make use of activity descriptors like so:
```fortran
  call enzyme_autodiff(my_subroutine, enzyme_const, n, &
                       enzyme_dup, x, dx, enzyme_dup, y, dy
```

## Flang compiler plugin (experimental sketch)

`Enzyme/Flang/EnzymeFlang.cpp` is a *sketch* of a Flang frontend plugin, built
as `FlangEnzyme-<LLVM major>` when Enzyme is configured with `-DENZYME_FLANG=ON`
against a Flang installation new enough to expose the plugin API
(LLVM ≥ 20; point CMake at it with `-DFlang_DIR=<flang>/lib/cmake/flang`). When
those are unavailable, `FlangEnzyme-<LLVM major>` falls back to a symlink to the
Clang plugin, as before.

> [!NOTE]
> This is a starting point, not a full frontend integration. Flang's plugin API
> (`PluginParseTreeAction`) is a *replacement* frontend action limited to
> `flang -fc1` (and, per the Flang docs, Linux only). Unlike the Clang plugin it
> has **no** hook to inject the Enzyme code-generation pass, so the frontend
> plugin currently only *inspects* the parse tree and reports the Enzyme
> differentiation calls it finds. Actual differentiation is still performed by
> the LLVM pass plugin.

The same `FlangEnzyme-<LLVM major>` shared object therefore serves two roles:

```bash
# 1. Inspect Enzyme usage at the Flang frontend (this plugin):
flang -fc1 -load ./FlangEnzyme-<ver>.so -plugin enzyme input.f90

# 2. Actually differentiate, using the LLVM-pass-plugin role of the *same*
#    shared object (see flang/docs/FlangDriver.md, `-fpass-plugin`):
flang -O2 -fpass-plugin=./FlangEnzyme-<ver>.so input.f90 -o input
```

The `TODO(enzyme)` markers in `EnzymeFlang.cpp` outline the natural next steps:
recognizing Enzyme activity annotations at the frontend level (analogous to the
`enzyme_inactive` / `enzyme_function_like` attributes handled by the Clang
plugin), and injecting the Enzyme pass directly once Flang grows a
code-generation plugin hook.
