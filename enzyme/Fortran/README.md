# Fortran bindings for Enzyme

Source files in this subdirectory provides Fortran bindings for Enzyme, as
detailed in the following.

## Note on compilers

Before providing details on the Fortran bindings, it is worth noting that Enzyme
only supports the `2023.0.0`, `2023.1.0`, and `2023.2.4` versions of the Intel
IFX Fortran compiler. We strongly recommend using the
[Flang](https://flang.llvm.org) compiler, which is available as part of the
[LLVM project](https://github.com/llvm/llvm-project).

## Running Enzyme from flang

Configuring Enzyme with `-DENZYME_FLANG=ON` builds `FlangEnzyme-<LLVM version>`, a
pass plugin that flang can load with `-fpass-plugin`. Enzyme then runs as part of
the flang optimization pipeline, so a single command differentiates and compiles:

```console
$ flang -fpass-plugin=/path/to/FlangEnzyme-21.so -I /path/to/enzyme/modules program.f90 -o program
```

The `-I` flag points at the directory holding the `enzyme.mod` module file, which is
built by `-DENZYME_FORTRAN=ON` (see the sections below).

Without the plugin the derivative has to be produced out of line, by emitting LLVM IR
from flang and running the Enzyme pass over it with `opt`:

```console
$ flang -flto -c -I /path/to/enzyme/modules program.f90 -o program.bc
$ opt -load-pass-plugin=/path/to/LLVMEnzyme-21.so -passes=enzyme program.bc -o program-enzyme.bc
$ flang -flto program-enzyme.bc -o program
```

Both routes are exercised by the tests in `enzyme/test/Fortran`. The plugin route is
flang-only; with ifx use the `opt` pipeline above.

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
