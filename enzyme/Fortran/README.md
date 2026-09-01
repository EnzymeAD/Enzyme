# Fortran bindings for Enzyme

Source files in this subdirectory provides Fortran bindings for Enzyme, as
detailed in the following.

## Note on compilers

Before providing details on the Fortran bindings, it is worth noting that Enzyme
only supports the `2023.0.0` and `2023.2.4` versions of the Intel
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

## Function hooks for differentiation

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

## Function hook for batching

We do not currently provide bindings for the `__enzyme_batch` function hook
because it requires `enzyme_width` to be passed-by-value as an integer and this
is not supported by the implicit interfacing approach used for the other
function hooks. As such, you will need to write your own explicit `interface`
block to handle the batching. See the Fortran
[batching test ](../test/Fortran/BatchMode/square_with_explicit_interface.f90)
for an example.

> [!NOTE]
> In C, the batched output is provided using a simple `struct`. The required
> syntax is different in Fortran - you should instead provide each entry of the
> output batch individually.

> [!NOTE]
> You will likely find that batching works more straightforwardly with
> subroutines than with Fortran functions.


## Function-like hooks

The `enzyme_function_like` hook tells Enzyme to differentiate a function as if
it were a known mathematical function. For example, Enzyme can use the
derivative of `log1p` for `log1p_like_function`, regardless of its
implementation.

### Call-style registration

The call-style interface follows the same pattern as `enzyme_autodiff`:

```fortran
use enzyme, only: enzyme_function_like, enzyme_log1p

call enzyme_function_like(log1p_like_function, enzyme_log1p)
```

`enzyme_log1p` supplies the symbolic function name `log1p`; its value is not
used. Functions passed to `enzyme_function_like` must have an LLVM-level
signature compatible with the selected mathematical function. Scalar arguments
must use the `value` attribute so that Flang lowers them as LLVM values rather
than using Fortran's usual by-reference calling convention. This binding is
currently supported with Flang.

When running Enzyme separately with `opt`, `preserve-nvvm` must process the
`enzyme_function_like` hook before differentiation:

```console
$ opt -load-pass-plugin=/path/to/LLVMEnzyme-21.so \
    -passes='preserve-nvvm,enzyme,preserve-nvvm-end' input.bc -o output.bc
```

When using this separate `opt` workflow, compile the Fortran source to LLVM
 with `-O0`. Otherwise, Flang may inline calls to the function before
`preserve-nvvm` processes the `enzyme_function_like` hook. The `FlangEnzyme`
compiler plugin runs `preserve-nvvm` at the start of Flang's LLVM optimization
pipeline and does not require this separate `opt` step.

Additional symbolic function names can be declared in user code. The `bind(C)`
name must use the `enzyme_math_` prefix followed by a function name recognized
by Enzyme:

```fortran
module enzyme_math_names
  use iso_c_binding, only: c_int
  implicit none

  integer(c_int), bind(C, name="enzyme_math_sin") :: enzyme_sin
end module enzyme_math_names
```

This makes the call site simple, but every symbolic name needs a corresponding
`enzyme_math_*` binding, either in the `enzyme` module or in user code.

### Procedure-pointer registration

Alternatively, a statically initialized procedure pointer can register the
same relationship without a hook call or symbolic-name binding:

```fortran
module function_like_example
  implicit none

  procedure(log1p_like_function), pointer, private :: &
    fn__enzyme_function_like__log1p => log1p_like_function

contains

  function log1p_like_function(x) result(y)
    double precision, value :: x
    double precision :: y

    y = 2.0d0 * x
  end function log1p_like_function
end module function_like_example
```

Here, `procedure(log1p_like_function)` gives the pointer the target's interface,
and `=> log1p_like_function` initializes it with the target. PreserveNVVM reads
the mathematical name after the exact `__enzyme_function_like__` delimiter, so
this example registers the target as `log1p`. The prefix before the delimiter
can be any valid name but must be unique in its scope. `private` is optional; it
keeps the registration marker out of the module's public API.
