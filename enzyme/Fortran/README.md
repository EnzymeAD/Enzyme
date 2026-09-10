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
                       enzyme_dup, x, dx, enzyme_dup, y, dy)
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
derivative of `log1p` for `double_value`, regardless of its
implementation. The examples below deliberately compute `2*x` while requesting
the derivative of `log1p`: at `x = 2`, Enzyme returns `1/3` instead of `2`. This
illustrates a derivative override; the two functions are not mathematically
equivalent.

### Choose a registration form

| Function location and interface | When to use each form |
|---|---|
| Module function | Use a pointer declaration before the module's `contains` to keep registration with the function. The compiler supplies its explicit interface. A registration call in executable code also works. |
| External function with an explicit interface | Put the pointer declaration after the interface block in the declaration section. A registration call in executable code also works. |
| Internal function | Use call registration only if Flang supplies a direct function reference. Access to variables from the containing program or procedure can prevent registration. See the restriction below. The current pointer mechanism cannot register an internal function. |
| External function with an implicit interface | Use `procedure(real)` for an ordinary pointer to a function with a real result. For scalar math registration, provide the required value arguments and an explicit interface. |

### Call-style registration

The call-style interface follows the same pattern as `enzyme_autodiff`:

```fortran
use enzyme, only: enzyme_function_like, enzyme_log1p

call enzyme_function_like(double_value, enzyme_log1p)
```

Put the registration call in executable code, after declarations. For an
internal function, you must use this form instead of an initialized procedure
pointer. An internal function follows `contains` inside a program or another
procedure. The compiler supplies its explicit interface.

The [call-style test](../test/Fortran/ReverseMode/function_like.f90) shows this
placement. Its registration call is in the main program. Its target function,
`double_value`, is inside that program, after `contains`.

> [!WARNING]
> Call registration does not support all internal functions. An internal
> function can access variables from its containing program or procedure.
> Fortran calls this access **host association**. For example, `double_value`
> could calculate `factor * x`, where `factor` is a variable in the containing
> procedure.
>
> Flang can then generate an adapter that gives the function access to those
> variables. The current registration code requires a direct function reference.
> It cannot process this adapter, and compilation can fail with
> `First argument of enzyme_function_like must be a constant function`.
>
> The example above uses only the argument `x` and does not need this adapter.


Here `enzyme_log1p` supplies the symbolic function name `log1p`; its value is not
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

> [!WARNING]
> When using this separate `opt` workflow, compile the Fortran source to LLVM
> with `-O0`. Otherwise, Flang may inline calls to the function before
> `preserve-nvvm` processes the `enzyme_function_like` hook.

The `FlangEnzyme`
compiler plugin runs `preserve-nvvm` at the start of Flang's LLVM optimization
pipeline and does not require this separate `opt` step.

The `enzyme` module exports these symbolic names. Import the required names
with `use enzyme, only: ...`.

| Function group | Bindings |
|---|---|
| Trigonometric functions | `enzyme_sin`, `enzyme_cos`, `enzyme_tan`, `enzyme_asin`, `enzyme_acos`, `enzyme_atan`, `enzyme_atan2` |
| Exponential functions | `enzyme_exp`, `enzyme_exp2`, `enzyme_exp10`, `enzyme_expm1` |
| Logarithms | `enzyme_log`, `enzyme_log2`, `enzyme_log10`, `enzyme_log1p` |
| Inverse hyperbolic functions | `enzyme_acosh`, `enzyme_asinh`, `enzyme_atanh` |
| Roots and powers | `enzyme_sqrt`, `enzyme_cbrt`, `enzyme_hypot`, `enzyme_pow` |
| Error functions | `enzyme_erf`, `enzyme_erfc` |

For example, use `enzyme_sin` to register a function with the `sin` rule:

```fortran
use enzyme, only: enzyme_function_like, enzyme_sin

call enzyme_function_like(my_sin, enzyme_sin)
```

You can declare other symbolic names in user code. Use the `enzyme_math_`
prefix followed by a function name that Enzyme supports:

```fortran
module enzyme_math_names
  use iso_c_binding, only: c_int
  implicit none

  integer(c_int), bind(C, name="enzyme_math_fmin") :: enzyme_fmin
end module enzyme_math_names
```

Each symbolic name needs an `enzyme_math_*` binding in the `enzyme` module
or in user code.

### Procedure-pointer registration

Alternatively, a statically initialized procedure pointer can register the
same relationship without a hook call or symbolic-name binding. Enzyme reads
and removes the registration marker at compile time. Do not call through the
registration pointer. Enzyme replaces remaining references to the marker with
null pointers. Call the target function directly, for example, `double_value(x)`.
Use the same FlangEnzyme plugin or separate `opt` pipeline described above for
call-style registration.

#### Register a module function

Place the declaration before the module's `contains` statement. Unlike a
`call`, a procedure-pointer declaration is allowed in this part of a module.

```fortran
module function_like_example
  implicit none

  procedure(double_value), pointer, private :: &
    fn__enzyme_function_like__log1p => double_value

contains

  function double_value(x) result(y)
    real, value :: x
    real :: y

    y = 2.0 * x
  end function double_value

  function test(x) result(y)
    real, intent(in) :: x
    real :: y

    y = double_value(x)
  end function test

end module function_like_example

program main
  use enzyme, only: enzyme_autodiff
  use function_like_example, only: test
  implicit none
  real :: x, dx

  x = 2.0
  dx = 0.0
  call enzyme_autodiff(test, x, dx)
  write(*,"(f6.4)") dx ! Prints 0.3333
end program main
```

Here, `procedure(double_value)` gives the pointer the target's interface,
and `=> double_value` initializes it with the target. PreserveNVVM reads
the mathematical name after the exact `__enzyme_function_like__` delimiter, so
this example registers the target as `log1p`. The prefix before the delimiter
can be any valid name but must be unique in its scope. `private` is optional
in a module; it keeps the registration marker out of the module's public API.

The `test` wrapper takes its argument by reference for the `enzyme_autodiff`
binding, while `double_value` takes its argument by value to match the scalar
`log1p` rule.

#### Register an external function with an explicit interface

An external function is defined outside any program, module, or other procedure.
It can be in the same source file as its caller.

Use an explicit interface when the function has a `value` argument, as required
by the scalar math rules shown here. Put the pointer declaration after the
interface block, before executable statements. Omit `private` outside a module.

In this example, `double_value` follows `end program main`. Its interface block
describes its value argument and result.

```fortran
program main
  use enzyme, only: enzyme_autodiff
  implicit none

  interface
    function double_value(x) result(y)
      real, value :: x
      real :: y
    end function double_value
  end interface

  procedure(double_value), pointer :: &
    fn__enzyme_function_like__log1p => double_value

  real :: x, dx

  x = 2.0
  dx = 0.0
  call enzyme_autodiff(test, x, dx)
  write(*,"(f6.4)") dx ! Prints 0.3333

contains

  function test(x) result(y)
    real, intent(in) :: x
    real :: y

    y = double_value(x)
  end function test

end program main

function double_value(x) result(y)
  implicit none
  real, value :: x
  real :: y

  y = 2.0 * x
end function double_value
```

Keep the explicit interface consistent with the external function definition.
The internal `test` wrapper is allowed here because the initialized pointer
targets the external `double_value`, not `test`.

#### External function with an implicit interface

A simple external function with scalar arguments passed by reference can use
an implicit interface. Use this form for ordinary Fortran pointer calls when
the function has no features that require an explicit interface.

```fortran
program main
  implicit none
  real, external :: double_value
  procedure(real), pointer :: p => double_value

  print *, p(3.0) ! Prints 6
end program main

real function double_value(x)
  implicit none
  real, intent(in) :: x

  double_value = 2 * x
end function double_value
```

Here, `procedure(real)` specifies a real result but does not describe the
arguments. The pointer therefore has an implicit interface. Use
`procedure(double_value)` only when an explicit interface for `double_value`
is available.

This example shows an ordinary pointer call. It does not register a math rule.
Its by-reference argument does not match the scalar `log1p` interface. To register
this function as `log1p`, give its argument the `value` attribute. Then provide
the explicit interface and registration declaration from the preceding example.
Changing the argument to `value` changes the calling convention. Update the
interfaces at all call sites and recompile the callers.

#### Register a module function from a subroutine

The declaration can instead appear in a subroutine's declaration section,
before executable statements. For example, remove the module-level registration
from `function_like_example` above and add this subroutine alongside its
`double_value` and `test` functions:

```fortran
subroutine differentiate(x, dx)
  use enzyme, only: enzyme_autodiff
  implicit none
  real, intent(in) :: x
  real, intent(inout) :: dx

  procedure(double_value), pointer :: &
    fn__enzyme_function_like__log1p => double_value

  call enzyme_autodiff(test, x, dx)
end subroutine differentiate
```

The main program can then import `differentiate` and call
`differentiate(x, dx)` with `x = 2.0` and `dx = 0.0`. The result is again
`0.3333`. The registration remains a compile-time annotation, not a runtime
switch local to this subroutine.
