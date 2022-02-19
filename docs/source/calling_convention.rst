.. _calling-convention:

Calling Convention
==================

Enzyme is invoked by calling a function `__enzyme_autodiff` with the function being differentiated,
followed by the corresponding primal and shadow arguments. This will result in the original function
being run with the corresponding derivative values being computed.

.. _function-hooks:

Function Hooks
^^^^^^^^^^^^^^

Enzyme replaces all calls to functions that contain the string ``__enzyme_autodiff`` with a call to the corresponding derivative.
This is done to allow Enzyme to register multiple function signatures.

.. code-block:: cpp

    #include <stdio.h>
    template<typename T>
    T square(T x) { return x * x; }

    float __enzyme_autodiffFloat(float (*)(float), float);
    double __enzyme_autodiffDouble(double (*)(double), double);

    int main() {
      printf("float  d/dx %f\n", __enzyme_autodiffFloat(square<float>, 1.0f));
      printf("double d/dx %f\n", __enzyme_autodiffDouble(square<double>, 1.0));
    }

This allows end-library makers to nicely incorporate Enzyme into their workflow through the use
of variadic arguments or templates.

.. code-block:: cpp

    void __enzyme_autodiff(...);

    template<typename RT, typename... Args>
    RT __enzyme_autodiff(void*, Args...);

The first argument should either be a function pointer to the code being differentiated, or
a cast of the function pointer.

.. _types:

Types
^^^^^

Arguments to functions being differentiaed are classified to three types:

* *_Inactive arguments_* whose values don't impact the derivative computation. An example of this would be an integer representing the size of an array.
* *_Output arguments_* are active values whose gradient result is passed as a return value. Examples include floats or doubles.
* *_Duplicated arguments_* are active values whose gradient result is stored in a second shadow argument. All active pointer values are duplicated arguments.

An example program using all three types of these arguments is shown below:

.. code-block:: c

    double sumAndMul(double* array, size_t size, double mul) {
      double sum = 0;
      for(int i=0; i<size; i++) {
        sum += array[i];
      }
      return sum * mul
    }

    ...
    double d_mul = __enzyme_autodiff(sumAndMul,
                         /*duplicated argument*/array, d_array,
                         /*inactive argument*/size,
                         /*output argument*/mul);
    ...

Enzyme will automatically attempt to deduce the classification of argument types. Generally, these rules assume that integer types are inactive arguments,
floating-point types are output arguments, and pointer-types are duplicated arguments. A user, however, can explicitly specify the desired classification
by using LLVM metadata.

Inactive arguments are given ``enzyme_const`` metadata; output arguments are given ``enzyme_out`` metadata; and duplicated arguments are given ``enzyme_dup``.

.. code-block:: llvm

    %d_mul = tail call double @__enzyme_autodiff(double (double*, i64, double)* @sumAndMul,
        metadata !"enzyme_dup", double* %array, double* %d_array,
        metadata !"enzyme_const", i64 %size,
        metadata !"enzyme_out", double %mul)

To ease the process of writing frontends, Enzyme will also consider loads to global values with specific names as a mechanism to specify argument classification.

.. code-block:: c

    int enzyme_dup;
    int enzyme_out;
    int enzyme_const;

    int main() {
      double d_mul = __enzyme_autodiff(sumAndMul,
                           enzyme_dup  , array, d_array,
                           enzyme_const, size,
                           enzyme_out  , mul);
    }

.. _shadow-argument-initialization:

Shadow Argument Initialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Enzyme assumes that shadow arguments passed in are already initialized and have the same structure as the primal values. Running Enzyme's generated gradient
will increment the shadow value by the amount of the resultant gradient. As a result, this usually means that you want to zero-initialize the shadow prior
to calling the gradient.

.. code-block:: c

    double   array[10] = { ... };
    double d_array[10] = { 0.0 };

    __enzyme_autodiff(sumSquare,
                      enzyme_dup, array, d_array);

    printf("d(output)/darray[0] = %f\n", d_array[0]);

For complex datastructures passed as arguments, this requires doing a corresponding initialization of the shadow.

.. code-block:: cpp

    struct List {
      double value;
      List* next;
    }

    double sumList(List* next);
    List* mklist(double value, List* next);

    List*   list = nullptr;
    List* d_list = nullptr;

    for(int i=0; i<5; ++i) {
        list = mklist(  i,   list);
      d_list = mklist(0.0, d_list);
    }

    __enzyme_autodiff(sumList, list, d_list);

.. _result-only-duplicated-argument:

Result-only Duplicated Argument
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Enzyme also supports a special version of duplicated argument where users only need the computed gradient of the argument and not
the value computed in the forward pass. For example, consider the function below that computes a loss function. All the user needs
is the gradient of the inputs with respect to the loss and not the loss itself.

We can instead use the value ``enzyme_dupnoneed`` to specify this property to Enzyme. This allows Enzyme to do additional optimization.

.. code-block:: cpp

    void neuralNet(double* loss, double* W, double* b, double* input);

    void main() {
      ...
      double loss;
      double d_loss = 1.0;
      __enzyme_autodiff(neuralNet,
                        enzyme_dupnoneed, &loss, &d_loss,
                        enzyme_dup,       W, d_W,
                        enzyme_dup,       b, d_b,
                        enzyme_const,     input);
      // This value is undefined behavior if using diffe_dupnoneed, otherwise
      // it is the same as it would be from calling neuralNet normally.
      printf("loss=%f\n", loss);

      printf("d_b[0]=%f\n", d_b[0]);
    }

.. _wrapper-functions:

Wrapper Functions
^^^^^^^^^^^^^^^^^

When passing complicated types as arguments, it's sometimes desirable to explicitly pass them as duplicated argument. This can
be accomplished by creating a wrapper function that takes a pointer argument and simply calls a function with the reference value.

.. code-block:: cpp

    class MyClass;

    MyClass compute(MyClass&);

    void wrapper(MyClass* in, MyClass* out) {
      *out = compute(*in);
    }

    MyClass d_compute(MyClass& in) {
      MyClass d_in(0.0);
      MyClass out;
      MyClass d_out(1.0);
      __enzyme_autodiff(wrapper, &in, d_in, out, d_out);
      return d_in;
    }

.. _globals:

Globals
^^^^^^^

All global variables that are active must have their shadow explicitly specified in LLVM. This is done by attaching
metadata that specifies what the shadow of that global is.

.. code-block:: llvm

    @global = external local_unnamed_addr global double, align 8,
        !enzyme_shadow !{double* @dglobal}
    @dglobal = external local_unnamed_addr global double, align

.. _custom-gradients:

Custom Gradients
^^^^^^^^^^^^^^^^

Functions can be given a custom gradient by attaching two pieces of metadata. These pieces of metadata specify an
augmented forward pass that saves any state necessary for the reverse pass and the reverse pass that computes the gradient.

Presently, custom gradients are only supported where Enzyme's default argument classification is correct. This
means that the all floating-point arguments must be treated as active output arguments, all pointer arguments
must be treated as active duplicated arguments, and all integers are inactive arguments.

Both functions have the same arguments the forward pass along with any duplicated arguments mixed in. The gradient
function then has a differential return value if the original function's return value is an output argument. The
final argument is a custom "tape" type that can be used to pass information from the forward to the reverse pass.

The return type of the augmented forward pass is a struct type containing first the tape type, followed by the
original return type, if any. If the return type is a duplicated type, then there is a third argument which
contains the shadow of the return.

The return type of the reverse pass is a struct containing derivatives of all of the output arguments.

.. code-block:: llvm

    define internal { {}, double } @augment_add2(double %x) {
    entry:
      %add = fadd fast double %x, 2.000000e+00
      %struct1 = insertvalue { {}, double } undef, double %add, 1
      ret { {}, double } %struct1
    }

    define internal { double } @gradient_add2(double %x, double %differet, {} %tapeArg) {
    entry:
      %struct1 = insertvalue { double } undef, double %differet, 0
      ret { double } %struct1
    }

    declare !enzyme_augment !{{ {}, double } (double)* @augment_add2} !enzyme_gradient !{{ double } (double, double, {})* @gradient_add2} double @add2(double %x)
