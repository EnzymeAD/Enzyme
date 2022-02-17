.. _first-derivative:

Taking a First Derivative
=========================

This tutorial will present an overview of how reverse-mode compiler-based automatic
differentiation with Enzyme works, how the different stages of compilation are
altered under Enzyme's usage, and how Enzyme plugs into the compiler's infrastructure.

.. _ad-as-a-compilation-step:

AD as a Compilation Step
^^^^^^^^^^^^^^^^^^^^^^^^

Using Enzyme, we are able to directly plug into the compiler. Taking ``C`` as an example,
we directly plug into ``clang`` to perform AD as part of the compilation steps, but on the
higher level to the user, as a part of the single compilation step. Taking a simple
test-program computing the derivative of :math:`f(x)=x^{2}`, we begin by defining our
simple C-test we seek to differentiate.

.. code-block:: c

    // test.c
    #include <stdio.h>
    extern double __enzyme_autodiff(void*, double);
    double square(double x) {
        return x * x;
    }
    double dsquare(double x) {
        // This returns the derivative of square or 2 * x
        return __enzyme_autodiff((void*) square, x);
    }
    int main() {
        for(double i=1; i<5; i++)
            printf("square(%f)=%f, dsquare(%f)=%f", i, square(i), i, dsquare(i));
    }

To now use Enzyme to differentiate our test function, in the process of which it
replaces any calls to functions whose name contain ``__enzyme_autodiff`` with calls
to the corresponding derivatives of the language primitives used by the
to-be-differentiated function, we have to call ``clang`` and plug Enzyme into the
compiler. Note that ``clang`` should be the path to whatever clang you built Enzyme
against in installation_guide_. For now, let's ignore the details of Enzyme's calling
convention/ABI which are described in detail in calling_convention_.

.. code-block:: bash

    clang test.c -O2 -Xclang -load -Xclang /path/to/Enzyme/enzyme/build/Enzyme/ClangEnzyme-<VERSION>.so -fno-vectorize -fno-slp-vectorize -fno-unroll-loops -o first_grad.exe

But what does Enzyme actually do here, and at which stages does it alter the compilation?


.. _generating-llvm:

Generating LLVM
---------------

To break Enzyme's individual steps down, we first need to begin by generating the
LLVM IR from our test snippet. We can generate LLVM from this code by calling clang as follows.

.. code-block:: bash

    clang test.c -S -emit-llvm -o input.ll -O2 -fno-vectorize -fno-slp-vectorize -fno-unroll-loops

The arguments ``-S -emit-llvm`` specify that we want to emit LLVM bitcode, rather than
an executable. The arguments ``-o input.ll`` specify that we want the output to be in a
file ``input.ll``. The argument ``-O2 -ffast-math`` runs optimization (with fast-math)
before we run Enzyme's AD process, which is often beneficial to performance. The
argument ``-fno-vectorize -fno-slp-vectorize -fno-unroll-loops`` specifies that we don't
want to run vectorization or loop unrolling. In practice, it is better for performance
to only run these scheduling optimizations after AD.

The generated LLVM IR should look something like the following:

.. code-block:: llvm

    ; input.ll
    ...
    define double @square(double %x) #0 {
    entry:
      %mul = fmul double %x, %x
      ret double %mul
    }

    define double @dsquare(double %x) local_unnamed_addr #1 {
    entry:
      %call = tail call double @__enzyme_autodiff(i8* bitcast (double (double)* @square to i8*), double %x) #4
      ret double %call
    }
    ...

.. _performing-ad-with-enzyme:

Performing AD with Enzyme
-------------------------

We can now run Enzyme to differentiate our LLVM IR. The following command will load Enzyme
and run the differentiation transformation pass. Note that ``opt`` should be the path to
whatever opt was created by the LLVM you built Enzyme against. If you see a segfault when
trying to run opt, this is likely an issue in LLVM's plugin infrastructure. Please see
installation_guide_ for more information on how to resolve this.

.. code-block:: bash

    opt input.ll -load=/path/to/Enzyme/enzyme/build/Enzyme/LLVMEnzyme-<VERSION>.so -enzyme -o output.ll -S

Taking a look at ``output.ll``, we find the following:

.. code-block:: llvm

    ; output.ll
    define internal { double } @diffesquare(double %x, double %differeturn) #0 {
    entry:
      %"mul'de" = alloca double
      store double 0.000000e+00, double* %"mul'de"
      %"x'de" = alloca double
      store double 0.000000e+00, double* %"x'de"
      br label %invertentry

    invertentry:                                      ; preds = %entry
      store double %differeturn, double* %"mul'de"
      %0 = load double, double* %"mul'de"
      %m0diffex = fmul fast double %0, %x
      %m1diffex = fmul fast double %0, %x
      store double 0.000000e+00, double* %"mul'de"
      %1 = load double, double* %"x'de"
      %2 = fadd fast double %1, %m0diffex
      store double %2, double* %"x'de"
      %3 = load double, double* %"x'de"
      %4 = fadd fast double %3, %m1diffex
      store double %4, double* %"x'de"
      %5 = load double, double* %"x'de"
      %6 = insertvalue { double } undef, double %5, 0
      ret { double } %6
    }

    define double @dsquare(double %x) local_unnamed_addr #1 {
    entry:
      %0 = call { double } @diffesquare(double %x, double 1.000000e+00)
      %1 = extractvalue { double } %0, 0
      ret double %1
    }

Enzyme has created a new gradient function, and replaced the corresponding call to
``__enzyme_autodiff``. Note that the newly-created gradient function isn't yet optimized.
Enzyme assumes that various post-processing will occur after creating the gradient.

For example, suppose we run ``-O2`` after Enzyme as shown below:

.. code-block:: bash

    opt output.ll -O2 -o output_opt.ll -S

Taking a look at ``output_opt.ll``, we see the following:

.. code-block:: llvm

    ; output_opt.ll
    define double @dsquare(double %x) local_unnamed_addr #0 {
    entry:
      %factor.i = fmul fast double %x, 2.000000e+00
      ret double %factor.i
    }

The generated gradient has been inlined and entirely simplified to return the input
times two.

We can then compile this into a final binary as follows:

.. code-block:: bash

    clang output_opt.ll -o first_grad.exe

For ease, we could combine the final optimization, and binary execution into one
command as follows:

.. code-block:: bash

    clang output.ll -O3 -o first_grad.exe
