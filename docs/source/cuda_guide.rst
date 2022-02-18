.. _cuda-guide:

AD of CUDA
==========

.. _reference-cpp-example:

Reference C++ example
---------------------

    **WARNING**: CUDA support is highly experimental and in active development.

Suppose we wanted to port the following C++ code to CUDA, with Enzyme autodiff support:

.. code-block:: cpp

    #include <stdio.h>

    void foo(double* x_in, double *x_out) {
        x_out[0] = x_in[0] * x_in[0];
    }


    int enzyme_dup;
    int enzyme_out;
    int enzyme_const;

    typedef void (*f_ptr)(double*, double*);

    extern void __enzyme_autodiff(f_ptr,
        int, double*, double*,
        int, double*, double*);

    int main() {

        double x = 1.4;
        double d_x = 0.0;
        double y;
        double d_y = 1.0;

        __enzyme_autodiff(foo,
            enzyme_dup, &x, &d_x,
            enzyme_dup, &y, &d_y);

        printf("%f %f\n", x, y);
        printf("%f %f\n", d_x, d_y);

    }

A one-liner compilation of the above using Enzyme:

.. code-block:: bash

    clang test2.cpp -Xclang -load -Xclang /path/to/ClangEnzyme-11.so -O2 -fno-vectorize -fno-unroll-loops

.. _cuda-example:

CUDA Example
------------

When porting the above code, there are some caveats to be aware of:

1. CUDA 10.1 is the latest supported CUDA at the time of writing (Jan/20/2021) for LLVM 11.
2. ``__enzyme_autodiff`` should only be invoked on ``___device___`` code, not ``__global__`` kernel code. ``__global__`` kernels may be supported in the future.
3. ``--cuda-gpu-arch=sm_xx`` is usually needed as the default ``sm_20`` is unsupported by modern CUDA versions.

.. code-block:: cpp

    #include <stdio.h>

    void __device__ foo_impl(double* x_in, double *x_out) {
        x_out[0] = x_in[0] * x_in[0];    
    }

    typedef void (*f_ptr)(double*, double*);

    extern void __device__ __enzyme_autodiff(f_ptr,
        int, double*, double*,
        int, double*, double*
    );

    void __global__ foo(double* x_in, double *x_out) {
        foo_impl(x_in, x_out);
    }

    int __device__ enzyme_dup;
    int __device__ enzyme_out;
    int __device__ enzyme_const;

    void __global__ foo_grad(double* x, double *d_x, double *y, double *d_y) {

        __enzyme_autodiff(foo_impl,
            enzyme_dup, x, d_x,
            enzyme_dup, y, d_y);

    }

    int main() {

        double *x, *d_x, *y, *d_y; // device pointers

        cudaMalloc(&x, sizeof(*x));
        cudaMalloc(&d_x, sizeof(*d_x));
        cudaMalloc(&y, sizeof(*y));
        cudaMalloc(&d_y, sizeof(*d_y));

        double host_x = 1.4;
        double host_d_x = 0.0;
        double host_y;
        double host_d_y = 1.0;

        cudaMemcpy(x, &host_x, sizeof(*x), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, &host_d_x, sizeof(*d_x), cudaMemcpyHostToDevice);
        cudaMemcpy(y, &host_y, sizeof(*y), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, &host_d_y, sizeof(*d_y), cudaMemcpyHostToDevice);

        // foo<<<1,1>>>(x, y); fwd-pass only
        foo_grad<<<1,1>>>(x, d_x, y, d_y); // fwd and bkwd pass

        cudaDeviceSynchronize(); // synchroniz

        cudaMemcpy(&host_x, x, sizeof(*x), cudaMemcpyDeviceToHost);
        cudaMemcpy(&host_d_x, d_x, sizeof(*d_x), cudaMemcpyDeviceToHost);
        cudaMemcpy(&host_y, y, sizeof(*y), cudaMemcpyDeviceToHost);
        cudaMemcpy(&host_d_y, d_y, sizeof(*d_y), cudaMemcpyDeviceToHost);

        printf("%f %f\n", host_x, host_y);
        printf("%f %f\n", host_d_x, host_d_y);

    }

For convenience, a one-liner compilation step is (against sm_70):

.. code-block:: bash

    clang test3.cu -Xclang -load -Xclang /path/to/ClangEnzyme-11.so -O2 -fno-vectorize -fno-unroll-loops -fPIC --cuda-gpu-arch=sm_70 -lcudart -L/usr/local/cuda-10.1/lib64

Note that this procedure (using ClangEnzyme as opposed to LLVMEnzyme manually) may not properly nest Enzyme between optimization passes and may impact performance in unintended ways.

.. _heterogeneous-ad:

Heterogeneous AD
----------------

It is often desirable to take derivatives of programs that run in part on the CPU and in part on the GPU. By placing a call to `__enzyme_autodiff` in a GPU kernel like above, one can successfully take the derivative of GPU programs. Similarly one can use ``__enzyme_autodiff`` within CPU programs to differentiate programs which run entirely on the CPU. Unfortunately, differentiating functions that call GPU kernels requires a bit of extra work (shown below) -- largely to work around the lack of support within LLVM for modules with multiple architecture targets.

To successfully differentiate across devices, we will use Enzyme on the GPU to export the augmented forward pass and reverse pass of the kernel being called, and then use Enzyme's custom derivative support to import that derivative function into the CPU code. This then allows Enzyme to differentiate any CPU code that also calls the kernel.

Suppose we have a heterogeneous program such as the following:

.. code-block:: cpp

    // GPU Kernel
    __global__ 
    void collide(float* src, float* dst) {
        size_t idx = threadIdx.x;
        if (idx < 100) {
            dsr[idx] += src[idx] * src[idx] - 3 * src[idx];
        }
    }

    // Wrapper CPU function which calls kernel
    void kern(float* src, float* dst) {
        collide<<<1, 100>>>(src, dst);
    }

    // Main CPU code that calls wrapper function
    void iter(int nTimeSteps, float* src, float* dst) {
        for (unsigned int i=0; i<nTimeSteps/2; i++) {
            kern(src, dst);
            kern(dst, src);
        }
    }

We would first try to differentiate the CPU side by calling ``__enzyme_autodiff`` on ``iter`` as shown below:

.. code-block:: cpp

    template <typename... Args>
    void __enzyme_autodiff(Args...);

    void grad_iter(int nTimeSteps, float* src, float* dsrc, float* dst, float* ddst) {
      __enzyme_autodiff(iter, nTimeSteps, src, dsrc, dst, ddst);
    }

Enzyme, however, would return an error saying it cannot differentiate through a CUDA call, which appears like the following:

.. code-block:: bash

    declare dso_local i32 @__cudaPushCallConfiguration(i64, i32, i64, i32, i64, i8*) local_unnamed_addr #2

    clang-13: /home/wmoses/git/Enzyme/enzyme/Enzyme/EnzymeLogic.cpp:1459: const AugmentedReturn& EnzymeLogic::CreateAugmentedPrimal(llvm::Function*, DIFFE_TYPE, const std::vector<DIFFE_TYPE>&, llvm::TargetLibraryInfo&, TypeAnalysis&, bool, const FnTypeInfo&, std::map<llvm::Argument*, bool>, bool, bool, bool, bool): Assertion '0 && "attempting to differentiate function without definition"' failed.
    PLEASE submit a bug report to https://bugs.llvm.org/ and include the crash backtrace, preprocessed source, and associated run script.


To remedy this, we can use Enzyme's custom derivative registration to define a custom forward and reverse pass for the wrapper function `kern` as follows:

.. code-block:: cpp

    // We move the body of collide into a separate device function collide_body to allow us
    // to pass collide_body to various differentiation methods. This is necessary as differentiation
    // can only be done on device, not global kernel functions.
    __device__
    void collide_body(float* src, float* dst) {
        size_t idx = threadIdx.x;
        if (idx < 100) {
            dst[idx] += src[idx] * src[idx] - 3 * src[idx];
        }
    }

    // GPU Kernel
    __global__
    void collide(float* src, float* dst) {
        collide_body(src, dst);
    }

    // Wrapper CPU function which calls kernel
    __attribute__((noinline))
    void kern(float* src, float* dst) {
        collide<<<1, 100>>>(src, dst);
    }

    // Main CPU code that calls wrapper function
    void iter(int nTimeSteps, float* src, float* dst) {
        for (unsigned int i=0; i<nTimeSteps/2; i++) {
            kern(src, dst);
            kern(dst, src);
        }
    }

    template <typename... Args>
    void __enzyme_autodiff(Args...);

    void grad_iter(int nTimeSteps, float* src, float* dsrc, float* dst, float* ddst) {
        __enzyme_autodiff(iter, nTimeSteps, src, dsrc, dst, ddst);
    }

    // A function similar to __enzyme_autodiff, except it only calls the augmented forward pass, returning
    // a tape structure to hold any values that may be overwritten and needed for the reverse.
    template <typename... Args>
    __device__ void* __enzyme_augmentfwd(Args...);

    // A function similar to __enzyme_autodiff, except it only calls the revese pass, taking in the tape
    // as its last argument.
    template <typename... Args>
    __device__ void __enzyme_reverse(Args...);

    // A wrapper GPU kernel for calling the forward pass of collide. The wrapper code stores
    // the tape generated by Enzyme into a unique location per thread
    __global__ void aug_collide(float* src, float* dsrc, float* dst, float* ddst, void** tape)
    {
        size_t idx = threadIdx.x;
        tape[idx] = __enzyme_augmentfwd((void*)collide_body, src, dsrc, dst, ddst);
    }

    // A wrapper GPU kernel for calling the reverse pass of collide. The wrapper code retrieves
    // the corresponding tape per thread being executed.
    __global__ void rev_collide( float* src, float* dsrc, float* dst, float* ddst, void** tape)
    {
        size_t idx = threadIdx.x;
        __enzyme_reverse((void*)collide_body, src, dsrc, dst, ddst, tape[idx]);
    }

    // The augmented forward pass of the CPU kern call, allocating and returning
    // tape memory  needed to compute the reverse pass. This calls a augmented collide
    // GPU kernel, passing in a unique 8-byte location to store the tape.
    void* aug_kern(float* src, float* dsrc, float* dst, float* ddst) {
        void** tape;
        cudaMalloc(&tape, sizeof(void*) * /*total number of threads*/100);
        aug_collide<<<1, 100>>>(src, dsrc, dst, ddst, tape);
        return (void*)tape;
    }

    // The reverse pass of the CPU kern call, using tape memory passed as the
    // last argument. This calls a reverse collide GPU kernel.
    void rev_kern(float* src, float* dsrc, float* dst, float* ddst, void* tape) {
        rev_collide<<<1, 100>>>(src, dsrc, dst, ddst, (void**)tape);
        cudaFree(tape);
    }

    // Here we register the custom forward pass aug_kern and reverse pass rev_kern
    void* __enzyme_register_gradient_kern[3] = { (void*)kern, (void*)aug_kern, (void*)rev_kern };

Finally, Enzyme has a performance optimization available when creating forward and reverse passes using ``__enzyme_augmentfwd`` and ``__enzyme_reverse``. By default, these methods store all variables inside the differentiated function within a generic pointer type (e.g.  ``void*``), thereby allowing Enzyme to store as much memory as it needs without issue. This, of course, requires an extra indirection to get to the underlying memory being stored.

If one knew statically how much memory is required per thread (in this case a single float to store ``src[idx]``), one could tell Enzyme to allocate directly into the tape rather than using this extra level of indirect. This is performed as follows:

.. code-block:: cpp

    // Magic Global used to specify how to call Enzyme. In this case, we specify how much memory
    // is allocated per invocation within the tape to allow the cache to be inlined.
    extern __device__ int enzyme_allocated;

    // A wrapper GPU kernel for calling the forward pass of collide. The wrapper code stores
    // the tape generated by Enzyme into a unique location per thread
    __global__ void aug_collide(float* src, float* dsrc, float* dst, float* ddst, float* tape)
    {
        size_t idx = threadIdx.x;
        tape[idx] = __enzyme_augmentfwd((void*)collide_body, enzyme_allocated, sizeof(float), src, dsrc, dst, ddst);
    }

    // A wrapper GPU kernel for calling the reverse pass of collide. The wrapper code retrieves
    // the corresponding tape per thread being executed.
    __global__ void rev_collide( float* src, float* dsrc, float* dst, float* ddst, float* tape)
    {
        size_t idx = threadIdx.x;
        __enzyme_reverse((void*)collide_body, enzyme_allocated, sizeof(float), src, dsrc, dst, ddst, tape[idx]);
    }

    // The augmented forward pass of the CPU kern call, allocating and returning
    // tape memory  needed to compute the reverse pass. This calls a augmented collide
    // GPU kernel, passing in a unique 8-byte location to store the tape.
    void* aug_kern(float* src, float* dsrc, float* dst, float* ddst) {
        float* tape;
        cudaMalloc(&tape, sizeof(float) * /*total number of threads*/100);
        aug_collide<<<1, 100>>>(src, dsrc, dst, ddst, tape);
        return (void*)tape;
    }

    // The reverse pass of the CPU kern call, using tape memory passed as the
    // last argument. This calls a reverse collide GPU kernel.
    void rev_kern(float* src, float* dsrc, float* dst, float* ddst, void* tape) {
        rev_collide<<<1, 100>>>(src, dsrc, dst, ddst, (float*)tape);
        cudaFree(tape);
    }