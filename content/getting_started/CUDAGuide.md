---
title: "CUDA Guide"
date: "2021-01-20"
draft: false
weight: 5
---

## Reference C++ example

Suppose we wanted to port the following C++ code to CUDA, with Enzyme autodiff support:

``` cpp
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
```

A one-liner compilation of the above using Enzyme:

``` sh
clang test2.cpp -Xclang -load -Xclang /path/to/ClangEnzyme-11.so -O2 -fno-vectorize -fno-unroll-loops
```

## CUDA Example

When porting the above code, there are some caveats to be aware of:

1. CUDA 10.1 is the latest supported CUDA at the time of writing (Jan/20/2021) for LLVM 11.
2. ```__enzyme_autodiff``` should only be invoked on ```___device___``` code, not ```__global__``` kernel code. ```__global__``` kernels may be supported in the future.
3. ```--cuda-gpu-arch=sm_xx``` is usually needed as the default ```sm_20``` is unsupported by modern CUDA versions.


```cpp
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
```

The one-liner compilation step is (against sm_70):

```sh
clang test3.cu -Xclang -load -Xclang /path/to/ClangEnzyme-11.so -O2 -fno-vectorize -fno-unroll-loops -fPIC --cuda-gpu-arch=sm_70 -lcudart -L/usr/local/cuda-10.1/lib64
```
