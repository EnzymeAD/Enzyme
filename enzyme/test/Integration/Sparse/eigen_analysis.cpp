// This should work on LLVM 7, 8, 9, however in CI the version of clang installed on Ubuntu 18.04 cannot load
// a clang plugin properly without segfaulting on exit. This is fine on Ubuntu 20.04 or later LLVM versions...
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions  -ffast-math -mllvm -enable-load-pre=0 -std=c++11 -O1 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-auto-sparsity=1 | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions  -ffast-math -mllvm -enable-load-pre=0 -std=c++11 -O2 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-auto-sparsity=1  | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions  -ffast-math -mllvm -enable-load-pre=0 -std=c++11 -O3 %s -S -emit-llvm -o - %loadClangEnzyme  -mllvm -enzyme-auto-sparsity=1 | %lli - ; fi
// TODO: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -ffast-math -mllvm -enable-load-pre=0  -std=c++11 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-auto-sparsity=1 -S | %lli - ; fi
// TODO: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -ffast-math -mllvm -enable-load-pre=0  -std=c++11 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-auto-sparsity=1 -S | %lli - ; fi
// TODO: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -ffast-math -mllvm -enable-load-pre=0  -std=c++11 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-auto-sparsity=1 -S | %lli - ; fi

#include <vector>
#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "matrix.h"

template<typename T>
__attribute__((always_inline))
static T eigenstuffM(const T *__restrict__ x, size_t n, const int *__restrict__ faces, const T *__restrict__ pos0) {
    T sum = 0;
    __builtin_assume(n != 0);
    for (size_t idx=0; idx<n; idx++) {
        int i = faces[3*idx];
        int j = faces[3*idx+1];
        int k = faces[3*idx+2];
        T tri_area = area(&pos0[3*i], &pos0[3*j], &pos0[3*k]);
        sum += (x[i] * x[i] + x[j] * x[j] + x[k] * x[k]) * (1.0 / 3.0) * tri_area;  // barycentric mass lumping
    }
    return sum;
}


// Calculate total energy for all faces in 3D
template<typename T>
__attribute__((always_inline))
static T eigenstuffL(const T *__restrict__ x, size_t num_faces, const int *__restrict__ faces, const T *__restrict__ verts) {
    T sum = 0;
    __builtin_assume(num_faces != 0);
    for (size_t idx=0; idx<num_faces; idx++) {
        int i = faces[3*idx];
        int j = faces[3*idx+1];
        int k = faces[3*idx+2];

        T X[2][3] = {
            {   verts[3*j+0] - verts[3*i+0],
                verts[3*j+1] - verts[3*i+1],
                verts[3*j+2] - verts[3*i+2]},
            {verts[3*k+0] - verts[3*i+0], verts[3*k+1] - verts[3*i+1], verts[3*k+2] - verts[3*i+2]}
        };

        T pInvX[3][2];
        pseudo_inverse(pInvX, X);
       
        T diffs[] = {x[j] - x[i], x[k] - x[i]};

        T g[3];
        #pragma clang loop unroll(full)
        for (int i = 0; i < 3; ++i) {
            T sum = 0.0f;
            #pragma clang loop unroll(full)
            for (int j = 0; j < 2; ++j) {
                sum += pInvX[i][j] * diffs[j];
            }
            g[i] = sum;
        }

        sum += dot_product<T, 3>(g, g) * area(&verts[3*i], &verts[3*j], &verts[3*k]);
    }

    return sum;
}


template<typename T>
__attribute__((always_inline))
static void gradient_ip(const T *__restrict__ x, const size_t num_faces, const int* faces, const T *__restrict__ pos, T *__restrict__ out)
{
    __enzyme_autodiff<void>((void *)eigenstuffM<T>,
                            enzyme_const, x,
                            enzyme_const, num_faces,
                            enzyme_const, faces,
                            enzyme_dup, pos, out);
}


template<typename T>
__attribute__((always_inline))
static T ident_load(unsigned long long offset, size_t i) {
    return (offset / sizeof(T) == i) ? T(1) : T(0); 
}


template<typename T>
__attribute__((always_inline))
static void err_store(T val, unsigned long long offset, size_t i) {
    assert(0 && "store is not legal");
}


template<typename T>
__attribute__((always_inline))
static T zero_load(unsigned long long offset, size_t i, std::vector<Triple<T>> &hess) {
    return T(0);
}


__attribute__((enzyme_sparse_accumulate))
void inner_store(size_t offset, size_t i, float val, std::vector<Triple<float>> &hess) {
    hess.push_back(Triple<float>(offset, i, val));
}

template<typename T>
__attribute__((always_inline))
static void csr_store(T val, unsigned long long offset, size_t i, std::vector<Triple<T>> &hess) {
    if (val == 0.0) return;
    offset /= sizeof(T);
    inner_store(offset, i, val, hess);
}

template<typename T>
__attribute__((noinline))
std::vector<Triple<T>> hessian(const T* x, size_t num_faces, const int* faces, const T* pos, size_t num_verts)
{

    std::vector<Triple<T>> hess;
    __builtin_assume(num_verts != 0);
    for (size_t i=0; i<3*num_verts; i++)
        __enzyme_fwddiff<void>((void *)gradient_ip<T>,
                                enzyme_const, x,
                                enzyme_const, num_faces,
                                enzyme_const, faces,
                                enzyme_dup, pos, __enzyme_todense<T*>(ident_load<T>, err_store<T>, i),
                                enzyme_dupnoneed, nullptr, __enzyme_todense<T*>(zero_load<T>, csr_store<T>, i, &hess));
    return hess;
}


int main() {
    const size_t num_elts_data = 3;
    const float x[] = {0.0, 1.0, 0.0};


    const size_t num_faces = 1;
    const int faces[] = {0, 1, 2};

    const size_t num_verts = 3;
    const float verts[] = {1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0, 1.0, 3.0};

    // Call eigenstuffM_simple
    const float resultM = eigenstuffM(x, num_faces, faces, verts);
    printf("Result for eigenstuffM_simple: %f\n", resultM);

    // Call eigenstuffL_simple
    const float resultL = eigenstuffL(x, num_faces, faces, verts);
    printf("Result for eigenstuffL_simple: %f\n", resultL);

    float dverts[sizeof(verts)/sizeof(verts[0])];
    for (size_t i=0; i<sizeof(dverts)/sizeof(dverts[0]); i++)
        dverts[i] = 0;
    gradient_ip(x, num_faces, faces, verts, dverts);

    for (size_t i=0; i<sizeof(dverts)/sizeof(dverts[0]); i++)
        printf("eigenstuffM grad_vert[%zu]=%f\n", i, dverts[i]);
    
    size_t num_elts = sizeof(verts)/sizeof(verts[0]) * sizeof(verts)/sizeof(verts[0]);

    auto hess_verts = hessian(x, num_faces, faces, verts, num_verts);

    for (auto &hess : hess_verts) {
        printf("i=%lu, j=%lu, val=%f\n", hess.row, hess.col, hess.val);
    }

    return 0;
}

