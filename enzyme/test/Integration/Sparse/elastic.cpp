// This should work on LLVM 7, 8, 9, however in CI the version of clang installed on Ubuntu 18.04 cannot load
// a clang plugin properly without segfaulting on exit. This is fine on Ubuntu 20.04 or later LLVM versions...
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions  -ffast-math -mllvm -enable-load-pre=0 -std=c++11 -O1 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-auto-sparsity=1 | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions  -ffast-math -mllvm -enable-load-pre=0 -std=c++11 -O2 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-auto-sparsity=1  | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions  -ffast-math -mllvm -enable-load-pre=0 -std=c++11 -O3 %s -S -emit-llvm -o - %loadClangEnzyme  -mllvm -enzyme-auto-sparsity=1 | %lli - ; fi
// TODO: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -ffast-math -mllvm -enable-load-pre=0  -std=c++11 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-auto-sparsity=1 -S | %lli - ; fi
// TODO: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -ffast-math -mllvm -enable-load-pre=0  -std=c++11 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-auto-sparsity=1 -S | %lli - ; fi
// TODO: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -ffast-math -mllvm -enable-load-pre=0  -std=c++11 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-auto-sparsity=1 -S | %lli - ; fi

#include <vector>
#include <cmath>
#include <cassert>

#include "matrix.h"


template<typename T>
__attribute__((always_inline))
static T neo_hookean(const T (&F)[3][3], T youngs_modulus, T poisson_ratio) {
    T lame_lambda = youngs_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio));
    T lame_mu = youngs_modulus / (2 * (1 + poisson_ratio));
    T J = F[0][0] * (F[1][1] * F[2][2] - F[1][2] * F[2][1])
             - F[0][1] * (F[1][0] * F[2][2] - F[1][2] * F[2][0])
             + F[0][2] * (F[1][0] * F[2][1] - F[1][1] * F[2][0]);
    T J_cube_root = cbrt(J);
    return lame_lambda / 2 * pow(J - 1, 2) + lame_mu / 2 * (pow(J_cube_root, -2) * (F[0][0] * F[0][0] + F[0][1] * F[0][1] + F[0][2] * F[0][2] +
                                                                                  F[1][0] * F[1][0] + F[1][1] * F[1][1] + F[1][2] * F[1][2] +
                                                                                  F[2][0] * F[2][0] + F[2][1] * F[2][1] + F[2][2] * F[2][2]) - 3);
}

template<typename T>
__attribute__((always_inline))
static T elastic_energy_simple(
    const T *__restrict__ pos0,
    const int tets[][4],
    size_t n,
    const T youngs_modulus,
    const T poisson_ratio,
    const T *__restrict__ pos
) {

    using Vector = T[3];
    using Matrix = Vector[3];
    using Matrix4 = Vector[4];

    T total_energy = 0.0f;
    __builtin_assume(n != 0);
    for (size_t i = 0; i<n; i++) {
        const int t[4] = {
            tets[i][0],
            tets[i][1],
            tets[i][2],
            tets[i][3]
        };

        Matrix4 pos_i;
        get_pos(pos_i, pos, t);

        Matrix shape_matrix;
        elementwise_difference(shape_matrix[0], pos_i[0], pos_i[3]);
        elementwise_difference(shape_matrix[1], pos_i[1], pos_i[3]);
        elementwise_difference(shape_matrix[2], pos_i[2], pos_i[3]);

        transpose(shape_matrix, shape_matrix);

        Matrix4 pos0_i;
        get_pos(pos0_i, pos0, t);

        Matrix shape_matrix0;
        elementwise_difference(shape_matrix0[0], pos0_i[0], pos0_i[3]);
        elementwise_difference(shape_matrix0[1], pos0_i[1], pos0_i[3]);
        elementwise_difference(shape_matrix0[2], pos0_i[2], pos0_i[3]);
        transpose(shape_matrix0, shape_matrix0);

        Matrix inv_shape_matrix0;
        inv(inv_shape_matrix0, shape_matrix0);

        Matrix product;
        matrix_multiply(product, shape_matrix, inv_shape_matrix0);

        T energy = neo_hookean(product, youngs_modulus, poisson_ratio);
        total_energy += energy;
    }

    return total_energy;
}

template<typename T>
__attribute__((always_inline))
static void gradient_ip(
    const T *__restrict__ pos0,
    const int tets[][4],
    const size_t n,
    const T youngs_modulus,
    const T poisson_ratio,
    const T *__restrict__ pos,
    T *__restrict__ out
    )
{
    __enzyme_autodiff<void>((void *)elastic_energy_simple<T>,
                            enzyme_const, pos0,
                            enzyme_const, tets,
                            enzyme_const, n,
                            enzyme_const, youngs_modulus,
                            enzyme_const, poisson_ratio,
                            enzyme_dup, pos,
                            out);
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

__attribute__((enzyme_sparse_accumulate))
void inner_store(size_t offset, size_t i, double val, std::vector<Triple<double>> &hess) {
    hess.push_back(Triple<double>(offset, i, val));
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
std::vector<Triple<T>> hessian(
    const T *__restrict__ pos0,
    const int tets[][4],
    const size_t n,
    const T youngs_modulus,
    const T poisson_ratio,
    const T *__restrict__ pos,
    const size_t num_tets)
{
    std::vector<Triple<T>> hess;
    __builtin_assume(num_tets != 0);
    for (size_t i=0; i<4*num_tets; i++)
        __enzyme_fwddiff<void>((void *)gradient_ip<T>,
                               enzyme_const, pos0,
                               enzyme_const, tets,
                               enzyme_const, n,
                               enzyme_const, youngs_modulus,
                               enzyme_const, poisson_ratio,
                               enzyme_dup, pos, __enzyme_todense<T*>(ident_load<T>,   err_store<T>, i),
                               enzyme_dupnoneed, nullptr, __enzyme_todense<T*>(zero_load<T>,   csr_store<T>, i, &hess));
    return hess;
}


int main() {

    using TestType = float;
    TestType pos[] = {
        -0.5, -0.5, -0.5, -0.5
        };
    TestType pos0[] = {
            0.6760001,  9.438364 ,  2.3247766,  7.3768997
        };

    const size_t n = 1;
    int tets[n][4] = {
        {0, 1, 2, 3},
    };

    TestType youngs_modulus = 0.1f;
    TestType poisson_ratio = 0.49f;

    // float result = elastic_energy_simple<n>(pos0, tets, youngs_modulus, poisson_ratio, pos);
    TestType result = elastic_energy_simple(pos0, tets, n, youngs_modulus, poisson_ratio, pos);

    printf("Result: %f\n", result);

    // Derivative
    TestType dpos[sizeof(pos)/sizeof(pos[0])];
    for (size_t i=0; i<sizeof(dpos)/sizeof(dpos[0]); i++)
        dpos[i] = 0;
    gradient_ip(pos0, tets, n, youngs_modulus, poisson_ratio, pos, dpos);

    for (size_t i=0; i<sizeof(dpos)/sizeof(dpos[0]); i++)
        printf("grad_vert[%zu] = %f\n", i, dpos[i]);

    // Hessian
    const size_t num_tets = 1;
    auto hess_verts = hessian(pos0, tets, n, youngs_modulus, poisson_ratio, pos, num_tets);

    for (auto &hess : hess_verts) {
        printf("i=%lu, j=%lu, val=%f", hess.row, hess.col, hess.val);
    }

    return 0;
}
