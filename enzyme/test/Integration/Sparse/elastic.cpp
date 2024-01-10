// This should work on LLVM 7, 8, 9, however in CI the version of clang installed on Ubuntu 18.04 cannot load
// a clang plugin properly without segfaulting on exit. This is fine on Ubuntu 20.04 or later LLVM versions...
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -std=c++11 -O1 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-auto-sparsity=1 | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -std=c++11 -O2 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-auto-sparsity=1  | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -std=c++11 -O3 %s -S -emit-llvm -o - %loadClangEnzyme  -mllvm -enzyme-auto-sparsity=1 | %lli - ; fi
// TODO: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -std=c++11 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-auto-sparsity=1 -S | %lli - ; fi
// TODO: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -std=c++11 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-auto-sparsity=1 -S | %lli - ; fi
// TODO: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -std=c++11 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-auto-sparsity=1 -S | %lli - ; fi

#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <cassert>

int enzyme_width;
int enzyme_dup;
int enzyme_dupv;
int enzyme_const;
int enzyme_dupnoneed;
template <typename T>
extern T __enzyme_autodiff(void*,
                           int, const float*,
                           int, const int (*)[4],
                           int, const size_t,
                           int, const float,
                           int, const float,
                           int, const float*,
                           float*,
                           ...);
template <typename T>
extern T __enzyme_fwddiff(void *,
                          int, const float*,
                          int, const int(*)[4],
                          int, const size_t,
                          int, const float,
                          int, const float,
                          int, const float*,
                          const float*,
                          int, std::nullptr_t, const float*,
                          ...);
template <typename T>
T __enzyme_todense(...);
// extern T __enzyme_todense(T (*)(unsigned long long, size_t), void (*)(T, unsigned long long, size_t), size_t, ...);


using namespace std;

using Vector3f = array<float, 3>;
using Matrix3f = array<Vector3f, 3>;


__attribute__((always_inline))
tuple<Vector3f, Vector3f, Vector3f, Vector3f> get_pos4(
    const float *__restrict__ pos,
    int i0,
    int i1,
    int i2,
    int i3
) {
    // extract the 3d points at i0, i1, i2, i3
    Vector3f pos_i0 = {pos[3 * i0], pos[3 * i0 + 1], pos[3 * i0 + 2]};
    Vector3f pos_i1 = {pos[3 * i1], pos[3 * i1 + 1], pos[3 * i1 + 2]};
    Vector3f pos_i2 = {pos[3 * i2], pos[3 * i2 + 1], pos[3 * i2 + 2]};
    Vector3f pos_i3 = {pos[3 * i3], pos[3 * i3 + 1], pos[3 * i3 + 2]};

    return make_tuple(pos_i0, pos_i1, pos_i2, pos_i3);
}


__attribute__((always_inline))
static float neo_hookean(const Matrix3f& F, float youngs_modulus, float poisson_ratio) {
    float lame_lambda = youngs_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio));
    float lame_mu = youngs_modulus / (2 * (1 + poisson_ratio));
    float J = F[0][0] * (F[1][1] * F[2][2] - F[1][2] * F[2][1])
             - F[0][1] * (F[1][0] * F[2][2] - F[1][2] * F[2][0])
             + F[0][2] * (F[1][0] * F[2][1] - F[1][1] * F[2][0]);
    float J_cube_root = cbrt(J);

    return lame_lambda / 2 * pow(J - 1, 2) + lame_mu / 2 * (pow(J_cube_root, -2) * (F[0][0] * F[0][0] + F[0][1] * F[0][1] + F[0][2] * F[0][2] +
                                                                                  F[1][0] * F[1][0] + F[1][1] * F[1][1] + F[1][2] * F[1][2] +
                                                                                  F[2][0] * F[2][0] + F[2][1] * F[2][1] + F[2][2] * F[2][2]) - 3);
}


__attribute__((always_inline))
static Vector3f elementwise_difference(const Vector3f& pos_i0, const Vector3f& pos_i1) {
    return {pos_i0[0] - pos_i1[0], pos_i0[1] - pos_i1[1], pos_i0[2] - pos_i1[2]};
}


__attribute__((always_inline))
static Matrix3f transpose(const Matrix3f& F) {
    return {{
        {F[0][0], F[1][0], F[2][0]},
        {F[0][1], F[1][1], F[2][1]},
        {F[0][2], F[1][2], F[2][2]}
    }};
}


__attribute__((always_inline))
static Matrix3f inv(const Matrix3f& F) {
    float det = F[0][0] * (F[1][1] * F[2][2] - F[1][2] * F[2][1])
              - F[0][1] * (F[1][0] * F[2][2] - F[1][2] * F[2][0])
              + F[0][2] * (F[1][0] * F[2][1] - F[1][1] * F[2][0]);

    float inv_det = 1 / det;
    return {{
        {
            (F[1][1] * F[2][2] - F[1][2] * F[2][1]) * inv_det, 
            (F[0][2] * F[2][1] - F[0][1] * F[2][2]) * inv_det,
            (F[0][1] * F[1][2] - F[0][2] * F[1][1]) * inv_det,
        },
        {
            (F[1][2] * F[2][0] - F[1][0] * F[2][2]) * inv_det,
            (F[0][0] * F[2][2] - F[0][2] * F[2][0]) * inv_det,
            (F[0][2] * F[1][0] - F[0][0] * F[1][2]) * inv_det,
        },
        {
            (F[1][0] * F[2][1] - F[1][1] * F[2][0]) * inv_det,
            (F[0][1] * F[2][0] - F[0][0] * F[2][1]) * inv_det,
            (F[0][0] * F[1][1] - F[0][1] * F[1][0]) * inv_det,
        }
    }};
}


__attribute__((always_inline))
static Matrix3f matrix_multiply(const Matrix3f& A, Matrix3f& B) {
    Matrix3f C = {{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            C[i][j] = 0;
            for (int k = 0; k < 3; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}


__attribute__((always_inline))
static float elastic_energy_simple(
    const float *__restrict__ pos0,
    const int tets[][4],
    size_t n,
    const float youngs_modulus,
    const float poisson_ratio,
    const float *__restrict__ pos
) {
    float total_energy = 0.0f;

    for (size_t i = 0; i<n; i++) {
        const int* t = tets[i];
        auto [pos_i0, pos_i1, pos_i2, pos_i3] = get_pos4(pos, t[0], t[1], t[2], t[3]);
        
        Matrix3f shape_matrix = {
            elementwise_difference(pos_i0, pos_i3),
            elementwise_difference(pos_i1, pos_i3),
            elementwise_difference(pos_i2, pos_i3)
        };
        shape_matrix = transpose(shape_matrix);

        auto [pos0_i0, pos0_i1, pos0_i2, pos0_i3] = get_pos4(pos0, t[0], t[1], t[2], t[3]);

        Matrix3f shape_matrix0 = {
            elementwise_difference(pos0_i0, pos0_i3),
            elementwise_difference(pos0_i1, pos0_i3),
            elementwise_difference(pos0_i2, pos0_i3)
        };
        shape_matrix0 = transpose(shape_matrix0);

        Matrix3f inv_shape_matrix0 = inv(shape_matrix0);

        Matrix3f product = matrix_multiply(shape_matrix, inv_shape_matrix0);

        float energy = neo_hookean(product, youngs_modulus, poisson_ratio);
        total_energy += energy;
    }

    return total_energy;
}

__attribute__((always_inline))
static void gradient_ip(
    const float *__restrict__ pos0,
    const int tets[][4],
    const size_t n,
    const float youngs_modulus,
    const float poisson_ratio,
    const float *__restrict__ pos,
    float *__restrict__ out
    )
{
    __enzyme_autodiff<void>((void *)elastic_energy_simple,
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
static T zero_load(unsigned long long offset, size_t i, std::vector<std::tuple<size_t, size_t, float>> &hess) {
    return T(0);
}


__attribute__((enzyme_sparse_accumulate))
void inner_store(size_t offset, size_t i, float val, std::vector<std::tuple<size_t, size_t, float>> &hess) {
    hess.push_back(std::tuple<size_t, size_t, float>(offset, i, val));
}


template<typename T>
__attribute__((always_inline))
static void csr_store(T val, unsigned long long offset, size_t i, std::vector<std::tuple<size_t, size_t, T>> &hess) {
    if (val == 0.0) return;
    offset /= sizeof(T);
    inner_store(offset, i, val, hess);
}


std::vector<std::tuple<size_t, size_t, float>> hessian(
    const float *__restrict__ pos0,
    const int tets[][4],
    const size_t n,
    const float youngs_modulus,
    const float poisson_ratio,
    const float *__restrict__ pos,
    const size_t num_tets)
{
    std::vector<std::tuple<size_t, size_t, float>> hess;
    for (size_t i=0; i<4*num_tets; i++)
        __enzyme_fwddiff<void>((void *)gradient_ip,
                               enzyme_const, pos0,
                               enzyme_const, tets,
                               enzyme_const, n,
                               enzyme_const, youngs_modulus,
                               enzyme_const, poisson_ratio,
                               enzyme_dup, pos, __enzyme_todense<float*>(ident_load<float>,   err_store<float>, i),
                               enzyme_dupnoneed, nullptr, __enzyme_todense<float*>(zero_load<float>,   csr_store<float>, i, &hess));
    return hess;
}


int main() {
    float pos[] = {
        -0.5, -0.5, -0.5, -0.5
        };
    float pos0[] = {
            0.6760001,  9.438364 ,  2.3247766,  7.3768997
        };

    const size_t n = 1;
    int tets[n][4] = {
        {0, 1, 2, 3},
    };

    float youngs_modulus = 0.1f;
    float poisson_ratio = 0.49f;

    // float result = elastic_energy_simple<n>(pos0, tets, youngs_modulus, poisson_ratio, pos);
    float result = elastic_energy_simple(pos0, tets, n, youngs_modulus, poisson_ratio, pos);

    cout << "Result: " << result << endl;

    // Derivative
    float dpos[sizeof(pos)/sizeof(pos[0])];
    for (size_t i=0; i<sizeof(dpos)/sizeof(dpos[0]); i++)
        dpos[i] = 0;
    gradient_ip(pos0, tets, n, youngs_modulus, poisson_ratio, pos, dpos);

    for (size_t i=0; i<sizeof(dpos)/sizeof(dpos[0]); i++)
        std::cout << "grad_vert[" << i << "]=" << dpos[i] << "\n";

    // Hessian
    const size_t num_tets = 1;
    auto hess_verts = hessian(pos0, tets, n, youngs_modulus, poisson_ratio, pos, num_tets);

    for (auto hess : hess_verts) {
        std::cout << "i=" << std::get<0>(hess) << ", j=" << std::get<1>(hess) << " val=" << std::get<2>(hess) << "\n";
    }

    return 0;
}
