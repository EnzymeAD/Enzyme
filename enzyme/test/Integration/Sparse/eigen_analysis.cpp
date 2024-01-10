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
#include <cmath>
#include <functional>
// #include <Eigen/Dense>
#include <assert.h>
#include <tuple>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>

int enzyme_width;
int enzyme_dup;
int enzyme_dupv;
int enzyme_const;
int enzyme_dupnoneed;
template <typename T>
extern T __enzyme_autodiff(void *, ...);
template <typename T>
extern T __enzyme_fwddiff(void *, ...);
template <typename T>
T __enzyme_todense(...);


template<typename T, size_t n>
__attribute__((always_inline))
static T dot_product(const T a[n], const T b[n]) {
    T result = 0.0;
    for (size_t i = 0; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}


__attribute__((always_inline))
static float norm(const float *__restrict__ v, const size_t n) {
    float sum_squares = 0.0;
    for (size_t i=0; i<n; i++) {
        float val = v[i];
        sum_squares += val * val;
    }
    return std::sqrt(sum_squares);
}


__attribute__((always_inline))
static float area(const float *__restrict__ u, const float *__restrict__ v, const float *__restrict__ w) {
    float cross_product[] = {
        (v[1] - u[1]) * (w[2] - u[2]) - (v[2] - u[2]) * (w[1] - u[1]),
        (v[2] - u[2]) * (w[0] - u[0]) - (v[0] - u[0]) * (w[2] - u[2]),
        (v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0])
    };

    return 0.5 * norm(cross_product, sizeof(cross_product)/sizeof(*cross_product));
}


template<typename T, size_t m, size_t n>
__attribute__((always_inline))
static void transposeMatrix(T (&out)[n][m], const T matrix[m][n]) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            out[j][i] = matrix[i][j];
        }
    }
}


template<typename T, size_t m, size_t n, size_t k>
__attribute__((always_inline))
static void matrixMultiply(T (&result)[m][k], const T matrix1[m][n], const T matrix2[n][k]) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            result[i][j] = 0.0;
            for (int z = 0; z < n; ++z) {
                result[i][j] += matrix1[i][z] * matrix2[z][j];
            }
        }
    }
}


template<typename T>
__attribute__((always_inline))
static void invertMatrix(T (&invertedMatrix)[2][2], const T matrix[2][2]) {
    float determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];

    if (determinant == 0) {
        std::cerr << "Matrix is not invertible (determinant is zero)." << std::endl;
        return;
    }

    float invDeterminant = 1.0 / determinant;
    invertedMatrix[0][0] = matrix[1][1] * invDeterminant;
    invertedMatrix[0][1] = -matrix[0][1] * invDeterminant;
    invertedMatrix[1][0] = -matrix[1][0] * invDeterminant;
    invertedMatrix[1][1] = matrix[0][0] * invDeterminant;
}


template<typename T, size_t m, size_t n>
__attribute__((always_inline))
static void pseudo_inverse(T (&matTsqrinv)[n][m], const T mat[m][n]) {
    T matT[n][m];
    transposeMatrix(matT, mat);
    T matmatT[m][m];
    matrixMultiply(matmatT, mat, matT);
    T sqrinv[m][m];
    invertMatrix(sqrinv, matmatT);
    matrixMultiply(matTsqrinv, matT, sqrinv);
}


__attribute__((always_inline))
static float eigenstuffM(const float *__restrict__ x, size_t n, const int *__restrict__ faces, const float *__restrict__ pos0) {
    float sum = 0;
    for (size_t idx=0; idx<n; idx++) {
        int i = faces[3*idx];
        int j = faces[3*idx+1];
        int k = faces[3*idx+2];
        float tri_area = area(&pos0[3*i], &pos0[3*j], &pos0[3*k]);
        sum += (x[i] * x[i] + x[j] * x[j] + x[k] * x[k]) * (1.0 / 3.0) * tri_area;  // barycentric mass lumping
    }
    return sum;
}


// Calculate total energy for all faces in 3D
__attribute__((always_inline))
static float eigenstuffL(const float *__restrict__ x, size_t num_faces, const int *__restrict__ faces, const float *__restrict__ verts) {
    float sum = 0;
    for (size_t idx=0; idx<num_faces; idx++) {
        int i = faces[3*idx];
        int j = faces[3*idx+1];
        int k = faces[3*idx+2];

        float X[2][3] = {
            {verts[3*j+0] - verts[3*i+0], verts[3*j+1] - verts[3*i+1], verts[3*j+2] - verts[3*i+2]},
            {verts[3*k+0] - verts[3*i+0], verts[3*k+1] - verts[3*i+1], verts[3*k+2] - verts[3*i+2]}
        };

        float pInvX[3][2];
        pseudo_inverse(pInvX, X);
       
        float diffs[] = {x[j] - x[i], x[k] - x[i]};

        float g[3];
        for (int i = 0; i < 3; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < 2; ++j) {
                sum += pInvX[i][j] * diffs[j];
            }
            g[i] = sum;
        }

        sum += dot_product<float, 3>(g, g) * area(&verts[3*i], &verts[3*j], &verts[3*k]);
    }

    return sum;
}



__attribute__((always_inline))
static void gradient_ip(const float *__restrict__ x, const size_t num_faces, const int* faces, const float *__restrict__ pos, float *__restrict__ out)
{
    __enzyme_autodiff<void>((void *)eigenstuffM,
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


std::vector<std::tuple<size_t, size_t, float>> hessian(const float* x, size_t num_faces, const int* faces, const float* pos, size_t num_verts)
{

    std::vector<std::tuple<size_t, size_t, float>> hess;
    for (size_t i=0; i<3*num_verts; i++)
        __enzyme_fwddiff<void>((void *)gradient_ip,
                                enzyme_const, x,
                                enzyme_const, num_faces,
                                enzyme_const, faces,
                                enzyme_dup, pos, __enzyme_todense<float*>(ident_load<float>, err_store<float>, i),
                                enzyme_dupnoneed, nullptr, __enzyme_todense<float*>(zero_load<float>, csr_store<float>, i, &hess));
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
    std::cout << "Result for eigenstuffM_simple: " << resultM << std::endl;

    // Call eigenstuffL_simple
    const float resultL = eigenstuffL(x, num_faces, faces, verts);
    std::cout << "Result for eigenstuffL_simple: " << resultL << std::endl;

    float dverts[sizeof(verts)/sizeof(verts[0])];
    for (size_t i=0; i<sizeof(dverts)/sizeof(dverts[0]); i++)
        dverts[i] = 0;
    gradient_ip(x, num_faces, faces, verts, dverts);

    for (size_t i=0; i<sizeof(dverts)/sizeof(dverts[0]); i++)
        std::cout << "eigenstuffM grad_vert[" << i << "]=" << dverts[i] << "\n";
    
    size_t num_elts = sizeof(verts)/sizeof(verts[0]) * sizeof(verts)/sizeof(verts[0]);

    auto hess_verts = hessian(x, num_faces, faces, verts, num_verts);

    for (auto hess : hess_verts) {
        std::cout << "i=" << std::get<0>(hess) << ", j=" << std::get<1>(hess) << " val=" << std::get<2>(hess) << "\n";
    }

    return 0;
}

