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
static T face_load(unsigned long long offset, T* x, const int* faces) {
    offset /= sizeof(T);
    return x[faces[offset]];
}

template<typename T>
__attribute__((always_inline))
static void face_store(unsigned long long offset, T* x, const int* faces) {
    assert(0 && "store is not legal");
}


template<typename T>
__attribute__((always_inline))
static T area_load(unsigned long long offset, T* pos0, const int* faces) {
    offset /= sizeof(T);

    int idx = offset / 9;

    int inc = offset % 9;

    int i = faces[3*idx];
    int j = faces[3*idx+1];
    int k = faces[3*idx+2];

    /// pos_data[0:3] -> pos[3*faces[i]:3*faces[i]+3]
    /// pos_data[3:6] -> pos[3*faces[j]:3*faces[j]+3]
    /// pos_data[6:9] -> pos[3*faces[k]:3*faces[k]+3]

    if (inc < 3) {
        return pos0[3*i+inc];
    } else if (inc < 6) {
        return pos0[3*j+inc-3];
    } else {
        return pos0[3*k+inc-6];
    }
}

template<typename T>
__attribute__((always_inline))
static void area_store(unsigned long long offset, T* pos0, const int* faces) {
    assert(0 && "store is not legal");
}

template<typename T>
__attribute__((always_inline))
static T eigenstuffM(const T *__restrict__ pos0, size_t n, const int *__restrict__ faces, const T *__restrict__ x) {
    T sum = 0;
    __builtin_assume(n != 0);
    for (size_t idx=0; idx<n; idx++) {
        int i = faces[3*idx];
        int j = faces[3*idx+1];
        int k = faces[3*idx+2];

/*
        T xi = x[i];
        T xj = x[j];
        T xk = x[k];
        */

        T xi = x[3 * idx];  /// x[i] -> real_x[faces[i]] 
        T xj = x[3 * idx + 1];
        T xk = x[3 * idx + 2];

        const T* pos_data = &pos0[9 * idx];
        /// 
        /// pos_data[0:3] -> pos[3*faces[i]:3*faces[i]+3]
        /// pos_data[3:6] -> pos[3*faces[j]:3*faces[j]+3]
        /// pos_data[6:9] -> pos[3*faces[k]:3*faces[k]+3]

        T tri_area = area(&pos_data[0], &pos_data[3], &pos_data[6]);

        sum += (xi * xi + xj * xj + xk * xk) * (1.0 / 3.0) * tri_area;  // barycentric mass lumping
    }
    return sum;
}


// Calculate total energy for all faces in 3D
template<typename T>
__attribute__((always_inline))
static T eigenstuffL(const T *__restrict__ x, size_t num_faces, const int *__restrict__ faces, const T *__restrict__ pos0) {
    T sum = 0;
    __builtin_assume(num_faces != 0);
    for (size_t idx=0; idx<num_faces; idx++) {
        int i = faces[3*idx];
        int j = faces[3*idx+1];
        int k = faces[3*idx+2];

        T X[2][3] = {
            {   pos0[3*j+0] - pos0[3*i+0],
                pos0[3*j+1] - pos0[3*i+1],
                pos0[3*j+2] - pos0[3*i+2]},
            {pos0[3*k+0] - pos0[3*i+0], pos0[3*k+1] - pos0[3*i+1], pos0[3*k+2] - pos0[3*i+2]}
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

        sum += dot_product<T, 3>(g, g) * area(&pos0[3*i], &pos0[3*j], &pos0[3*k]);
    }

    return sum;
}


template<typename T>
__attribute__((always_inline))
static void gradient_ip(const T *__restrict__ pos0, const size_t num_faces, const int* faces, const T *__restrict__ x, T *__restrict__ out)
{
    __enzyme_autodiff<void>((void *)eigenstuffM<T>,
                            enzyme_const, pos0,
                            enzyme_const, num_faces,
                            enzyme_const, faces,
                            enzyme_dup, x, out);
}

template<typename T>
__attribute__((noinline))
std::vector<Triple<T>> hessian(const T*__restrict__ pos0, size_t num_faces, const int* faces, const T*__restrict__ x, size_t x_pts)
{    
    float* x2 = __enzyme_post_sparse_todense<float*>(face_load<float>, face_store<float>, x, faces);

    /*
    float* x3 = (float*)malloc(sizeof(float)*9*num_faces);
    for (size_t idx=0; idx<num_faces; idx++) {
        int i = faces[3*idx];
        int j = faces[3*idx+1];
        int k = faces[3*idx+2];
        x3[idx * 9 + 0] = pos[3*i+0];
        x3[idx * 9 + 1] = pos[3*i+1];
        x3[idx * 9 + 2] = pos[3*i+2];
        x3[idx * 9 + 3] = pos[3*j+0];
        x3[idx * 9 + 4] = pos[3*j+1];
        x3[idx * 9 + 5] = pos[3*j+2];
        x3[idx * 9 + 6] = pos[3*k+0];
        x3[idx * 9 + 7] = pos[3*k+1];
        x3[idx * 9 + 8] = pos[3*k+2];
    }
    */

    float* pos02 = __enzyme_post_sparse_todense<float*>(area_load<float>, area_store<float>, pos0, faces);
    std::vector<Triple<T>> hess;
    __builtin_assume(x_pts != 0);
    for (size_t i=0; i<3*x_pts; i++)
        __enzyme_fwddiff<void>((void *)gradient_ip<T>,
                                enzyme_const, pos02,
                                enzyme_const, num_faces,
                                enzyme_const, faces,
                                enzyme_dup, x2, __enzyme_todense<T*>(ident_load<T>, ident_store<T>, i),
                                enzyme_dupnoneed, nullptr, __enzyme_todense<T*>(sparse_load<T>, sparse_store<T>, i, &hess));
    return hess;
}

// Check the sparse Hessian against a dense one. The Hessian is taken with
// respect to the per-face slots x2[s] = x[faces[s]] (see face_load), with the
// positions of each face expanded as in area_load. Compute it with plain
// arrays, without __enzyme_todense, so that -enzyme-auto-sparsity does not
// apply to it.
template <typename T>
static void check_hessian(const std::vector<Triple<T>> &hess, const T *pos0,
                          size_t num_faces, const int *faces, const T *x,
                          size_t seeds) {
  size_t n = 3 * num_faces;
  T *xs = (T *)malloc(sizeof(T) * n);
  T *ps = (T *)malloc(sizeof(T) * 9 * num_faces);
  for (size_t s = 0; s < n; s++)
    xs[s] = x[faces[s]];
  for (size_t idx = 0; idx < num_faces; idx++)
    for (size_t inc = 0; inc < 9; inc++)
      ps[9 * idx + inc] = pos0[3 * faces[3 * idx + inc / 3] + inc % 3];

  size_t R = seeds > n ? seeds : n;
  double *S = (double *)calloc(R * n, sizeof(double));
  double *H = (double *)calloc(R * n, sizeof(double));
  int bad = 0;
  for (auto &t : hess) {
    if (t.row >= R || t.col >= n) {
      printf("out of range: i=%lu, j=%lu, val=%f\n", t.row, t.col, t.val);
      bad++;
      continue;
    }
    S[t.row * n + t.col] += t.val;
  }
  T *dx = (T *)malloc(sizeof(T) * n);
  T *g = (T *)malloc(sizeof(T) * n);
  T *dg = (T *)malloc(sizeof(T) * n);
  for (size_t r = 0; r < n; r++) {
    for (size_t c = 0; c < n; c++) {
      dx[c] = (c == r);
      g[c] = 0;
      dg[c] = 0;
    }
    __enzyme_fwddiff<void>((void *)gradient_ip<T>, enzyme_const, ps,
                           enzyme_const, num_faces, enzyme_const, faces,
                           enzyme_dup, xs, dx, enzyme_dup, g, dg);
    for (size_t c = 0; c < n; c++)
      H[r * n + c] = dg[c];
  }
  for (size_t i = 0; i < R * n; i++)
    if (fabs(S[i] - H[i]) > 1e-4 * (1 + fabs(H[i]))) {
      printf("mismatch i=%zu, j=%zu: sparse %f dense %f\n", i / n, i % n,
             S[i], H[i]);
      bad++;
    }
  if (bad) {
    printf("%d mismatches\n", bad);
    exit(1);
  }
  printf("Hessian of %zu faces matches the dense reference\n", num_faces);
}

int main(int argc, char** argv) {
    size_t x_pts = 8;

    if (argc >= 2) {
         x_pts = atoi(argv[1]);
    }

    // TODO generate data for more inputs
    assert(x_pts == 8);
    const float x[] = {0.0, 1.0, 0.0};


    const size_t num_faces = 1;
    const int faces[] = {0, 1, 2};

    const float pos0[] = {1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0, 1.0, 3.0};

    // Call eigenstuffM_simple
    struct timeval start, end;
    gettimeofday(&start, NULL);
    const float resultM = eigenstuffM(pos0, num_faces, faces, x);
    gettimeofday(&end, NULL);
    printf("Result for eigenstuffM_simple: %f, runtime:%f\n", resultM, tdiff(&start, &end));

    // Call eigenstuffL_simple
    gettimeofday(&start, NULL);
    const float resultL = eigenstuffL(x, num_faces, faces, pos0);
    gettimeofday(&end, NULL);
    printf("Result for eigenstuffL_simple: %f, runtime:%f\n", resultL, tdiff(&start, &end));

    float dx[sizeof(x)/sizeof(x[0])];
    for (size_t i=0; i<sizeof(dx)/sizeof(x[0]); i++)
        dx[i] = 0;
    gradient_ip(pos0, num_faces, faces, x, dx);

    if (x_pts < 30) {
    for (size_t i=0; i<sizeof(dx)/sizeof(dx[0]); i++)
        printf("eigenstuffM grad_vert[%zu]=%f\n", i, dx[i]);
    }

    gettimeofday(&start, NULL);
    auto hess_x = hessian(pos0, num_faces, faces, x, x_pts);
    gettimeofday(&end, NULL);

    printf("Number of elements %ld\n", hess_x.size());
  
    printf("Runtime %0.6f\n", tdiff(&start, &end));

    if (x_pts <= 8)
    for (auto &hess : hess_x) {
        printf("i=%lu, j=%lu, val=%f\n", hess.row, hess.col, hess.val);
    }
    check_hessian(hess_x, pos0, num_faces, faces, x, 3 * x_pts);

    {
        // Four faces of a tetrahedron-like surface over five vertices, with
        // shared vertices and different areas.
        const size_t mesh_faces = 4;
        const int mesh[] = {0, 1, 2, 0, 2, 3, 1, 3, 4, 2, 4, 0};
        const float mesh_x[] = {0.5, 1.0, -2.0, 3.0, 0.25};
        const float mesh_pos[] = {0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0, 3.0, 0.5,
                                  1.0, 1.0, 4.0, 3.0, 2.0, -1.0};
        auto hess_mesh = hessian(mesh_pos, mesh_faces, mesh, mesh_x, mesh_faces);
        printf("Number of elements %ld\n", hess_mesh.size());
        check_hessian(hess_mesh, mesh_pos, mesh_faces, mesh, mesh_x,
                      3 * mesh_faces);
    }

    return 0;
}

