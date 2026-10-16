#include <cmath>
#include <stdio.h>

#include <sys/time.h>
float tdiff(struct timeval *start, struct timeval *end) {
  return (end->tv_sec-start->tv_sec) + 1e-6*(end->tv_usec-start->tv_usec);
}

template<typename T>
struct Triple {
    size_t row;
    size_t col;
    T val;
    Triple(Triple&&) = default;
    Triple(size_t row, size_t col, T val) : row(row), col(col), val(val) {}
};

__attribute__((enzyme_sparse_accumulate))
static void inner_storeflt(size_t row, size_t col, float val, std::vector<Triple<float>> &triplets) {
#ifdef BENCHMARK
    if (val == 0.0) return;
#else
#warning "Compiling for debug/verfication, performance may be slowed"
#endif
    triplets.emplace_back(row, col, val);
}

__attribute__((enzyme_sparse_accumulate))
static void inner_storedbl(size_t row, size_t col, double val, std::vector<Triple<double>> &triplets) {
#ifdef BENCHMARK
    if (val > -1e-10 && val < 1e-10) return;
#else
#warning "Compiling for debug/verfication, performance may be slowed"
#endif
    triplets.emplace_back(row, col, val);
}

template<typename T>
__attribute__((always_inline))
static void sparse_store(T val, size_t idx, size_t i, std::vector<Triple<T>> &triplets) {
    if (val == 0.0) return;
    idx /= sizeof(T);
    if constexpr (sizeof(T) == 4)
      inner_storeflt(i, idx, val, triplets);
    else
      inner_storedbl(i, idx, val, triplets);
}

template<typename T>
__attribute__((always_inline))
static T sparse_load(size_t idx, size_t i, std::vector<Triple<T>> &triplets) {
    return 0.0;
}


__attribute__((enzyme_sparse_accumulate))
static void inner_storeflt_modn(size_t row, size_t col, float val, size_t N, std::vector<Triple<float>> &triplets) {
#ifdef BENCHMARK
    if (val == 0.0) return;
#else
#warning "Compiling for debug/verfication, performance may be slowed"
#endif
    triplets.emplace_back(row, col % N, val);
}

__attribute__((enzyme_sparse_accumulate))
static void inner_storedbl_modn(size_t row, size_t col, double val, size_t N, std::vector<Triple<double>> &triplets) {
#ifdef BENCHMARK
    if (val > -1e-10 && val < 1e-10) return;
#else
#warning "Compiling for debug/verfication, performance may be slowed"
#endif
    triplets.emplace_back(row, col % N, val);
}

template<typename T>
__attribute__((always_inline))
static void sparse_store_modn(T val, size_t idx, size_t i, size_t N, std::vector<Triple<T>> &triplets) {
    if (val == 0.0) return;
    idx /= sizeof(T);
    if constexpr (sizeof(T) == 4)
      inner_storeflt_modn(i, idx, val, N, triplets);
    else
      inner_storedbl_modn(i, idx, val, N, triplets);
}

template<typename T>
__attribute__((always_inline))
static T sparse_load_modn(int64_t idx, size_t i, size_t N, std::vector<Triple<T>> &triplets) {
    return 0.0;
}

template<typename T>
__attribute__((always_inline))
static void ident_store(T, size_t idx, size_t i) {
    assert(0 && "should never load");
}

template<typename T>
__attribute__((always_inline))
static T ident_load(size_t idx, size_t i) {
    idx /= sizeof(T);
    return (T)(idx == i);// ? 1.0 : 0.0;
}

extern int enzyme_width;
extern int enzyme_dup;
extern int enzyme_dupv;
extern int enzyme_const;
extern int enzyme_dupnoneed;

template <typename T, typename... Tys>
extern T __enzyme_autodiff(void*, Tys...) noexcept;

template <typename T, typename... Tys>
extern T __enzyme_fwddiff(void *, Tys...) noexcept;

template <typename T, typename... Tys>
extern T __enzyme_todense(Tys...) noexcept;

template <typename T, typename... Tys>
extern T __enzyme_post_sparse_todense(Tys...) noexcept;

template<typename T, size_t n>
__attribute__((always_inline))
static void elementwise_difference(T (&out)[n], const T x[n], const T y[n]) {
    #pragma clang loop unroll(full)
    for (int i=0; i<n; i++)
        out[i] = x[i] - y[i];
}

template<typename T, size_t n>
__attribute__((always_inline))
static void elementwise_sum(T (&out)[n], const T x[n], const T y[n]) {
    #pragma clang loop unroll(full)
    for (int i=0; i<n; i++)
        out[i] = x[i] + y[i];
}

template<typename T, size_t n>
__attribute__((always_inline))
static T dot_product(const T a[n], const T b[n]) {
    T result = 0.0;
    #pragma clang loop unroll(full)
    for (size_t i = 0; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}


template<typename T, size_t n>
__attribute__((always_inline))
static T norm(const T v[n]) {
    T sum_squares = 0.0;
    #pragma clang loop unroll(full)
    for (size_t i=0; i<n; i++) {
        T val = v[i];
        sum_squares += val * val;
    }
    return std::sqrt(sum_squares);
}

template<typename T, size_t n, size_t m>
__attribute__((always_inline))
static void transpose(T (&out)[n][m], const T in[m][n]) {
    #pragma clang loop unroll(full)
    for (int i=0; i<n; i++)
        #pragma clang loop unroll(full)
        for (int j=0; j<m; j++)
            out[i][j] = in[j][i];
}

template<typename T, size_t m, size_t n, size_t k>
__attribute__((always_inline))
static void matrix_multiply(T (&result)[m][k], const T matrix1[m][n], const T matrix2[n][k]) {
    #pragma clang loop unroll(full)
    for (int i = 0; i < m; ++i) {
        #pragma clang loop unroll(full)
        for (int j = 0; j < k; ++j) {
            result[i][j] = 0.0;
            #pragma clang loop unroll(full)
            for (int z = 0; z < n; ++z) {
                result[i][j] += matrix1[i][z] * matrix2[z][j];
            }
        }
    }
}


template<typename T>
__attribute__((always_inline))
static void inv(T (&out)[3][3], const T (&F)[3][3]) {
    T det = F[0][0] * (F[1][1] * F[2][2] - F[1][2] * F[2][1])
              - F[0][1] * (F[1][0] * F[2][2] - F[1][2] * F[2][0])
              + F[0][2] * (F[1][0] * F[2][1] - F[1][1] * F[2][0]);

    T inv_det = 1 / det;

    out[0][0] = (F[1][1] * F[2][2] - F[1][2] * F[2][1]) * inv_det;
    out[0][1] = (F[0][2] * F[2][1] - F[0][1] * F[2][2]) * inv_det;
    out[0][2] = (F[0][1] * F[1][2] - F[0][2] * F[1][1]) * inv_det;

    out[1][0] = (F[1][2] * F[2][0] - F[1][0] * F[2][2]) * inv_det;
    out[1][1] = (F[0][0] * F[2][2] - F[0][2] * F[2][0]) * inv_det;
    out[1][2] = (F[0][2] * F[1][0] - F[0][0] * F[1][2]) * inv_det;

    out[2][0] = (F[1][0] * F[2][1] - F[1][1] * F[2][0]) * inv_det;
    out[2][1] = (F[0][1] * F[2][0] - F[0][0] * F[2][1]) * inv_det;
    out[2][2] = (F[0][0] * F[1][1] - F[0][1] * F[1][0]) * inv_det;
}


template<typename T>
__attribute__((always_inline))
static void inv(T (&out)[2][2], const T (&F)[2][2]) {
    T det = F[0][0] * F[1][1] - F[0][1] * F[1][0];

    T inv_det = 1 / det;

    out[0][0] = F[1][1] * inv_det;
    out[0][1] = -F[0][1] * inv_det;
    out[1][0] = -F[1][0] * inv_det;
    out[1][1] = F[0][0] * inv_det;
}

template<typename T, size_t m, size_t n>
__attribute__((always_inline))
static void pseudo_inverse(T (&matTsqrinv)[n][m], const T mat[m][n]) {
    T matT[n][m];
    transpose(matT, mat);
    T matmatT[m][m];
    matrix_multiply(matmatT, mat, matT);
    T sqrinv[m][m];
    inv(sqrinv, matmatT);
    matrix_multiply(matTsqrinv, matT, sqrinv);
}

// m is 2 n is 3
template<typename T, int n, int m>
__attribute__((always_inline))
static void get_pos(
    T (&__restrict__ out)[n][m],
    const float *__restrict__ pos,
    const int idx[n]) {

    static_assert(m == 3, "Only Vector3 is supported");

    // extract the 3d points at idx[0], idx[1], idx[2], idx[3]
    #pragma clang loop unroll(full)
    for (int i = 0; i < n; ++i) {
        out[i][0] = pos[m * idx[i]];
        out[i][1] = pos[m * idx[i] + 1];
        out[i][2] = pos[m * idx[i] + 2];
    }
}


// m is 2 n is 3
template<typename T, int n, int m>
__attribute__((always_inline))
static void get_pos_affine(
    T (&__restrict__ out)[n][m],
    const float *__restrict__ pos) {

    static_assert(m == 3, "Only Vector3 is supported");

    // extract the 3d points at idx[0], idx[1], idx[2], idx[3]
    #pragma clang loop unroll(full)
    for (int i = 0; i < n; ++i) {
        out[i][0] = pos[m * i];
        out[i][1] = pos[m * i + 1];
        out[i][2] = pos[m * i + 2];
    }
}

template<typename T>
__attribute__((always_inline))
static void cross(T (&out)[3], const T v1[3], const T v2[3]) {
    out[0] = v1[1] * v2[2] - v1[2] * v2[1];
    out[1] = v1[2] * v2[0] - v1[0] * v2[2];
    out[2] = v1[0] * v2[1] - v1[1] * v2[0];
}


template<typename T>
__attribute__((always_inline))
static T area(const T *__restrict__ u, const T *__restrict__ v, const T *__restrict__ w) {
    T diff1[3];
    elementwise_difference(diff1, v, u);
    T diff2[3];
    elementwise_difference(diff2, w, u);
    T cross_product[3];
    cross(cross_product, diff1, diff2);
    return 0.5 * norm<T, 3>(cross_product);
}
