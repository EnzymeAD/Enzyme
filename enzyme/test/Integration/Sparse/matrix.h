extern int enzyme_width;
extern int enzyme_dup;
extern int enzyme_dupv;
extern int enzyme_const;
extern int enzyme_dupnoneed;

template <typename T, typename... Tys>
extern T __enzyme_autodiff(void*, Tys...);

template <typename T, typename... Tys>
extern T __enzyme_fwddiff(void *, Tys...);

template <typename T, typename... Tys>
extern T __enzyme_todense(Tys...);

template<typename T, size_t n>
__attribute__((always_inline))
static void elementwise_difference(T (&out)[n], const T x[n], const T y[n]) {
    #pragma clang loop unroll(full)
    for (int i=0; i<n; i++)
        out[i] = x[i] - y[i];
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


template<typename T>
__attribute__((always_inline))
static void inv(T (&out)[3][3], const T (&F)[3][3]) {
    float det = F[0][0] * (F[1][1] * F[2][2] - F[1][2] * F[2][1])
              - F[0][1] * (F[1][0] * F[2][2] - F[1][2] * F[2][0])
              + F[0][2] * (F[1][0] * F[2][1] - F[1][1] * F[2][0]);

    float inv_det = 1 / det;

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


template<typename T, int n, int m>
__attribute__((always_inline))
static void get_pos(
    T (&__restrict__ out)[n][m],
    const float *__restrict__ pos,
    const int idx[n]) {

    // extract the 3d points at idx[0], idx[1], idx[2], idx[3]
    #pragma clang loop unroll(full)
    for (int i = 0; i < n; ++i) {
        #pragma clang loop unroll(full)
        for (int j = 0; j < m; ++j) {
            out[i][j] = pos[m * idx[i] + i];
        }
    }
}