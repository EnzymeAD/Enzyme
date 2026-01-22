// This should work on LLVM 7, 8, 9, however in CI the version of clang installed on Ubuntu 18.04 cannot load
// a clang plugin properly without segfaulting on exit. This is fine on Ubuntu 20.04 or later LLVM versions...
// TODO: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions  -ffast-math -mllvm -enable-load-pre=0 -std=c++11 -O1 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-auto-sparsity=0 | %lli - ; fi
// TODO: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions  -ffast-math -mllvm -enable-load-pre=0 -std=c++11 -O2 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-auto-sparsity=0  | %lli - ; fi
// TODO: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions  -ffast-math -mllvm -enable-load-pre=0 -std=c++11 -O3 %s -S -emit-llvm -o - %loadClangEnzyme  -mllvm -enzyme-auto-sparsity=0 | %lli - ; fi
// TODO: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -ffast-math -mllvm -enable-load-pre=0  -std=c++11 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-auto-sparsity=1 -S | %lli - ; fi
// TODO: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -ffast-math -mllvm -enable-load-pre=0  -std=c++11 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-auto-sparsity=1 -S | %lli - ; fi
// TODO: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -ffast-math -mllvm -enable-load-pre=0  -std=c++11 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-auto-sparsity=1 -S | %lli - ; fi

#include <vector>
#include <array>
#include <cmath>
#include <cassert>
#include <tuple>

#include "matrix.h"

template<typename T, size_t n>
__attribute__((always_inline))
static float length(const T v1[n], const T v2[n]) {
    T diff[n];
    elementwise_difference(diff, v2, v1);
    return norm<T, n>(diff);
}


template<typename T, size_t n>
__attribute__((always_inline))
std::tuple<T, T> dihedral_angle(const T (&v1)[n], const T (&v2)[n], const T (&v3)[n], const T (&v4)[n]) {
    T diff13[n];
    elementwise_difference(diff13, v1, v3);
    T diff23[n];
    elementwise_difference(diff23, v2, v3);
    T diff14[n];
    elementwise_difference(diff14, v1, v4);
    T diff24[n];
    elementwise_difference(diff24, v2, v4);
    T n1[n];
    cross<T>(n1, diff13, diff23);
    T n2[n];
    cross<T>(n2, diff14, diff24);
    T a1 = norm<T, n>(n1);
    T a2 = norm<T, n>(n2);
    T ev[n];
    elementwise_difference(ev, v2, v1);
    T l2 = dot_product<T, n>(ev, ev);
    T diff_n1_n2[n];
    elementwise_difference(diff_n1_n2, n1, n2);
    T sum_n1_n2[n];
    elementwise_sum(sum_n1_n2, n1, n2);
    T theta = M_PI - 2 * std::atan2(norm<T, n>(diff_n1_n2), norm<T, n>(sum_n1_n2));
    T w = 3 * l2 / (a1 + a2);
    return std::make_tuple(theta, w);
}

template<typename T>
__attribute__((always_inline))
std::tuple<T, T> dihedral_angle3(const T* __restrict__ v1, const T* __restrict__ v2, const T* __restrict__ v3, const T* __restrict__ v4) {
    const int n = 3;
    T diff13[n];
    elementwise_difference(diff13, v1, v3);
    T diff23[n];
    elementwise_difference(diff23, v2, v3);
    T diff14[n];
    elementwise_difference(diff14, v1, v4);
    T diff24[n];
    elementwise_difference(diff24, v2, v4);
    T n1[n];
    cross<T>(n1, diff13, diff23);
    T n2[n];
    cross<T>(n2, diff14, diff24);
    T a1 = norm<T, n>(n1);
    T a2 = norm<T, n>(n2);
    T ev[n];
    elementwise_difference(ev, v2, v1);
    T l2 = dot_product<T, n>(ev, ev);
    T diff_n1_n2[n];
    elementwise_difference(diff_n1_n2, n1, n2);
    T sum_n1_n2[n];
    elementwise_sum(sum_n1_n2, n1, n2);
    T theta = M_PI - 2 * std::atan2(norm<T, n>(diff_n1_n2), norm<T, n>(sum_n1_n2));
    T w = 3 * l2 / (a1 + a2);
    return std::make_tuple(theta, w);
}

template<typename T>
__attribute__((always_inline))
static T pos_load(unsigned long long offset, T* pos, const int* edges) {
    offset /= sizeof(T);

    int idx = offset / 3;

    int inc = offset % 3;

    int idx0 = edges[3*idx];

    /// pos_data[0:3] -> pos[3*edges[i]:3*edges[i]+3]

    return pos[3*idx0+inc];
}

template<typename T>
__attribute__((always_inline))
static float edge_energy(const T *__restrict__ pos, const T *__restrict__ pos0, const int i0, const int i1, const T edge_coefficient) {
    using Vector = T[3];

    const int idxs[2] = {i0, i1};

    //Vector pos_i[2];
    //get_pos(pos_i, pos, idxs);

    const T* pos_data0 = &pos[3 * i0];
    const T* pos_data1 = &pos[3 * i1];

    float l = length<T, 3>(pos_data0, pos_data1);

    Vector pos0_i[2];
    get_pos(pos0_i, pos0, idxs);

    float l0 = length<T, 3>(pos0_i[0], pos0_i[1]);

    float diff = 1 - l / l0;
    return diff * diff * l0 * edge_coefficient;
}

template<typename T>
__attribute__((always_inline))
static T face_energy(
    const T * __restrict__ pos,
    const T *__restrict__ pos0,
    const int i0,
    const int i1,
    const int i2,
    const T face_coefficient,
    const int num_edges
) {
    using Vector = T[3];
    const int idxs[3] = {i0, i1, i2};

    //Vector pos_i[3];
    //get_pos(pos_i, pos, idxs);
    const int base_offset = 3 * num_edges;
    const T* pos_data0 = &pos[base_offset + 3 * i0];
    const T* pos_data1 = &pos[base_offset + 3 * i1];
    const T* pos_data2 = &pos[base_offset + 3 * i2];

    T a = area(pos_data0, pos_data1, pos_data2);

    Vector pos0_i[3];
    get_pos(pos0_i, pos0, idxs);

    T a0 = area(pos0_i[0], pos0_i[1], pos0_i[2]);
    T diff = 1 - a / a0;
    return diff * diff * a0 * face_coefficient;
}

template<typename T>
__attribute__((always_inline))
static T flap_energy(
    const T *__restrict__ pos,
    const T *__restrict__ pos0,
    const int i0,
    const int i1,
    const int i2,
    const int i3,
    const T bending_stiffness,
    const int num_edges,
    const int num_faces
) {
    using Vector = T[3];
    const int idxs[4] = {i0, i1, i2, i3};

    //Vector pos_i[4];
    //get_pos(pos_i, pos, idxs);

    const int base_offset = 3 * num_edges + 3 * num_faces;
    const T* pos_data0 = &pos[base_offset + 3 * i0];
    const T* pos_data1 = &pos[base_offset + 3 * i1];
    const T* pos_data2 = &pos[base_offset + 3 * i2];
    const T* pos_data3 = &pos[base_offset + 3 * i3];

    T theta, w;
    std::tie(theta, w) = dihedral_angle3(pos_data0, pos_data1, pos_data2, pos_data3);

    Vector pos0_i[4];
    get_pos(pos0_i, pos0, idxs);

    T theta0, w0;
    std::tie(theta0, w0) = dihedral_angle(pos0_i[0], pos0_i[1], pos0_i[2], pos0_i[3]);
    T diff = theta - theta0;
    return diff * diff * w0 * bending_stiffness;
}

template<typename T>
__attribute__((always_inline))
static T discrete_shell_simple_edges(
    const T *__restrict__ pos0,
    const int *__restrict__ edges,
    const int num_edges,
    const int *__restrict__ faces,
    const int num_faces,
    const int *__restrict__ flaps,
    const int num_flaps,
    const T edge_coefficient,
    const T face_coefficient,
    const T bending_stiffness,
    const T *__restrict__  pos
) {
    T total_energy = 0;
    __builtin_assume(num_edges != 0);
    for (int i = 0; i < num_edges; i++) {

        total_energy += edge_energy(pos, pos0, edges[2 * i], edges[2 * i + 1], edge_coefficient);
    }

    return total_energy;
}

template<typename T>
__attribute__((always_inline))
static T discrete_shell_simple_faces(
    const T *__restrict__ pos0,
    const int *__restrict__ edges,
    const int num_edges,
    const int *__restrict__ faces,
    const int num_faces,
    const int *__restrict__ flaps,
    const int num_flaps,
    const T edge_coefficient,
    const T face_coefficient,
    const T bending_stiffness,
    const T *__restrict__  pos
) {
    T total_energy = 0;
    __builtin_assume(num_faces != 0);
    for (int i = 0; i < num_faces; i++) {
        total_energy += face_energy(pos, pos0, faces[3 * i], faces[3 * i + 1], faces[3 * i + 2], face_coefficient, num_edges);
    }

    return total_energy;
}

template<typename T>
__attribute__((always_inline))
static T discrete_shell_simple_flaps(
    const T *__restrict__ pos0,
    const int *__restrict__ edges,
    const int num_edges,
    const int *__restrict__ faces,
    const int num_faces,
    const int *__restrict__ flaps,
    const int num_flaps,
    const T edge_coefficient,
    const T face_coefficient,
    const T bending_stiffness,
    const T *__restrict__  pos
) {
    T total_energy = 0;
    __builtin_assume(num_flaps != 0);
    for (int i = 0; i < num_flaps; i++) {
        total_energy += flap_energy(pos, pos0, flaps[4 * i], flaps[4 * i + 1], flaps[4 * i + 2], flaps[4 * i + 3], bending_stiffness, num_edges, num_faces);
    }

    return total_energy;
}

template<typename T>
__attribute__((always_inline))
static void gradient_ip_edges(
    const T *__restrict__ pos0,
    const int *__restrict__ edges,
    const int num_edges,
    const int *__restrict__ faces,
    const int num_faces,
    const int *__restrict__ flaps,
    const int num_flaps,
    const T edge_coefficient,
    const T face_coefficient,
    const T bending_stiffness,
    const T *__restrict__  pos,
    T *__restrict__ out
    )
{
    __enzyme_autodiff<void>((void *)discrete_shell_simple_edges<T>,
                            enzyme_const, pos0,
                            enzyme_const, edges,
                            enzyme_const, num_edges,
                            enzyme_const, faces,
                            enzyme_const, num_faces,
                            enzyme_const, flaps,
                            enzyme_const, num_flaps,
                            enzyme_const, edge_coefficient,
                            enzyme_const, face_coefficient,
                            enzyme_const, bending_stiffness,
                            enzyme_dup, pos,
                            out);
}

template<typename T>
__attribute__((always_inline))
static void gradient_ip_faces(
    const T *__restrict__ pos0,
    const int *__restrict__ edges,
    const int num_edges,
    const int *__restrict__ faces,
    const int num_faces,
    const int *__restrict__ flaps,
    const int num_flaps,
    const T edge_coefficient,
    const T face_coefficient,
    const T bending_stiffness,
    const T *__restrict__  pos,
    T *__restrict__ out
    )
{
    __enzyme_autodiff<void>((void *)discrete_shell_simple_faces<T>,
                            enzyme_const, pos0,
                            enzyme_const, edges,
                            enzyme_const, num_edges,
                            enzyme_const, faces,
                            enzyme_const, num_faces,
                            enzyme_const, flaps,
                            enzyme_const, num_flaps,
                            enzyme_const, edge_coefficient,
                            enzyme_const, face_coefficient,
                            enzyme_const, bending_stiffness,
                            enzyme_dup, pos,
                            out);
}

template<typename T>
__attribute__((always_inline))
static void gradient_ip_flaps(
    const T *__restrict__ pos0,
    const int *__restrict__ edges,
    const int num_edges,
    const int *__restrict__ faces,
    const int num_faces,
    const int *__restrict__ flaps,
    const int num_flaps,
    const T edge_coefficient,
    const T face_coefficient,
    const T bending_stiffness,
    const T *__restrict__  pos,
    T *__restrict__ out
    )
{
    __enzyme_autodiff<void>((void *)discrete_shell_simple_flaps<T>,
                            enzyme_const, pos0,
                            enzyme_const, edges,
                            enzyme_const, num_edges,
                            enzyme_const, faces,
                            enzyme_const, num_faces,
                            enzyme_const, flaps,
                            enzyme_const, num_flaps,
                            enzyme_const, edge_coefficient,
                            enzyme_const, face_coefficient,
                            enzyme_const, bending_stiffness,
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
    const int *__restrict__ edges,
    const int num_edges,
    const int *__restrict__ faces,
    const int num_faces,
    const int *__restrict__ flaps,
    const int num_flaps,
    const T edge_coefficient,
    const T face_coefficient,
    const T bending_stiffness,
    const T *__restrict__  pos,
    const size_t num_verts)
{

    std::vector<Triple<T>> hess;

    T* pos2 = __enzyme_post_sparse_todense<T*>(pos_load<T>, err_store<T>, pos);

    __builtin_assume(num_verts != 0);
    for (size_t i=0; i<3*num_verts; i++)
        __enzyme_fwddiff<void>((void *)gradient_ip_edges<T>,
                               enzyme_const, pos0,
                               enzyme_const, edges,
                               enzyme_const, num_edges,
                               enzyme_const, faces,
                               enzyme_const, num_faces,
                               enzyme_const, flaps,
                               enzyme_const, num_flaps,
                               enzyme_const, edge_coefficient,
                               enzyme_const, face_coefficient,
                               enzyme_const, bending_stiffness,
                               enzyme_dup, pos2, __enzyme_todense<T*>(ident_load<T>,   err_store<T>, i),
                               enzyme_dupnoneed, nullptr, __enzyme_todense<T*>(zero_load<T>,   csr_store<T>, i, &hess));
    
    for (size_t i=0; i<3*num_verts; i++)
        __enzyme_fwddiff<void>((void *)gradient_ip_faces<T>,
                               enzyme_const, pos0,
                               enzyme_const, edges,
                               enzyme_const, num_edges,
                               enzyme_const, faces,
                               enzyme_const, num_faces,
                               enzyme_const, flaps,
                               enzyme_const, num_flaps,
                               enzyme_const, edge_coefficient,
                               enzyme_const, face_coefficient,
                               enzyme_const, bending_stiffness,
                               enzyme_dup, pos2, __enzyme_todense<T*>(ident_load<T>,   err_store<T>, i),
                               enzyme_dupnoneed, nullptr, __enzyme_todense<T*>(zero_load<T>,   csr_store<T>, i, &hess));
    

    for (size_t i=0; i<3*num_verts; i++)
        __enzyme_fwddiff<void>((void *)gradient_ip_flaps<T>,
                               enzyme_const, pos0,
                               enzyme_const, edges,
                               enzyme_const, num_edges,
                               enzyme_const, faces,
                               enzyme_const, num_faces,
                               enzyme_const, flaps,
                               enzyme_const, num_flaps,
                               enzyme_const, edge_coefficient,
                               enzyme_const, face_coefficient,
                               enzyme_const, bending_stiffness,
                               enzyme_dup, pos, __enzyme_todense<T*>(ident_load<T>,   err_store<T>, i),
                               enzyme_dupnoneed, nullptr, __enzyme_todense<T*>(zero_load<T>,   csr_store<T>, i, &hess));
    

    return hess;
}



int main() {
    const float pos[] = {
        -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, -0.5
        };
    const float pos0[] = {
            0.6760001,  9.438364 ,  2.3247766,  7.3768997,  2.6186442, -5.9123445,
            4.8309755, -6.994741 , -5.6606984, -3.5013247,  4.7117257,  5.850687,
            -2.569952,  -7.387514,  -9.032347
        };

    const size_t num_edges = 1;
    const size_t num_faces = 1;
    const size_t num_flaps = 1;
    int edges[2 * num_edges] = {
        1, 2,
    };
    int faces[3 * num_faces] = {
        1, 2, 3,
    };
    int flaps[4 * num_flaps] = {
        1, 2, 3, 4,
    };

    const float edge_coefficient = 0.5f;
    const float face_coefficient = 0.1f;
    const float bending_stiffness = 0.1f;

    float result = discrete_shell_simple_edges(pos0, edges, num_edges, faces, num_faces, flaps, num_flaps, edge_coefficient, face_coefficient, bending_stiffness, pos);
    result += discrete_shell_simple_faces(pos0, edges, num_edges, faces, num_faces, flaps, num_flaps, edge_coefficient, face_coefficient, bending_stiffness, pos);
    result += discrete_shell_simple_flaps(pos0, edges, num_edges, faces, num_faces, flaps, num_flaps, edge_coefficient, face_coefficient, bending_stiffness, pos);

    printf("Result: %f\n", result);

    // Derivative
    float dpos[sizeof(pos)/sizeof(pos[0])];
    for (size_t i=0; i<sizeof(dpos)/sizeof(dpos[0]); i++)
        dpos[i] = 0;
    gradient_ip_edges(pos0, edges, num_edges, faces, num_faces, flaps, num_flaps, edge_coefficient, face_coefficient, bending_stiffness, pos, dpos);
    gradient_ip_faces(pos0, edges, num_edges, faces, num_faces, flaps, num_flaps, edge_coefficient, face_coefficient, bending_stiffness, pos, dpos);
    gradient_ip_flaps(pos0, edges, num_edges, faces, num_faces, flaps, num_flaps, edge_coefficient, face_coefficient, bending_stiffness, pos, dpos);

    for (size_t i=0; i<sizeof(dpos)/sizeof(dpos[0]); i++)
        printf("grad_vert[%zu] = %f\n", i, dpos[i]);

    // Hessian
    const size_t num_verts = 4;
    auto hess_verts = hessian(pos0, edges, num_edges, faces, num_faces, flaps, num_flaps, edge_coefficient, face_coefficient, bending_stiffness, pos, num_verts);

    for (auto &hess : hess_verts) {
        printf("i=%lu, j=%lu, val=%f\n", hess.row, hess.col, hess.val);
    }

    return 0;
}
