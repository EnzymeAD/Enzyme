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
extern T __enzyme_autodiff(void *,
                           int, const float*,  // pos0
                           int, const int*,   // edges
                           int, const int,  // num_edges
                           int, const int*, // faces
                           int, const int, // num_faces
                           int, const int*, // flaps
                           int, const int, // num_flaps
                           int, const float, // edge_coefficient
                           int, const float, // face_coefficient
                           int, const float, // bending_stiffness
                           int, const float*, // pos
                           float*);
template <typename T>
extern T __enzyme_fwddiff(void *,
                            int, const float*,  // pos0
                            int, const int*,   // edges
                            int, const int,  // num_edges
                            int, const int*, // faces
                            int, const int, // num_faces
                            int, const int*, // flaps
                            int, const int, // num_flaps
                            int, const float, // edge_coefficient
                            int, const float, // face_coefficient
                            int, const float, // bending_stiffness
                            int, const float*, // pos
                            ...
                            );
template <typename T>
T __enzyme_todense(...);

using Vector3f = std::array<float, 3>;


std::tuple<Vector3f, Vector3f> get_pos2(
    const float *__restrict__ pos,
    int i0,
    int i1
) {
    Vector3f pos_i0 = {pos[3 * i0], pos[3 * i0 + 1], pos[3 * i0 + 2]};
    Vector3f pos_i1 = {pos[3 * i1], pos[3 * i1 + 1], pos[3 * i1 + 2]};
    return std::make_tuple(pos_i0, pos_i1);
}


std::tuple<Vector3f, Vector3f, Vector3f> get_pos3(
    const float *__restrict__ pos,
    int i0,
    int i1,
    int i2
) {
    Vector3f pos_i0 = {pos[3 * i0], pos[3 * i0 + 1], pos[3 * i0 + 2]};
    Vector3f pos_i1 = {pos[3 * i1], pos[3 * i1 + 1], pos[3 * i1 + 2]};
    Vector3f pos_i2 = {pos[3 * i2], pos[3 * i2 + 1], pos[3 * i2 + 2]};
    return std::make_tuple(pos_i0, pos_i1, pos_i2);
}


std::tuple<Vector3f, Vector3f, Vector3f, Vector3f> get_pos4(
    const float *__restrict__ pos,
    int i0,
    int i1,
    int i2,
    int i3
) {
    Vector3f pos_i0 = {pos[3 * i0], pos[3 * i0 + 1], pos[3 * i0 + 2]};
    Vector3f pos_i1 = {pos[3 * i1], pos[3 * i1 + 1], pos[3 * i1 + 2]};
    Vector3f pos_i2 = {pos[3 * i2], pos[3 * i2 + 1], pos[3 * i2 + 2]};
    Vector3f pos_i3 = {pos[3 * i3], pos[3 * i3 + 1], pos[3 * i3 + 2]};
    return std::make_tuple(pos_i0, pos_i1, pos_i2, pos_i3);
}


Vector3f elementwise_difference(const Vector3f& v1, const Vector3f& v2) {
    return {v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]};
}


Vector3f elementwise_sum(const Vector3f& v1, const Vector3f& v2) {
    return {v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]};
}


float norm(const Vector3f& v) {
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}


float length(const Vector3f& v1, const Vector3f& v2) {
    return norm(elementwise_difference(v2, v1));
}


Vector3f cross(const Vector3f& v1, const Vector3f& v2) {
    return {
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0]
    };
}


float area(const Vector3f& v1, const Vector3f& v2, const Vector3f& v3) {
    Vector3f diff1 = elementwise_difference(v2, v1);
    Vector3f diff2 = elementwise_difference(v3, v1);
    Vector3f cp = cross(diff1, diff2);
    return 0.5 * norm(cp);
}


float dot(const Vector3f& v1, const Vector3f& v2) {
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}


std::tuple<float, float> dihedral_angle(const Vector3f& v1, const Vector3f& v2, const Vector3f& v3, const Vector3f& v4) {
    Vector3f n1 = cross(elementwise_difference(v1, v3), elementwise_difference(v2, v3));
    Vector3f n2 = cross(elementwise_difference(v2, v4), elementwise_difference(v1, v4));
    float a1 = norm(n1);
    float a2 = norm(n2);
    Vector3f ev = elementwise_difference(v2, v1);
    float l2 = dot(ev, ev);
    Vector3f diff_n1_n2 = elementwise_difference(n1, n2);
    Vector3f sum_n1_n2 = elementwise_sum(n1, n2);
    float theta = M_PI - 2 * std::atan2(norm(diff_n1_n2), norm(sum_n1_n2));
    float w = 3 * l2 / (a1 + a2);
    return std::make_tuple(theta, w);
}

__attribute__((always_inline))
static float edge_energy(const float *__restrict__ pos, const float *__restrict__ pos0, const int i0, const int i1, const float edge_coefficient) {
    Vector3f pos_i0, pos_i1;
    std::tie(pos_i0, pos_i1) = get_pos2(pos, i0, i1);
    float l = length(pos_i0, pos_i1);
    Vector3f pos0_i0, pos0_i1;
    std::tie(pos0_i0, pos0_i1) = get_pos2(pos0, i0, i1);
    float l0 = length(pos0_i0, pos0_i1);
    float diff = 1 - l / l0;
    return diff * diff * l0 * edge_coefficient;
}

__attribute__((always_inline))
static float face_energy(
    const float * __restrict__ pos,
    const float *__restrict__ pos0,
    const int i0,
    const int i1,
    const int i2,
    const float face_coefficient
) {
    Vector3f pos_i0, pos_i1, pos_i2;
    std::tie(pos_i0, pos_i1, pos_i2) = get_pos3(pos, i0, i1, i2);
    float a = area(pos_i0, pos_i1, pos_i2);
    Vector3f pos0_i0, pos0_i1, pos0_i2;
    std::tie(pos0_i0, pos0_i1, pos0_i2) = get_pos3(pos0, i0, i1, i2);
    float a0 = area(pos0_i0, pos0_i1, pos0_i2);
    float diff = 1 - a / a0;
    return diff * diff * a0 * face_coefficient;
}

__attribute__((always_inline))
static float flap_energy(
    const float *__restrict__ pos,
    const float *__restrict__ pos0,
    const int i0,
    const int i1,
    const int i2,
    const int i3,
    const float bending_stiffness
) {
    Vector3f pos_i0, pos_i1, pos_i2, pos_i3;
    std::tie(pos_i0, pos_i1, pos_i2, pos_i3) = get_pos4(pos, i0, i1, i2, i3);
    float theta, w;
    std::tie(theta, w) = dihedral_angle(pos_i0, pos_i1, pos_i2, pos_i3);
    Vector3f pos0_i0, pos0_i1, pos0_i2, pos0_i3;
    std::tie(pos0_i0, pos0_i1, pos0_i2, pos0_i3) = get_pos4(pos0, i0, i1, i2, i3);
    float theta0, w0;
    std::tie(theta0, w0) = dihedral_angle(pos0_i0, pos0_i1, pos0_i2, pos0_i3);
    float diff = theta - theta0;
    return diff * diff * w0 * bending_stiffness;
}


float discrete_shell_simple(
    const float *__restrict__ pos0,
    const int *__restrict__ edges,
    const int num_edges,
    const int *__restrict__ faces,
    const int num_faces,
    const int *__restrict__ flaps,
    const int num_flaps,
    const float edge_coefficient,
    const float face_coefficient,
    const float bending_stiffness,
    const float *__restrict__  pos
) {
    float total_energy = 0;
    for (int i = 0; i < num_edges; i++) {
        total_energy += edge_energy(pos, pos0, edges[2 * i], edges[2 * i + 1], edge_coefficient);
    }

    for (int i = 0; i < num_faces; i++) {
        total_energy += face_energy(pos, pos0, faces[3 * i], faces[3 * i + 1], faces[3 * i + 2], face_coefficient);
    }

    for (int i = 0; i < num_flaps; i++) {
        total_energy += flap_energy(pos, pos0, flaps[4 * i], flaps[4 * i + 1], flaps[4 * i + 2], flaps[4 * i + 3], bending_stiffness);
    }

    return total_energy;
}


__attribute__((always_inline))
static void gradient_ip(
    const float *__restrict__ pos0,
    const int *__restrict__ edges,
    const int num_edges,
    const int *__restrict__ faces,
    const int num_faces,
    const int *__restrict__ flaps,
    const int num_flaps,
    const float edge_coefficient,
    const float face_coefficient,
    const float bending_stiffness,
    const float *__restrict__  pos,
    float *__restrict__ out
    )
{
    __enzyme_autodiff<void>((void *)discrete_shell_simple,
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
    const int *__restrict__ edges,
    const int num_edges,
    const int *__restrict__ faces,
    const int num_faces,
    const int *__restrict__ flaps,
    const int num_flaps,
    const float edge_coefficient,
    const float face_coefficient,
    const float bending_stiffness,
    const float *__restrict__  pos,
    const size_t num_verts)
{
    std::vector<std::tuple<size_t, size_t, float>> hess;
    for (size_t i=0; i<3*num_verts; i++)
        __enzyme_fwddiff<void>((void *)gradient_ip,
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
                               enzyme_dup, pos, __enzyme_todense<float*>(ident_load<float>,   err_store<float>, i),
                               enzyme_dupnoneed, nullptr, __enzyme_todense<float*>(zero_load<float>,   csr_store<float>, i, &hess));
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

    float result = discrete_shell_simple(pos0, edges, num_edges, faces, num_faces, flaps, num_flaps, edge_coefficient, face_coefficient, bending_stiffness, pos);

    std::cout << "Result: " << result << std::endl;

    // Derivative
    float dpos[sizeof(pos)/sizeof(pos[0])];
    for (size_t i=0; i<sizeof(dpos)/sizeof(dpos[0]); i++)
        dpos[i] = 0;
    gradient_ip(pos0, edges, num_edges, faces, num_faces, flaps, num_flaps, edge_coefficient, face_coefficient, bending_stiffness, pos, dpos);

    for (size_t i=0; i<sizeof(dpos)/sizeof(dpos[0]); i++)
        std::cout << "grad_vert[" << i << "]=" << dpos[i] << "\n";

    // Hessian
    const size_t num_verts = 4;
    auto hess_verts = hessian(pos0, edges, num_edges, faces, num_faces, flaps, num_flaps, edge_coefficient, face_coefficient, bending_stiffness, pos, num_verts);

    for (auto hess : hess_verts) {
        std::cout << "i=" << std::get<0>(hess) << ", j=" << std::get<1>(hess) << " val=" << std::get<2>(hess) << "\n";
    }

    return 0;
}
