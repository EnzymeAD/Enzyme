// This should work on LLVM 7, 8, 9, however in CI the version of clang installed on Ubuntu 18.04 cannot load
// a clang plugin properly without segfaulting on exit. This is fine on Ubuntu 20.04 or later LLVM versions...
// RUN: if [ %llvmver -ge 10 ]; then %clang -std=c11 -O0 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 10 ]; then %clang -std=c11 -O1 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 10 ]; then %clang -std=c11 -O2 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 10 ]; then %clang -std=c11 -O3 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 10 ]; then %clang -std=c11 -O0 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -ge 10 ]; then %clang -std=c11 -O1 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -ge 10 ]; then %clang -std=c11 -O2 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -ge 10 ]; then %clang -std=c11 -O3 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi

#include <vector>
#include <list>
#include <forward_list>
#include "test_utils.h"

template<typename RT, typename... Args> RT __enzyme_fwddiff(Args...);
template<typename RT, typename... Args> RT __enzyme_autodiff(Args...);
template<typename RT, typename... Args> RT __enzyme_augmentfwd(Args...);
template<typename RT, typename... Args> RT __enzyme_reverse(Args...);

// example that layers enzyme_nocache over enzyme_dup
// in the case of std::(forward_)list:
// the contents must be cached since the list cannot be traversed in reverse otherwise

template<typename T>
void f(T& y, T& x) {
    auto xi = x.begin();
    auto yi = y.begin();
    for (; xi != x.end(); ++xi, ++yi)
        *yi = *xi * *xi;
}

template<typename T>
__attribute__((noinline)) void* f_aug(T& y, T& y_b, T& x, T& x_b) {
    return __enzyme_augmentfwd<void*>(f<T>, enzyme_nocache, &y, &y_b,
                                            enzyme_nocache, &x, &x_b);
}

template<typename T>
__attribute__((noinline)) void f_rev(T& y, T& y_b, T& x, T& x_b, void* tape) {
    __enzyme_reverse<void*>(f<T>, enzyme_nocache, &y, &y_b,
                                  enzyme_nocache, &x, &x_b,
                                  tape);
}

template<typename T>
void test_f() {
    size_t n = 7;
    T x(n, 0.0), x_b(n, 0.0), y(n, 0.0), y_b(n, 1.0);
    size_t i = 0;
    for (auto& xi: x)
        xi = i++;

    void* tape = f_aug<T>(y, y_b, x, x_b);
    printf("tape: %p\n", tape);
    f_rev<T>(y, y_b, x, x_b, tape);

    i = 0;
    auto xi = x.begin();
    auto yi = y.begin();
    auto xi_b = x_b.begin();
    auto yi_b = y_b.begin();
    for (; xi != x.end(); ++xi, ++xi_b, ++yi, ++yi_b, ++i) {
        APPROX_EQ(*xi,     i, 1e-10);
        APPROX_EQ(*xi_b, 2*i, 1e-10);
        APPROX_EQ(*yi,   i*i, 1e-10);
        APPROX_EQ(*yi_b,   0, 1e-10);
    }
}

// example that layers enzyme_nocache over enzyme_const
// without enzyme_nocache, the contents of c are cached on the tape
// with enzyme_nocache - at least in the case of std::vector - there is no caching

template<typename T>
void g(T& y, T& x, T& c) {
    auto xi = x.begin();
    auto yi = y.begin();
    auto ci = c.begin();
    for (; xi != x.end(); ++xi, ++yi, ++ci)
        *yi = *ci * *xi * *xi;
}

template<typename T>
__attribute__((noinline)) void* g_aug(T& y, T& y_b, T& x, T& x_b, T& c) {
    return __enzyme_augmentfwd<void*>(g<T>, enzyme_nocache, &y, &y_b,
                                            enzyme_nocache, &x, &x_b,
                                            enzyme_nocache, enzyme_const, &c);
}

template<typename T>
__attribute__((noinline)) void g_rev(T& y, T& y_b, T& x, T& x_b, T& c, void* tape) {
    __enzyme_reverse<void*>(g<T>, enzyme_nocache, &y, &y_b,
                                  enzyme_nocache, &x, &x_b,
                                  enzyme_nocache, enzyme_const, &c,
                                  tape);
}

template<typename T>
void test_g() {
    size_t n = 7;
    T x(n, 0.0), x_b(n, 0.0), y(n, 0.0), y_b(n, 1.0), c(n, 0.0);
    size_t i = 0;
    for (auto& xi: x)
        xi = i++;
    for (auto& ci: c)
        ci = i++;

    void* tape = g_aug<T>(y, y_b, x, x_b, c);
    printf("tape: %p\n", tape);
    g_rev<T>(y, y_b, x, x_b, c, tape);

    i = 0;
    auto xi = x.begin();
    auto xi_b = x_b.begin();
    auto yi = y.begin();
    auto yi_b = y_b.begin();
    auto ci = c.begin();
    for (; xi != x.end(); ++xi, ++xi_b, ++yi, ++yi_b, ++ci, ++i) {
        APPROX_EQ(*xi,         i, 1e-10);
        APPROX_EQ(*xi_b, *ci*2*i, 1e-10);
        APPROX_EQ(*yi,   *ci*i*i, 1e-10);
        APPROX_EQ(*yi_b,       0, 1e-10);
    }
}

int main()
{
    test_f<std::vector<double>>();
    test_f<std::list<double>>();
    test_f<std::forward_list<double>>();
    test_g<std::vector<double>>();
    // that the following are broken on -O0 is not the fault of the enzyme_nocache implementation
    test_g<std::list<double>>();
    test_g<std::forward_list<double>>();
}
