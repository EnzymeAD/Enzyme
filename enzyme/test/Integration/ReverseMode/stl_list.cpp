// FIXME: -O0 fails reverse mode (wrong result) https://github.com/EnzymeAD/Enzyme/pull/2370#issuecomment-3046307237
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -

#include "../test_utils.h"

#include <list>


template<typename ...T>
extern double __enzyme_fwddiff(void*, T...);
template<typename ...T>
extern double __enzyme_autodiff(void*, T...);


double test_iterate_list(std::list<double>& vals) {
    // iterate over list
    double result = 0.0;
    for (const auto& val : vals) {
        result += val * val;
    }
    return result;
}

struct S {
    S(double r) : x(r) {};
    double x = 0.0;
};

double test_modify_list(std::list<S> vals, double x) {
    vals.front().x = x;

    // iterate over list
    double result = 0.0;
    for (const auto& val : vals) {
        result += val.x * val.x;
    }
    return result;
}

void test_forward_list() {
    // diff all values of list
    {
        std::list<double> vals = {1.0, 2.0, 3.0};
        std::list<double> dvals = {1.0, 1.0, 1.0};

        double ret = __enzyme_fwddiff((void*)test_iterate_list, enzyme_dup, vals, dvals);
        APPROX_EQ(ret, 12., 1e-10);
    }

    // list is const, then first value set to active
    {
        std::list<S> vals = {S{1.0}, S{2.0}, S{3.0}};
        double x = 3.0;
        double dx = 1.0;

        double ret = __enzyme_fwddiff((void*)test_modify_list, enzyme_const, vals, enzyme_dup, x, dx);
        APPROX_EQ(ret, 6., 1e-10);
    }
}

void test_reverse_list() {
    // diff all values of list
    {
        std::list<double> vals = {1.0, 2.0, 3.0};
        std::list<double> dvals = {1.0, 1.0, 1.0};

        double ret = __enzyme_autodiff((void*)test_iterate_list, enzyme_dup, vals, dvals);
        APPROX_EQ(ret, 12., 1e-10);
    }

    // list is const, then first value set to active
    {
        std::list<S> vals = {S{1.0}, S{2.0}, S{3.0}};
        double x = 3.0;
        double dx = 1.0;

        double ret = __enzyme_autodiff((void*)test_modify_list, enzyme_const, vals, enzyme_dup, x, dx);
        APPROX_EQ(ret, 6., 1e-10);
    }
}


int main() {
    test_forward_list();
    test_reverse_list();
    return 0;
}

