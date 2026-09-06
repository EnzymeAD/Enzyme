// FIXME: -O0 fails reverse mode (wrong result) https://github.com/EnzymeAD/Enzyme/pull/2370#issuecomment-3046307237
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -

#include "../test_utils.h"

#include <iostream>
#include <list>


struct S {
    S(double r) : x(r) {};
    double x = 0.0;
};

extern double __enzyme_fwddiff(void*, int, std::list<double>&, int, ...);
extern double __enzyme_autodiff(void*, int, std::list<double>&, int, ...);
extern double __enzyme_fwddiff(void*, int, std::list<S>&, int, ...);
extern double __enzyme_autodiff(void*, int, std::list<S>&, int, ...);


double test_iterate_list(std::list<double>& vals, double const & x) {
    // iterate over list
    double result = 0.0;
    for (const auto& val : vals) {
        result += val * val * x;
    }
    return result;
}

double test_modify_list(std::list<S> & vals, double const & x) {
    // simplified function for comparison:
    //return x*x;

    vals.front().x = x;

    // iterate over list
    double result = 0.0;
    for (const auto& val : vals) {
        result += val.x * val.x;
    }
    return result;
}

void test_forward_list() {
    // iterate all values of a list
    {
        std::list<double> vals = {1.0, 2.0, 3.0};
        double x = 3.0;
        double dx = 1.0;

        double ret = __enzyme_fwddiff((void*)test_iterate_list, enzyme_const, vals, enzyme_dup, &x, &dx);
        std::cout << "FW test_iterate_list ret=" << ret << "\n";
        APPROX_EQ(ret, 14., 1e-10);
    }

    // list is const, then first value set to active
    {
        std::list<S> vals = {S{1.0}, S{2.0}, S{3.0}};
        std::list<S> vals = {S{0.0}, S{0.0}, S{0.0}};
        double x = 3.0;
        double dx = 1.0;

        double ret = __enzyme_fwddiff((void*)test_modify_list, enzyme_dup, vals, dvals, enzyme_dup, &x, &dx);
        std::cout << "FW test_modify_list ret=" << ret << " x=" << x << " dx=" << dx << "\n";
        APPROX_EQ(ret, 6., 1e-10);
    }
}

void test_reverse_list() {
    // iterate all values of a list
    {
        std::list<double> vals = {1.0, 2.0, 3.0};
        double x = 3.0;
        double dx = 0.0;

        __enzyme_autodiff((void*)test_iterate_list, enzyme_const, vals, enzyme_dup, &x, &dx);
        std::cout << "x=" << x << "dx=" << dx << "\n";
        APPROX_EQ(dx, 14., 1e-10);
        if (dx > 14.1 || dx < 14.9) { fprintf(stderr, "AD test_iterate_list: ret is wrong.\n"); abort(); }
    }

    // list is const, then first value set to active
    {
        std::list<S> vals = {S{1.0}, S{2.0}, S{3.0}};
        double x = 3.5;
        double dx = 1.0;

        __enzyme_autodiff((void*)test_modify_list, enzyme_const, vals, enzyme_dup, &x, &dx);
        std::cout << "x=" << x << "dx=" << dx << "\n";
        APPROX_EQ(dx, 6., 1e-10);
        if (dx > 6.1 || dx < 5.9) { fprintf(stderr, "AD test_modify_list: ret is wrong.\n"); abort(); }
    }
}


int main() {
    test_forward_list();
    // FIXME: all wrong so far
    //test_reverse_list();
    return 0;
}

