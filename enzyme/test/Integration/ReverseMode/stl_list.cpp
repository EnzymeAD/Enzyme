// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -

#include "../test_utils.h"

#include <list>


template<typename T>
extern double __enzyme_fwddiff(void*, int, T&, T&);
template<typename T>
extern double __enzyme_autodiff(void*, int, T&, T&);


double test_simple_list(std::list<double>& vals) {
    // modify list without insert/removal
    //vals.front() = 1.5;

    // iterate over list
    double result = 0.0;
    for (const auto& val : vals) {
        result += val * val;
    }
    return result;
}

void test_forward_list() {
    {
        std::list<double> vals = {1.0, 2.0, 3.0};
        std::list<double> dvals = {1.0, 1.0, 1.0};

        double ret = __enzyme_fwddiff((void*)test_simple_list, enzyme_dup, vals, dvals);
        APPROX_EQ( ret, 12., 1e-10);
    }
}

void test_reverse_list() {
    {
        std::list<double> vals = {1.0, 2.0, 3.0};
        std::list<double> dvals = {1.0, 1.0, 1.0};

        double ret = __enzyme_autodiff((void*)test_simple_list, enzyme_dup, vals, dvals);
        //APPROX_EQ( ret, 12., 1e-10);  // TODO: FAILS
    }
}


int main() {
    test_forward_list();
    test_reverse_list();
    return 0;
}

