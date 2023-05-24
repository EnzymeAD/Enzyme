// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-loose-types -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -enzyme-loose-types -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -

#include <stdio.h>
#include "test_utils.h"

#include <stdio.h>
#include <map>
#include <unordered_map>
#include <string>

int enzyme_dup;

using dictSU = std::unordered_map<std::string, double>;
using dictCU = std::unordered_map<const char*, double>;

using dictSO = std::map<std::string, double>;
using dictCO = std::map<const char*, double>;

template<typename T>
extern void __enzyme_autodiff(void*, int, T&, T&);

template<typename dict>
void square(dict& vals) {
    vals["x"] = vals["x"] * vals["x"];
}

template<typename dict>
void dsquare() {
    dict vals;
    vals["x"] = 3.0;
    dict dvals;
    dvals["x"] = 1.0;
    
    __enzyme_autodiff((void*)square<dict>, enzyme_dup, vals, dvals);

    printf("%f\n", dvals["x"]);
    APPROX_EQ(dvals["x"], 6.0, 1e-7);
}

int main() {
    dsquare<dictSU>();
    dsquare<dictCU>();
    // dsquare<dictSO>();
    // dsquare<dictCO>();
}

