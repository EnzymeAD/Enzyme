// TODO: -O0 need support for map inserts https://fwd.gymni.ch/R6Q2bQ
// RUN: %clang++ -std=c++11 -ffast-math -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang++ -std=c++11 -ffast-math -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang++ -std=c++11 -ffast-math -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -

#include "../test_utils.h"

#include <cmath>
#include <string>
#include <unordered_map>


extern double __enzyme_autodiff(void*, double);

std::unordered_map<std::string, double>
fill_umap(double x, double y) {
    std::unordered_map<std::string, double> data;

    data["alpha_x"] = x*x + y;
    data["alpha_y"] = y*y + 2.;

    return data;
}

double compute_umap_op(double r) {
    double x = std::sqrt(r);
    double y = std::log(r);

    std::unordered_map<std::string, double> data = fill_umap(x, y);

    // FIXME: memory leak
    // https://github.com/EnzymeAD/Enzyme/issues/2367#issuecomment-3025856672
    return data["alpha_x"];
}

double compute_umap_at(double r) {
    double x = std::sqrt(r);
    double y = std::log(r);

    std::unordered_map<std::string, double> const data = fill_umap(x, y);

    // FIXME: fails to compile
    // https://fwd.gymni.ch/oGas9k
    return data.at("alpha_x");
}

void test_umap() {
    double r0 = 3.0;

    // operator[]
    {
        // normal
        double const alpha_x = compute_umap_op(r0);
        APPROX_EQ( alpha_x, 4.098612, 1e-6);

        // diff
        double ddx = __enzyme_autodiff((void*) compute_umap_op, r0);
        APPROX_EQ( ddx, 1.333333, 1e-6);
    }

    // at()
    {
        // normal
        double const alpha_x = compute_umap_at(r0);
        APPROX_EQ( alpha_x, 4.098612, 1e-6);

        // diff
        double ddx = __enzyme_autodiff((void*) compute_umap_at, r0);
        APPROX_EQ( ddx, 1.333333, 1e-6);
    }
}

int main() {
    test_umap();

    return 0;
}


