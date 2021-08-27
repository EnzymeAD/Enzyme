// RUN: %clang -g -std=c11 -O0 %s -S -emit-llvm -o - %loadClangEnzyme | %lli -
// RUN: %clang -g -std=c11 -O1 %s -S -emit-llvm -o - %loadClangEnzyme | %lli -
// RUN: %clang -g -std=c11 -O2 %s -S -emit-llvm -o - %loadClangEnzyme | %lli -
// RUN: %clang -g -std=c11 -O3 %s -S -emit-llvm -o - %loadClangEnzyme | %lli -

#include "test_utils.h"

extern double __enzyme_autodiff(void*, double);

struct Data {
    double x1;
    double x2;
    double res;
};

void foo(double x, struct Data *data) {
    data->x1 = 2 * x;
    data->x2 = x + 3;
    data->res = data->x1 * data->x2;
    return;
}

double call(double x) {
    struct Data *data = malloc(sizeof(struct Data));
    foo(x, data);
    double res = data->res;
    free(data);
    return res;
}

double dcall(double x) {
    // This returns the derivative of square or 2 * x
    return __enzyme_autodiff((void*)call, x);
}
int main() {
    double res = dcall(24.5);
    double exp = 104.0;
    APPROX_EQ(res, exp, 1e-10)
}
