#include "enzyme/enzyme.h"

#include <iostream>

double foo(double x) { return x * sin(x);  }

int main() {

    enzyme::autodiff_return< enzyme::active<double>&& >::type q1;
    double z1 = q1;

    enzyme::active<double> x{1.0};
    auto y = enzyme::autodiff(foo, x);
    std::cout << y << std::endl;

    //auto z = __enzyme_autodiff<double>((void*)foo, enzyme_out, x.value);
    //std::cout << z << std::endl;

}