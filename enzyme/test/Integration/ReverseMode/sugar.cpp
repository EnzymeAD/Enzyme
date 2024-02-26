// RUN: if [ %llvmver -ge 11 ]; then %clang++ -std=c++17 -O0 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 11 ]; then %clang++ -std=c++17 -O1 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 11 ]; then %clang++ -std=c++17 -O2 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 11 ]; then %clang++ -std=c++17 -O3 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O0 %s -mllvm -print-before-all -mllvm -print-after-all -mllvm -print-module-scope -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi

#include "../test_utils.h"

#include <enzyme/enzyme>

double foo(double x, double y) { return x * y;  }

struct pair {
    double x;
    double y;
};

int main() {

    enzyme::autodiff_return< enzyme::active<double>&& >::type q1;
    double mo = q1;

    enzyme::active<double> x1{3.1};
    enzyme::active<double> x2{2.7};
    auto y = enzyme::autodiff(foo, x1, x2);
    auto y1 = enzyme::get<0>(y);
    auto y2 = enzyme::get<1>(y);
    printf("%f %f\n", y1, y2);
    APPROX_EQ(y1, 2.7, 1e-10); 
    APPROX_EQ(y2, 3.1, 1e-10); 

    auto &&[z1, z2] = __enzyme_autodiff<pair>((void*)foo, enzyme_out, x1.value, enzyme_out, x2.value);
    printf("%f %f\n", z1, z2);
    APPROX_EQ(z1, 2.7, 1e-10); 
    APPROX_EQ(z2, 3.1, 1e-10); 

}
