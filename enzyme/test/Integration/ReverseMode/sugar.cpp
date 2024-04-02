// RUN: if [ %llvmver -ge 11 ]; then %clang++ -std=c++17 -O0 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 11 ]; then %clang++ -std=c++17 -O1 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 11 ]; then %clang++ -std=c++17 -O2 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 11 ]; then %clang++ -std=c++17 -O3 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi

// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi

#include "../test_utils.h"

#include <enzyme/enzyme>

double foo(double x, double y) { return x * y;  }

double square(double x) { return x * x;  }

struct overload {
  double operator()(double x) { return x * 10;  }
  float operator()(float x) { return x * 2;  }
};

struct pair {
    double x;
    double y;
};

int main() {
    
    {
    enzyme::Active<double> x1{3.1};
    enzyme::tuple< enzyme::tuple<double> > dsq = enzyme::autodiff<enzyme::Reverse, enzyme::Active<double>>(square, x1);
    double dd = enzyme::get<0>(enzyme::get<0>(dsq));
    printf("dsq = %f\n", dd);
    APPROX_EQ(dd, 3.1*2, 1e-10); 
    }

    {
    enzyme::Active<double> x1{3.1};
    enzyme::tuple< enzyme::tuple<double> > dsq = enzyme::autodiff<enzyme::Reverse>(square, x1);
    double dd = enzyme::get<0>(enzyme::get<0>(dsq));
    printf("dsq2 = %f\n", dd);
    APPROX_EQ(dd, 3.1*2, 1e-10); 
    }

    {
    enzyme::Active<double> x1{3.1};
    enzyme::tuple< enzyme::tuple<double>, double > dsq = enzyme::autodiff<enzyme::ReverseWithPrimal, enzyme::Active<double>>(square, x1);
    double dd = enzyme::get<0>(enzyme::get<0>(dsq));
    printf("dsq3 = %f\n", dd);
    APPROX_EQ(dd, 3.1*2, 1e-10); 
    double prim = enzyme::get<1>(dsq);
    printf("dsq3_prim = %f\n", prim);
    APPROX_EQ(prim, 3.1*3.1, 1e-10); 
    }

    {
    enzyme::Active<double> x1{3.1};
    enzyme::tuple< enzyme::tuple<double>, double > dsq = enzyme::autodiff<enzyme::ReverseWithPrimal>(square, x1);
    double dd = enzyme::get<0>(enzyme::get<0>(dsq));
    printf("dsq4 = %f\n", dd);
    APPROX_EQ(dd, 3.1*2, 1e-10); 
    double prim = enzyme::get<1>(dsq);
    printf("dsq4_prim = %f\n", prim);
    APPROX_EQ(prim, 3.1*3.1, 1e-10); 
    }

    {
    auto y = enzyme::autodiff<enzyme::Reverse>(foo, enzyme::Active<double>(3.1), enzyme::Active<double>(2.7));
    auto y1 = enzyme::get<0>(enzyme::get<0>(y));
    auto y2 = enzyme::get<1>(enzyme::get<0>(y));
    printf("dmul %f %f\n", y1, y2);
    APPROX_EQ(y1, 2.7, 1e-10); 
    APPROX_EQ(y2, 3.1, 1e-10); 
    }

    {
    auto y = enzyme::autodiff<enzyme::ReverseWithPrimal>(foo, enzyme::Active<double>(3.1), enzyme::Active<double>(2.7));
    auto y1 = enzyme::get<0>(enzyme::get<0>(y));
    auto y2 = enzyme::get<1>(enzyme::get<0>(y));
    auto prim = enzyme::get<1>(y);
    printf("dmul2 %f %f\n", y1, y2);
    printf("dmul_prim %f\n", prim);
    APPROX_EQ(y1, 2.7, 1e-10); 
    APPROX_EQ(y2, 3.1, 1e-10); 
    APPROX_EQ(prim, 2.7*3.1, 1e-10); 
    }

    {
    auto y = enzyme::autodiff<enzyme::Reverse>(overload{}, enzyme::Active<double>(3.1));
    auto y1 = enzyme::get<0>(enzyme::get<0>(y));
    printf("dmul %f\n", y1);
    APPROX_EQ(y1, 10, 1e-10); 
    }

    {
    auto y = enzyme::autodiff<enzyme::Reverse>(overload{}, enzyme::Active<float>(3.1f));
    auto y1 = enzyme::get<0>(enzyme::get<0>(y));
    printf("dmul %f\n", y1);
    APPROX_EQ(y1, 2, 1e-10); 
    }

    {
    auto &&[z1, z2] = __enzyme_autodiff<pair>((void*)foo, enzyme_out, 3.1, enzyme_out, 2.7);
    printf("dmul2 %f %f\n", z1, z2);
    APPROX_EQ(z1, 2.7, 1e-10); 
    APPROX_EQ(z2, 3.1, 1e-10); 
    }

}
