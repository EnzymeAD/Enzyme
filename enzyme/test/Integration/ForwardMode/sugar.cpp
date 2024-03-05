// RUN: if [ %llvmver -ge 11 ]; then %clang++ -std=c++17 -O0 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 11 ]; then %clang++ -std=c++17 -O1 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 11 ]; then %clang++ -std=c++17 -O2 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 11 ]; then %clang++ -std=c++17 -O3 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi

#include "../test_utils.h"

#include <enzyme/enzyme>

double foo(double x, double y) { return x * y;  }

double square(double x) { return x * x;  }

struct pair {
    double x;
    double y;
};

int main() {
    
    {
    enzyme::tuple< double, double > dsq = enzyme::autodiff<enzyme::Forward, enzyme::Duplicated<double>>(square, enzyme::Duplicated<double>(3.1, 1.0));
    double dd = enzyme::get<1>(dsq);
    printf("dsq = %f\n", dd);
    APPROX_EQ(dd, 3.1*2, 1e-10); 
    
    double pp = enzyme::get<0>(dsq);
    printf("sq = %f\n", pp);
    APPROX_EQ(dd, 3.1*2, 1e-10); 
    }
    
    {
    enzyme::tuple< double > dsq = enzyme::autodiff<enzyme::Forward, enzyme::DuplicatedNoNeed<double>>(square, enzyme::Duplicated<double>(3.1, 1.0));
    double dd = enzyme::get<0>(dsq);
    printf("dsq = %f\n", dd);
    APPROX_EQ(dd, 3.1*2, 1e-10); 
    }
    
    {
    enzyme::tuple< double > dsq = enzyme::autodiff<enzyme::Forward>(square, enzyme::Duplicated<double>(3.1, 1.0));
    double dd = enzyme::get<0>(dsq);
    printf("dsq = %f\n", dd);
    APPROX_EQ(dd, 3.1*2, 1e-10); 
    }
    
    {
    enzyme::tuple< double > dsq = enzyme::autodiff<enzyme::Forward, enzyme::Const<double>>(square, enzyme::Duplicated<double>(3.1, 1.0));
    double pp = enzyme::get<0>(dsq);
    printf("sq = %f\n", pp);
    APPROX_EQ(pp, 3.1*3.1, 1e-10); 
    }
}
