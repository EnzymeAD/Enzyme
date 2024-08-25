// RUN: if [ %llvmver -ge 11 ]; then %clang++ -std=c++17 -O0 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 11 ]; then %clang++ -std=c++17 -O1 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 11 ]; then %clang++ -std=c++17 -O2 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 11 ]; then %clang++ -std=c++17 -O3 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi

#include "../test_utils.h"
#include <vector>
#include <enzyme/enzyme>


std::vector<double> elasticity_kernel(const std::vector<double> &dudxi) {
    return dudxi;
}

int main() {
   std::vector<double> dudxi { 3.4, 5.7 };
   std::vector<double> s_dudxi{ 9.2, 11.3 };

   auto dres = enzyme::get<0>
   (enzyme::autodiff<enzyme::Forward,
    enzyme::DuplicatedNoNeed<std::vector<double>>>
    (elasticity_kernel,
     enzyme::Duplicated<const std::vector<double> &>(dudxi, s_dudxi)));

    APPROX_EQ(dres[0],  9.2, 1e-10);
    APPROX_EQ(dres[1], 11.3, 1e-10);
   
    auto tup =
   (enzyme::autodiff<enzyme::Forward,
    enzyme::Duplicated<std::vector<double>>>
    (elasticity_kernel,
     enzyme::Duplicated<const std::vector<double> &>(dudxi, s_dudxi)));

    APPROX_EQ(enzyme::get<0>(tup)[0],  3.4, 1e-10);
    APPROX_EQ(enzyme::get<0>(tup)[1],  5.7, 1e-10);
    APPROX_EQ(enzyme::get<1>(tup)[0],  9.2, 1e-10);
    APPROX_EQ(enzyme::get<1>(tup)[1], 11.3, 1e-10);
    return 0;
}
