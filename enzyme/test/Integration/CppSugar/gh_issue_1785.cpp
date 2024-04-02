// RUN: if [ %llvmver -ge 11 ]; then %clang++ -std=c++17 -O0 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 11 ]; then %clang++ -std=c++17 -O1 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 11 ]; then %clang++ -std=c++17 -O2 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 11 ]; then %clang++ -std=c++17 -O3 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi

// XFAIL: *

#include "../test_utils.h"

#include <vector>
#include <enzyme/enzyme>

int main() {
   auto elasticity_kernel = [](const std::vector<double> &dudxi,
                               const std::vector<double> &J,
                               const double &w)
   {
      auto r = dudxi;
      return r;
   };

   std::vector<double> dudxi(4), s_dudxi(4), J(4);
   double w = 1.0;

   enzyme::get<0>
   (enzyme::autodiff<enzyme::Forward,
    enzyme::DuplicatedNoNeed<std::vector<double>>>
    (+elasticity_kernel,
     enzyme::Duplicated<std::vector<double> *>{&dudxi, &s_dudxi},
     enzyme::Const<std::vector<double> *>{&J},
     enzyme::Const<double*>{&w}));

    return 0;
}