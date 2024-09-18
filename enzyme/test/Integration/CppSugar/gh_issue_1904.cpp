// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O0 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O1 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O2 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O3 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -std=c++17 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi

#include "../test_utils.h"
#include <vector>

#include <enzyme/enzyme>

using Flt = double;
using Vec = std::vector<Flt>;

Flt sum(const Vec& x) {
    Flt ret = 0.0;
    for (const auto xi : x) ret += xi;
    return ret;
} // <-- sum

Vec dsum(const Vec& x) {
    Vec dx(x.size());
    enzyme::autodiff<enzyme::Reverse>(
        sum,
        enzyme::DuplicatedNoNeed<const Vec&>{ x, dx }
    );
    return dx;
} // <-- dsum

int main() {
    const Vec x{ 3, 42, 451 };
    const auto df = dsum(x);
    TEST_EQ(df.size(), 3);
    APPROX_EQ(df[0], 1, 1e-10);
    APPROX_EQ(df[1], 1, 1e-10);
    APPROX_EQ(df[2], 1, 1e-10);
    return 0;
}
