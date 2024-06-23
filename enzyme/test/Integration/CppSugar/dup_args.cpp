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

float f(float x)        { return  x *  x; }
float g(const float& x) { return  x *  x; }
float h(const float* x) { return *x * *x; }

int main() {
    const float x      = 10;
    const float etalon = 2 * x;

    {   // Ok, reverse mode + active
        const float dfdx = enzyme::get<0>(
            enzyme::get<0>(
                enzyme::autodiff<enzyme::Reverse>(
                    f, enzyme::Active{ x }
                )
            )
        );
        APPROX_EQ(dfdx, etalon, 1e-10);
    }
#if 0
    {
        /*
         * This will not work in the C++ interface because it is also not allowed
         * with the C-style __enzyme_autodiff call
         */
        const float dfdx = enzyme::get<0>(
            enzyme::get<0>(
                enzyme::autodiff<enzyme::Reverse>(
                    g, enzyme::Active<const float&>{ x }
                )
            )
        );
        APPROX_EQ(dfdx, etalon, 1e-10);
    }
#endif
#if 0
    {
        /*
         * Error: non-reference duplicated arguments don't make sense in reverse
         * mode
         */
        float dfdx = 0;
        enzyme::autodiff<enzyme::Reverse>(
            f, enzyme::Duplicated<float>{ x, dfdx }
        );
        APPROX_EQ(dfdx, etalon, 1e-10);
    }
#endif
    {   // Ok, reverse mode + reference arg
        float dfdx = 0;
        enzyme::autodiff<enzyme::Reverse>(
            g, enzyme::Duplicated<const float&>{ x, dfdx }
        );
        APPROX_EQ(dfdx, etalon, 1e-10);
    }
    {   // Ok, reverse mode + pointer arg
        float dfdx = 0;
        enzyme::autodiff<enzyme::Reverse>(
            h, enzyme::Duplicated<const float*>{ &x, &dfdx }
        );
        APPROX_EQ(dfdx, etalon, 1e-10);
    }
    {   // OK, forward mode + pass-by-value
        const float dfdx = enzyme::get<0>(
            enzyme::autodiff<enzyme::Forward>(
                f, enzyme::Duplicated<float>{ x, 1.0f }
            )
        );
        APPROX_EQ(dfdx, etalon, 1e-10);
    }
    {
        /*
         * OK: forward mode + pass by reference
         * Regular __enzyme_fwddiff will not allow a non-pointer here too
         */
        const float dx = 1.0f;
        const float dfdx = enzyme::get<0>(
            enzyme::autodiff<enzyme::Forward>(
                g, enzyme::Duplicated<const float&>{ x, dx }
            )
        );
        APPROX_EQ(dfdx, etalon, 1e-10);
    }
    {
        /*
         * OK: forward mode + pass by pointer
         * Regular __enzyme_fwddiff will not allow a non-pointer here too
         */
        const float dx = 1.0f;
        const float dfdx = enzyme::get<0>(
            enzyme::autodiff<enzyme::Forward>(
                h, enzyme::Duplicated<const float*>{ &x, &dx }
            )
        );
        APPROX_EQ(dfdx, etalon, 1e-10);
    }

#if 0 // Lambdas don't work with the C++ interface yet :(
    {
        const auto l = [] (const float& x) { return x * x; };
        float dfdx = 0;
        enzyme::autodiff<enzyme::Reverse>(
            l, enzyme::Duplicated<const float&>{ x, dfdx }
        );
        APPROX_EQ(dfdx, etalon, 1e-10);
    }
#endif
    return 0;
}
