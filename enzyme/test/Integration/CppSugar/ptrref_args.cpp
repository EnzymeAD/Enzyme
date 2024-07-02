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

static const float x      = 10;
static const float etalon = 2 * x;

void test_active() {
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
}

void test_duplicated() {
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
}

float f2(float x, float y)        { return x * y; }
float g2(float x, const float& y) { return x * y; }
float h2(float x, float& y)       { return x * (y++); }
float j2(float x, const float* y) { return x * (*y); }

static const float y       = 5;
static const float etalon2 = 5;

void test_const() {
    {
        // OK: By-value const for by-value arg
        const auto dfdx = enzyme::get<0>(
            enzyme::get<0>(
                enzyme::autodiff<enzyme::Reverse>(
                    f2, enzyme::Active{ x }, enzyme::Const<float>{ y }
                )
            )
        );
        APPROX_EQ(dfdx, etalon2, 1e-10);
    }
    {
        // OK: By-const-ref const for by-const-ref arg
        const auto dfdx = enzyme::get<0>(
            enzyme::get<0>(
                enzyme::autodiff<enzyme::Reverse>(
                    g2, enzyme::Active{ x }, enzyme::Const<const float&>{ y }
                )
            )
        );
        APPROX_EQ(dfdx, etalon2, 1e-10);
    }
    {
        // OK: By-non-const-ref Const for by-non-const-ref arg
        // fun fact: putting `const T*` instead of `T*` in `enzyme/utils:108`
        // not only is semantically incorrect, but also makes the compiler
        // crash on this example!
        float y2 = y;
        const auto dfdx = enzyme::get<0>(
            enzyme::get<0>(
                enzyme::autodiff<enzyme::Reverse>(
                    h2, enzyme::Active{ x }, enzyme::Const<float&>{ y2 }
                )
            )
        );
        APPROX_EQ(dfdx, etalon2, 1e-10);
        APPROX_EQ(y2, y + 1, 1e-10);
    }
    {
        // OK: By-pointer Const for by-pointer arg
        // I feel like testing with non-const pointer too would be excessive
        const auto dfdx = enzyme::get<0>(
            enzyme::get<0>(
                enzyme::autodiff<enzyme::Reverse>(
                    j2, enzyme::Active{ x }, enzyme::Const<const float*>{ &y }
                )
            )
        );
        APPROX_EQ(dfdx, etalon2, 1e-10);
    }
}

int main() {
    test_active();
    test_duplicated();
    test_const();

    return 0;
}
