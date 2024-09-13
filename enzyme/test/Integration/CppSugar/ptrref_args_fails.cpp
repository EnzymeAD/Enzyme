// RUN: if [ %llvmver -ge 17 ]; then %clang++ -std=c++17 -O0 %s -S -emit-llvm -o - %loadClangEnzyme -Xclang -verify; fi
// RUN: if [ %llvmver -ge 17 ]; then %clang++ -std=c++17 -O1 %s -S -emit-llvm -o - %loadClangEnzyme -Xclang -verify; fi
// RUN: if [ %llvmver -ge 17 ]; then %clang++ -std=c++17 -O2 %s -S -emit-llvm -o - %loadClangEnzyme -Xclang -verify; fi
// RUN: if [ %llvmver -ge 17 ]; then %clang++ -std=c++17 -O3 %s -S -emit-llvm -o - %loadClangEnzyme -Xclang -verify; fi
// RUN: if [ %llvmver -ge 17 ]; then %clang++ -std=c++17 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme -Xclang -verify; fi
// RUN: if [ %llvmver -ge 17 ]; then %clang++ -std=c++17 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme -Xclang -verify; fi
// RUN: if [ %llvmver -ge 17 ]; then %clang++ -std=c++17 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme -Xclang -verify; fi
// RUN: if [ %llvmver -ge 17 ]; then %clang++ -std=c++17 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme -Xclang -verify; fi

#include "../test_utils.h"

#include <enzyme/enzyme>

float f(float x)        { return  x *  x; }
float g(const float& x) { return  x *  x; }
float h(const float* x) { return *x * *x; }

int test_failures() {
    const float x = 10;
    {
        /*
         * This will not work in the C++ interface because it is also not allowed
         * with the C-style __enzyme_autodiff call
         */
        const float dfdx = enzyme::get<0>(
            enzyme::get<0>(
                enzyme::autodiff<enzyme::Reverse>( // expected-error@/enzymeroot/enzyme/utils:259 {{no member named 'value' in 'enzyme::Active<const float &>'}} expected-note {{}}
                    g, enzyme::Active<const float&>{ x } // expected-error@/enzymeroot/enzyme/utils:48 {{static assertion failed due to requirement '!std::is_reference_v<const float &>': Reference/pointer active arguments don't make sense for AD!}} expected-note {{}} expected-note@/enzymeroot/enzyme/utils:535 {{}}
                )
            )
        );
    }


    {
        /*
         * Error: non-reference duplicated arguments don't make sense in reverse
         * mode
         */
        float dfdx = 0;
        enzyme::autodiff<enzyme::Reverse>( // expected-error@/enzymeroot/enzyme/utils:527 {{static assertion failed due to requirement 'detail::verify_dup_args<enzyme::ReverseMode<false>, enzyme::Duplicated<float>>::value': Non-reference/pointer Duplicated/DuplicatedNoNeed args don't make sense for Reverse mode AD}} expected-note {{}}
            f, enzyme::Duplicated<float>{ x, dfdx }
        );
    }
    return 0;
}

