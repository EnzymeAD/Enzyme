// RUN: if [ %llvmver -ge 10 ]; then %clang -g -O0 %s -S -emit-llvm -o -  %loadClangEnzyme -Xclang -verify; fi

#include "test_utils.h"

#include <stdio.h>

int enzyme_dup;
int enzyme_out;
int enzyme_const;

template < typename return_type, typename ... T >
extern return_type __enzyme_fwddiff(void*, T ... );

double square(double x) { return x * x; }

template < typename function, typename T > 
auto jvp(function f, T arg, T darg) {
  using output_type = decltype(f(arg));
  return __enzyme_fwddiff<output_type>((void*)f, enzyme_dup, arg, darg); // expected-error {{Enzyme: failed to find fn to differentiate}}
}

int main() {
    // calling fwddiff with the same args in a function template crashes
    auto output2 = jvp(square, 1.0, 1.0);
    printf("%f\n", output2);
}
