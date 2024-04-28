// RUN: if [ %llvmver -ge 12 ] && [ %hasMPFR == "yes" ] ; then %clang -c -DTRUNC_MEM -O2    %s -o /dev/null -emit-llvm %newLoadClangEnzyme -include enzyme/fprt/mpfr.h -Xclang -verify -Rpass=enzyme; fi
// RUN: if [ %llvmver -ge 12 ] && [ %hasMPFR == "yes" ] ; then %clang -c -DTRUNC_MEM -O2 -g %s -o /dev/null -emit-llvm %newLoadClangEnzyme -include enzyme/fprt/mpfr.h -Xclang -verify -Rpass=enzyme; fi
// COM: if [ %llvmver -ge 12 ] && [ %hasMPFR == "yes" ] ; then %clang -c -DTRUNC_OP  -O2    %s -o /dev/null -emit-llvm %newLoadClangEnzyme -include enzyme/fprt/mpfr.h -Xclang -verify -Rpass=enzyme; fi
// COM: if [ %llvmver -ge 12 ] && [ %hasMPFR == "yes" ] ; then %clang -c -DTRUNC_OP  -O2 -g %s -o /dev/null -emit-llvm %newLoadClangEnzyme -include enzyme/fprt/mpfr.h -Xclang -verify -Rpass=enzyme; fi

#include <math.h>

#define FROM 64
#define TO 32

double bithack(double a) {
    return *((int64_t *)&a) + 1; // expected-remark {{Will not follow FP through this cast.}}, expected-remark {{Will not follow FP through this cast.}}
}
__attribute__((noinline))
float truncf(double a) {
    return (float)a; // expected-remark {{Will not follow FP through this cast.}}
}

double intrinsics(double a, double b) {
    return bithack(a) * truncf(b); // expected-remark {{Will not follow FP through this cast.}}
}

typedef double (*fty)(double *, double *, double *, int);

typedef double (*fty2)(double, double);

extern fty __enzyme_truncate_mem_func_2(...);
extern fty2 __enzyme_truncate_mem_func(...);
extern fty __enzyme_truncate_op_func_2(...);
extern fty2 __enzyme_truncate_op_func(...);
extern double __enzyme_truncate_mem_value(...);
extern double __enzyme_expand_mem_value(...);


int main() {
    #ifdef TRUNC_MEM
    {
        double a = 2;
        double b = 3;
        a = __enzyme_truncate_mem_value(a, FROM, TO);
        b = __enzyme_truncate_mem_value(b, FROM, TO);
        double trunc = __enzyme_expand_mem_value(__enzyme_truncate_mem_func(intrinsics, FROM, TO)(a, b), FROM, TO);
    }
    #endif
    #ifdef TRUNC_OP
    {
        double a = 2;
        double b = 3;
        double trunc = __enzyme_truncate_op_func(intrinsics, FROM, TO)(a, b);
    }
    #endif

}
