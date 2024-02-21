#include <cstdint>
#include <mpfr.h>

extern "C" {

#define BINOP(LLVM_OP_NAME, MPFR_FUNC_NAME, FROM_TYPE, RET, MPFR_GET, ARG1,    \
              MPFR_SET_ARG1, ARG2, MPFR_SET_ARG2, ROUNDING_MODE)               \
  __attribute__((weak)) RET __enzyme_mpfr_##FROM_TYPE_binop_##LLVM_OP_NAME(    \
      ARG1 a, ARG2 b, int64_t exponent, int64_t significand) {                 \
    mpfr_t ma, mb, mc;                                                         \
    mpfr_init2(ma, significand);                                               \
    mpfr_init2(mb, significand);                                               \
    mpfr_init2(mc, significand);                                               \
    mpfr_set_##MPFR_SET_ARG1(ma, a, ROUNDING_MODE);                            \
    mpfr_set_##MPFR_SET_ARG1(mb, b, ROUNDING_MODE);                            \
    mpfr_##MPFR_FUNC_NAME(mc, ma, mb, ROUNDING_MODE);                          \
    RET c = mpfr_get_##MPFR_GET(mc, ROUNDING_MODE);                            \
    mpfr_clear(ma);                                                            \
    mpfr_clear(mb);                                                            \
    mpfr_clear(mc);                                                            \
    return c;                                                                  \
  }

#define DEFAULT_ROUNDING_MODE GMP_RNDN
#define DBL_MANGLE 64_52
#define DOUBLE_BINOP(LLVM_OP_NAME, MPFR_FUNC_NAME, ROUNDING_MODE)              \
  BINOP(LLVM_OP_NAME, MPFR_FUNC_NAME, DBL_MANGLE, double, d, double, d,        \
        double, d, ROUNDING_MODE)
#define DOUBLE_BINOP_DEFAULT_ROUNDING(LLVM_OP_NAME, MPFR_FUNC_NAME)              \
     DOUBLE_BINOP(LLVM_OP_NAME, MPFR_FUNC_NAME, DEFAULT_ROUNDING_MODE)

    //  BINOP(fmul, mul, 64_52, double, d, double, d, double, d, GMP_RNDN)
    DOUBLE_BINOP_DEFAULT_ROUNDING(fmul, mul)
    DOUBLE_BINOP_DEFAULT_ROUNDING(fadd, add)
    DOUBLE_BINOP_DEFAULT_ROUNDING(fdiv, div)

}
