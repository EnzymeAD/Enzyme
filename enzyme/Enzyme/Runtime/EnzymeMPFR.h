//===- EnzymeMPFR.h - MPFR wrappers ---------------------------------------===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// If using this code in an academic setting, please cite the following:
// @incollection{enzymeNeurips,
// title = {Instead of Rewriting Foreign Code for Machine Learning,
//          Automatically Synthesize Fast Gradients},
// author = {Moses, William S. and Churavy, Valentin},
// booktitle = {Advances in Neural Information Processing Systems 33},
// year = {2020},
// note = {To appear in},
// }
//
//===----------------------------------------------------------------------===//
//
// This file contains easy to use wrappers around MPFR functions.
//
//===----------------------------------------------------------------------===//
#ifndef __ENZYME_RUNTIME_ENZYME_MPFR__
#define __ENZYME_RUNTIME_ENZYME_MPFR__

#include <mpfr.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// TODO s
//
// (for MPFR ver. 2.1)
//
// We need to set the range of the allowed exponent using `mpfr_set_emin` and
// `mpfr_set_emax`. (This means we can also play with whether the range is
// centered around 0 (1?) or somewhere else)
//
// (also these need to be mutex'ed as the exponent change is global in mpfr and
// not float-specific) ... (mpfr seems to have thread safe mode - check if it is
// enabled or if it is enabled by default)
//
// For that we need to do this check:
//   If the user changes the exponent range, it is her/his responsibility to
//   check that all current floating-point variables are in the new allowed
//   range (for example using mpfr_check_range), otherwise the subsequent
//   behavior will be undefined, in the sense of the ISO C standard.
//
// MPFR docs state the following:
//   Note: Overflow handling is still experimental and currently implemented
//   partially. If an overflow occurs internally at the wrong place, anything
//   can happen (crash, wrong results, etc).
//
// Which we would like to avoid somehow.
//
// MPFR also has this limitation that we need to address for accurate
// simulation:
//   [...] subnormal numbers are not implemented.
//

#define __ENZYME_MPFR_BINOP(OP_TYPE, LLVM_OP_NAME, MPFR_FUNC_NAME, FROM_TYPE,  \
                            RET, MPFR_GET, ARG1, MPFR_SET_ARG1, ARG2,          \
                            MPFR_SET_ARG2, ROUNDING_MODE)                      \
  __attribute__((weak))                                                        \
  RET __enzyme_mpfr_##FROM_TYPE_##OP_TYPE_##LLVM_OP_NAME(                      \
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

#define __ENZYME_MPFR_DEFAULT_ROUNDING_MODE GMP_RNDN
#define __ENZYME_MPFR_DBL_MANGLE 64_52
#define __ENZYME_MPFR_DOUBLE_BINOP(LLVM_OP_NAME, MPFR_FUNC_NAME,               \
                                   ROUNDING_MODE)                              \
  __ENZYME_MPFR_BINOP(binop, LLVM_OP_NAME, MPFR_FUNC_NAME,                     \
                      __ENZYME_MPFR_DBL_MANGLE, double, d, double, d, double,  \
                      d, ROUNDING_MODE)
#define __ENZYME_MPFR_DOUBLE_BINOP_DEFAULT_ROUNDING(LLVM_OP_NAME,              \
                                                    MPFR_FUNC_NAME)            \
  __ENZYME_MPFR_DOUBLE_BINOP(LLVM_OP_NAME, MPFR_FUNC_NAME,                     \
                             __ENZYME_MPFR_DEFAULT_ROUNDING_MODE)

__ENZYME_MPFR_DOUBLE_BINOP_DEFAULT_ROUNDING(fmul, mul)
__ENZYME_MPFR_DOUBLE_BINOP_DEFAULT_ROUNDING(fadd, add)
__ENZYME_MPFR_DOUBLE_BINOP_DEFAULT_ROUNDING(fdiv, div)

#ifdef __cplusplus
}
#endif

#endif // #ifndef __ENZYME_RUNTIME_ENZYME_MPFR__
