//===- fprt/mpfr - MPFR wrappers ---------------------------------------===//
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
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// TODO TODO TODO
// TODO TODO TODO
// TODO TODO TODO
// TODO TODO TODO
// TODO TODO TODO
// I dont think we intercept comparisons - we most definitely should.
// TODO TODO TODO
// TODO TODO TODO
// TODO TODO TODO
// TODO TODO TODO
// TODO TODO TODO

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
// TODO maybe take debug info as parameter - then we can emit warnings or tie
// operations to source location
//
// TODO we need to provide f32 versions, and also instrument the
// truncation/expansion between f32/f64/etc

#define __ENZYME_MPFR_ATTRIBUTES __attribute__((weak))
#define __ENZYME_MPFR_ORIGINAL_ATTRIBUTES __attribute__((weak))
#define __ENZYME_MPFR_DEFAULT_ROUNDING_MODE GMP_RNDN

static bool __enzyme_fprt_is_mem_mode(int64_t mode) { return mode & 0b0001; }
static bool __enzyme_fprt_is_op_mode(int64_t mode) { return mode & 0b0010; }

typedef struct {
  mpfr_t v;
} __enzyme_fp;

static double __enzyme_fprt_ptr_to_double(__enzyme_fp *p) {
  return *((double *)(&p));
}
static __enzyme_fp *__enzyme_fprt_double_to_ptr(double d) {
  return *((__enzyme_fp **)(&d));
}

__ENZYME_MPFR_ATTRIBUTES
double __enzyme_fprt_64_52_get(double _a, int64_t exponent, int64_t significand,
                               int64_t mode) {
  __enzyme_fp *a = __enzyme_fprt_double_to_ptr(_a);
  return mpfr_get_d(a->v, __ENZYME_MPFR_DEFAULT_ROUNDING_MODE);
}

__ENZYME_MPFR_ATTRIBUTES
double __enzyme_fprt_64_52_new(double _a, int64_t exponent, int64_t significand,
                               int64_t mode) {
  __enzyme_fp *a = (__enzyme_fp *)malloc(sizeof(__enzyme_fp));
  mpfr_init2(a->v, significand);
  mpfr_set_d(a->v, _a, __ENZYME_MPFR_DEFAULT_ROUNDING_MODE);
  return __enzyme_fprt_ptr_to_double(a);
}

__ENZYME_MPFR_ATTRIBUTES
double __enzyme_fprt_64_52_const(double _a, int64_t exponent,
                                 int64_t significand, int64_t mode) {
  // TODO This should really be called only once for an appearance in the code,
  // currently it is called every time a flop uses a constant.
  return __enzyme_fprt_64_52_new(_a, exponent, significand, mode);
}

__ENZYME_MPFR_ATTRIBUTES
__enzyme_fp *__enzyme_fprt_64_52_new_intermediate(int64_t exponent,
                                                  int64_t significand,
                                                  int64_t mode) {
  __enzyme_fp *a = (__enzyme_fp *)malloc(sizeof(__enzyme_fp));
  mpfr_init2(a->v, significand);
  return a;
}

__ENZYME_MPFR_ATTRIBUTES
void __enzyme_fprt_64_52_delete(double a, int64_t exponent, int64_t significand,
                                int64_t mode) {
  free(__enzyme_fprt_double_to_ptr(a));
}

#define __ENZYME_MPFR_SINGOP(OP_TYPE, LLVM_OP_NAME, MPFR_FUNC_NAME, FROM_TYPE, \
                             RET, MPFR_GET, ARG1, MPFR_SET_ARG1,               \
                             ROUNDING_MODE)                                    \
  __ENZYME_MPFR_ATTRIBUTES                                                     \
  RET __enzyme_fprt_##FROM_TYPE##_##OP_TYPE##_##LLVM_OP_NAME(                  \
      ARG1 a, int64_t exponent, int64_t significand, int64_t mode) {           \
    if (__enzyme_fprt_is_op_mode(mode)) {                                      \
      mpfr_t ma, mc;                                                           \
      mpfr_init2(ma, significand);                                             \
      mpfr_init2(mc, significand);                                             \
      mpfr_set_##MPFR_SET_ARG1(ma, a, ROUNDING_MODE);                          \
      mpfr_##MPFR_FUNC_NAME(mc, ma, ROUNDING_MODE);                            \
      RET c = mpfr_get_##MPFR_GET(mc, ROUNDING_MODE);                          \
      mpfr_clear(ma);                                                          \
      mpfr_clear(mc);                                                          \
      return c;                                                                \
    } else if (__enzyme_fprt_is_mem_mode(mode)) {                              \
      __enzyme_fp *ma = __enzyme_fprt_double_to_ptr(a);                        \
      __enzyme_fp *mc =                                                        \
          __enzyme_fprt_64_52_new_intermediate(exponent, significand, mode);   \
      mpfr_##MPFR_FUNC_NAME(mc->v, ma->v, ROUNDING_MODE);                      \
      return __enzyme_fprt_ptr_to_double(mc);                                  \
    } else {                                                                   \
      abort();                                                                 \
    }                                                                          \
  }

// TODO this is a bit sketchy if the user cast their float to int before calling
// this. We need to detect these patterns
#define __ENZYME_MPFR_BIN_INT(OP_TYPE, LLVM_OP_NAME, MPFR_FUNC_NAME,           \
                              FROM_TYPE, RET, MPFR_GET, ARG1, MPFR_SET_ARG1,   \
                              ARG2, ROUNDING_MODE)                             \
  __ENZYME_MPFR_ATTRIBUTES                                                     \
  RET __enzyme_fprt_##FROM_TYPE##_##OP_TYPE##_##LLVM_OP_NAME(                  \
      ARG1 a, ARG2 b, int64_t exponent, int64_t significand, int64_t mode) {   \
    if (__enzyme_fprt_is_op_mode(mode)) {                                      \
      mpfr_t ma, mc;                                                           \
      mpfr_init2(ma, significand);                                             \
      mpfr_init2(mc, significand);                                             \
      mpfr_set_##MPFR_SET_ARG1(ma, a, ROUNDING_MODE);                          \
      mpfr_##MPFR_FUNC_NAME(mc, ma, b, ROUNDING_MODE);                         \
      RET c = mpfr_get_##MPFR_GET(mc, ROUNDING_MODE);                          \
      mpfr_clear(ma);                                                          \
      mpfr_clear(mc);                                                          \
      return c;                                                                \
    } else if (__enzyme_fprt_is_mem_mode(mode)) {                              \
      __enzyme_fp *ma = __enzyme_fprt_double_to_ptr(a);                        \
      __enzyme_fp *mc =                                                        \
          __enzyme_fprt_64_52_new_intermediate(exponent, significand, mode);   \
      mpfr_##MPFR_FUNC_NAME(mc->v, ma->v, b, ROUNDING_MODE);                   \
      return __enzyme_fprt_ptr_to_double(mc);                                  \
    } else {                                                                   \
      abort();                                                                 \
    }                                                                          \
  }

#define __ENZYME_MPFR_BIN(OP_TYPE, LLVM_OP_NAME, MPFR_FUNC_NAME, FROM_TYPE,    \
                          RET, MPFR_GET, ARG1, MPFR_SET_ARG1, ARG2,            \
                          MPFR_SET_ARG2, ROUNDING_MODE)                        \
  __ENZYME_MPFR_ATTRIBUTES                                                     \
  RET __enzyme_fprt_##FROM_TYPE##_##OP_TYPE##_##LLVM_OP_NAME(                  \
      ARG1 a, ARG2 b, int64_t exponent, int64_t significand, int64_t mode) {   \
    if (__enzyme_fprt_is_op_mode(mode)) {                                      \
      mpfr_t ma, mb, mc;                                                       \
      mpfr_init2(ma, significand);                                             \
      mpfr_init2(mb, significand);                                             \
      mpfr_init2(mc, significand);                                             \
      mpfr_set_##MPFR_SET_ARG1(ma, a, ROUNDING_MODE);                          \
      mpfr_set_##MPFR_SET_ARG2(mb, b, ROUNDING_MODE);                          \
      mpfr_##MPFR_FUNC_NAME(mc, ma, mb, ROUNDING_MODE);                        \
      RET c = mpfr_get_##MPFR_GET(mc, ROUNDING_MODE);                          \
      mpfr_clear(ma);                                                          \
      mpfr_clear(mb);                                                          \
      mpfr_clear(mc);                                                          \
      return c;                                                                \
    } else if (__enzyme_fprt_is_mem_mode(mode)) {                              \
      __enzyme_fp *ma = __enzyme_fprt_double_to_ptr(a);                        \
      __enzyme_fp *mb = __enzyme_fprt_double_to_ptr(b);                        \
      __enzyme_fp *mc =                                                        \
          __enzyme_fprt_64_52_new_intermediate(exponent, significand, mode);   \
      mpfr_##MPFR_FUNC_NAME(mc->v, ma->v, mb->v, ROUNDING_MODE);               \
      return __enzyme_fprt_ptr_to_double(mc);                                  \
    } else {                                                                   \
      abort();                                                                 \
    }                                                                          \
  }

#define __ENZYME_MPFR_FMULADD(LLVM_OP_NAME, FROM_TYPE, TYPE, MPFR_TYPE,        \
                              LLVM_TYPE, ROUNDING_MODE)                        \
  __ENZYME_MPFR_ATTRIBUTES                                                     \
  TYPE __enzyme_fprt_##FROM_TYPE##_intr_##LLVM_OP_NAME##_##LLVM_TYPE(          \
      TYPE a, TYPE b, TYPE c, int64_t exponent, int64_t significand,           \
      int64_t mode) {                                                          \
    if (__enzyme_fprt_is_op_mode(mode)) {                                      \
      mpfr_t ma, mb, mc, mmul, madd;                                           \
      mpfr_init2(ma, significand);                                             \
      mpfr_init2(mb, significand);                                             \
      mpfr_init2(mc, significand);                                             \
      mpfr_init2(mmul, significand);                                           \
      mpfr_init2(madd, significand);                                           \
      mpfr_set_##MPFR_TYPE(ma, a, ROUNDING_MODE);                              \
      mpfr_set_##MPFR_TYPE(mb, b, ROUNDING_MODE);                              \
      mpfr_set_##MPFR_TYPE(mc, c, ROUNDING_MODE);                              \
      mpfr_mul(mmul, ma, mb, ROUNDING_MODE);                                   \
      mpfr_add(madd, mmul, mc, ROUNDING_MODE);                                 \
      TYPE res = mpfr_get_##MPFR_TYPE(madd, ROUNDING_MODE);                    \
      mpfr_clear(ma);                                                          \
      mpfr_clear(mb);                                                          \
      mpfr_clear(mc);                                                          \
      mpfr_clear(mmul);                                                        \
      mpfr_clear(madd);                                                        \
      return res;                                                              \
    } else if (__enzyme_fprt_is_mem_mode(mode)) {                              \
      __enzyme_fp *ma = __enzyme_fprt_double_to_ptr(a);                        \
      __enzyme_fp *mb = __enzyme_fprt_double_to_ptr(b);                        \
      __enzyme_fp *mc = __enzyme_fprt_double_to_ptr(c);                        \
      double mmul = __enzyme_fprt_##FROM_TYPE##_binop_fmul(                    \
          __enzyme_fprt_ptr_to_double(ma), __enzyme_fprt_ptr_to_double(mb),    \
          exponent, significand, mode);                                        \
      double madd = __enzyme_fprt_##FROM_TYPE##_binop_fadd(                    \
          mmul, __enzyme_fprt_ptr_to_double(mc), exponent, significand, mode); \
      return madd;                                                             \
    } else {                                                                   \
      abort();                                                                 \
    }                                                                          \
  }

// TODO This does not currently make distinctions between ordered/unordered.
#define __ENZYME_MPFR_FCMP_IMPL(NAME, ORDERED, CMP, FROM_TYPE, TYPE, MPFR_GET, \
                                ROUNDING_MODE)                                 \
  __ENZYME_MPFR_ATTRIBUTES                                                     \
  bool __enzyme_fprt_##FROM_TYPE##_fcmp_##NAME(                                \
      TYPE a, TYPE b, int64_t exponent, int64_t significand, int64_t mode) {   \
    if (__enzyme_fprt_is_op_mode(mode)) {                                      \
      mpfr_t ma, mb;                                                           \
      mpfr_init2(ma, significand);                                             \
      mpfr_init2(mb, significand);                                             \
      mpfr_set_##MPFR_GET(ma, a, ROUNDING_MODE);                               \
      mpfr_set_##MPFR_GET(mb, b, ROUNDING_MODE);                               \
      int ret = mpfr_cmp(ma, mb);                                              \
      mpfr_clear(ma);                                                          \
      mpfr_clear(mb);                                                          \
      return ret CMP;                                                          \
    } else if (__enzyme_fprt_is_mem_mode(mode)) {                              \
      __enzyme_fp *ma = __enzyme_fprt_double_to_ptr(a);                        \
      __enzyme_fp *mb = __enzyme_fprt_double_to_ptr(b);                        \
      int ret = mpfr_cmp(ma->v, mb->v);                                        \
      return ret CMP;                                                          \
    } else {                                                                   \
      abort();                                                                 \
    }                                                                          \
  }

__ENZYME_MPFR_ORIGINAL_ATTRIBUTES
bool __enzyme_fprt_original_64_52_intr_llvm_is_fpclass_f64(double a,
                                                           int32_t tests);
__ENZYME_MPFR_ATTRIBUTES bool
__enzyme_fprt_64_52_intr_llvm_is_fpclass_f64(double a, int32_t tests) {
  return __enzyme_fprt_original_64_52_intr_llvm_is_fpclass_f64(a, tests);
}

#include "flops.def"

#ifdef __cplusplus
}
#endif

#endif // #ifndef __ENZYME_RUNTIME_ENZYME_MPFR__
