//===- Trace.cpp - FLOP Tracing wrappers ---------------------------------===//
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
// This file contains infrastructure for flop tracing
//
// It is implemented as a .cpp file and not as a header becaues we want to use
// C++ features and still be able to use it in C code.
//
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <iostream>
#include <list>
#include <stdint.h>
#include <stdlib.h>
#include <vector>

#include "enzyme/fprt/fprt.h"

#define __ENZYME_MPFR_ATTRIBUTES __attribute__((weak))
#define __ENZYME_MPFR_ORIGINAL_ATTRIBUTES __attribute__((weak))

#define VERBOSE 1

extern "C" {
typedef struct __enzyme_fp {
  double v;
} __enzyme_fp;
}

static void print_enzyme_fp(std::ostream &out, __enzyme_fp *fp) {
  out << "[" << fp << ":" << fp->v << "]";
}

template <typename T>
static void __enzyme_fprt_trace_flop_in(std::vector<T> inputs,
                                        const char *name) {
  std::cerr << name << "(";
  bool seen = false;
  for (T input : inputs) {
    if (seen)
      std::cerr << ", ";
    seen = true;
    __enzyme_fp *fp = __enzyme_fprt_double_to_ptr(input);
    print_enzyme_fp(std::cerr, fp);
  }
  std::cerr << ")";
}

template <typename T>
static void __enzyme_fprt_trace_flop(std::vector<T> inputs, const char *name) {
  if (!VERBOSE)
    return;
  __enzyme_fprt_trace_flop_in(inputs, name);
  std::cerr << std::endl;
}

template <typename T>
static void __enzyme_fprt_trace_flop(std::vector<T> inputs, T output,
                                     const char *name) {
  if (!VERBOSE)
    return;
  __enzyme_fprt_trace_flop_in(inputs, name);
  std::cerr << " -> ";
  __enzyme_fp *fp = __enzyme_fprt_double_to_ptr(output);
  print_enzyme_fp(std::cerr, fp);
  std::cerr << std::endl;
}

// TODO ultimately we probably want a linked list of arrays or something like
// that for this (std::list probably is that but we may want our own impl)
static std::list<__enzyme_fp> FPs;

extern "C" {

__ENZYME_MPFR_ATTRIBUTES
double __enzyme_fprt_64_52_get(double _a, int64_t exponent, int64_t significand,
                               int64_t mode, const char *loc) {
  __enzyme_fp *a = __enzyme_fprt_double_to_ptr(_a);
  __enzyme_fprt_trace_flop<double>({_a}, "get");
  return a->v;
}

__ENZYME_MPFR_ATTRIBUTES
double __enzyme_fprt_64_52_new(double _a, int64_t exponent, int64_t significand,
                               int64_t mode, const char *loc) {
  FPs.push_back({_a});
  __enzyme_fp *a = &FPs.back();
  auto ret = __enzyme_fprt_ptr_to_double(a);
  __enzyme_fprt_trace_flop({}, ret, "new");
  return ret;
}

__ENZYME_MPFR_ATTRIBUTES
double __enzyme_fprt_64_52_const(double _a, int64_t exponent,
                                 int64_t significand, int64_t mode,
                                 const char *loc) {
  // TODO This should really be called only once for an appearance in the code,
  // currently it is called every time a flop uses a constant.
  auto ret = __enzyme_fprt_64_52_new(_a, exponent, significand, mode, loc);
  __enzyme_fprt_trace_flop({}, ret, "const");
  return ret;
}

__ENZYME_MPFR_ATTRIBUTES
__enzyme_fp *__enzyme_fprt_64_52_new_intermediate(int64_t exponent,
                                                  int64_t significand,
                                                  int64_t mode,
                                                  const char *loc) {
  FPs.push_back({0});
  __enzyme_fp *a = &FPs.back();
  return a;
}

__ENZYME_MPFR_ATTRIBUTES
void __enzyme_fprt_64_52_delete(double a, int64_t exponent, int64_t significand,
                                int64_t mode, const char *loc) {
  // TODO
  __enzyme_fprt_trace_flop<double>({a}, "delete");
}

__ENZYME_MPFR_ATTRIBUTES
void __enzyme_fprt_delete_all() { FPs.clear(); }

#define __ENZYME_MPFR_SINGOP(OP_TYPE, LLVM_OP_NAME, MPFR_FUNC_NAME, FROM_TYPE, \
                             RET, MPFR_GET, ARG1, MPFR_SET_ARG1,               \
                             ROUNDING_MODE)                                    \
  __ENZYME_MPFR_ATTRIBUTES                                                     \
  RET __enzyme_fprt_original_##FROM_TYPE##_##OP_TYPE##_##LLVM_OP_NAME(ARG1 a); \
  __ENZYME_MPFR_ATTRIBUTES                                                     \
  RET __enzyme_fprt_##FROM_TYPE##_##OP_TYPE##_##LLVM_OP_NAME(                  \
      ARG1 a, int64_t exponent, int64_t significand, int64_t mode,             \
      const char *loc) {                                                       \
    RET res = __enzyme_fprt_original_##FROM_TYPE##_##OP_TYPE##_##LLVM_OP_NAME( \
        __enzyme_fprt_double_to_ptr(a)->v);                                    \
    __enzyme_fp *intermediate = __enzyme_fprt_64_52_new_intermediate(          \
        exponent, significand, mode, loc);                                     \
    intermediate->v = res;                                                     \
    double ret = __enzyme_fprt_ptr_to_double(intermediate);                    \
    __enzyme_fprt_trace_flop({a}, ret, #LLVM_OP_NAME);                         \
    return ret;                                                                \
  }

// TODO this is a bit sketchy if the user cast their float to int before calling
// this. We need to detect these patterns
#define __ENZYME_MPFR_BIN_INT(OP_TYPE, LLVM_OP_NAME, MPFR_FUNC_NAME,           \
                              FROM_TYPE, RET, MPFR_GET, ARG1, MPFR_SET_ARG1,   \
                              ARG2, ROUNDING_MODE)                             \
  __ENZYME_MPFR_ORIGINAL_ATTRIBUTES                                            \
  RET __enzyme_fprt_original_##FROM_TYPE##_##OP_TYPE##_##LLVM_OP_NAME(ARG1 a,  \
                                                                      ARG2 b); \
  __ENZYME_MPFR_ATTRIBUTES RET                                                 \
      __enzyme_fprt_##FROM_TYPE##_##OP_TYPE##_##LLVM_OP_NAME(                  \
          ARG1 a, ARG2 b, int64_t exponent, int64_t significand, int64_t mode, \
          const char *loc) {                                                   \
    RET res = __enzyme_fprt_original_##FROM_TYPE##_##OP_TYPE##_##LLVM_OP_NAME( \
        __enzyme_fprt_double_to_ptr(a)->v, __enzyme_fprt_double_to_ptr(b)->v); \
    __enzyme_fp *intermediate = __enzyme_fprt_64_52_new_intermediate(          \
        exponent, significand, mode, loc);                                     \
    intermediate->v = res;                                                     \
    double ret = __enzyme_fprt_ptr_to_double(intermediate);                    \
    __enzyme_fprt_trace_flop({a}, ret, #LLVM_OP_NAME);                         \
    return ret;                                                                \
  }

#define __ENZYME_MPFR_BIN(OP_TYPE, LLVM_OP_NAME, MPFR_FUNC_NAME, FROM_TYPE,    \
                          RET, MPFR_GET, ARG1, MPFR_SET_ARG1, ARG2,            \
                          MPFR_SET_ARG2, ROUNDING_MODE)                        \
  __ENZYME_MPFR_ORIGINAL_ATTRIBUTES                                            \
  RET __enzyme_fprt_original_##FROM_TYPE##_##OP_TYPE##_##LLVM_OP_NAME(ARG1 a,  \
                                                                      ARG2 b); \
  __ENZYME_MPFR_ATTRIBUTES                                                     \
  RET __enzyme_fprt_##FROM_TYPE##_##OP_TYPE##_##LLVM_OP_NAME(                  \
      ARG1 a, ARG2 b, int64_t exponent, int64_t significand, int64_t mode,     \
      const char *loc) {                                                       \
    RET res = __enzyme_fprt_original_##FROM_TYPE##_##OP_TYPE##_##LLVM_OP_NAME( \
        __enzyme_fprt_double_to_ptr(a)->v, __enzyme_fprt_double_to_ptr(b)->v); \
    __enzyme_fp *intermediate = __enzyme_fprt_64_52_new_intermediate(          \
        exponent, significand, mode, loc);                                     \
    intermediate->v = res;                                                     \
    double ret = __enzyme_fprt_ptr_to_double(intermediate);                    \
    __enzyme_fprt_trace_flop({a, b}, ret, #LLVM_OP_NAME);                      \
    return ret;                                                                \
  }

#define __ENZYME_MPFR_FMULADD(LLVM_OP_NAME, FROM_TYPE, TYPE, MPFR_TYPE,        \
                              LLVM_TYPE, ROUNDING_MODE)                        \
  __ENZYME_MPFR_ORIGINAL_ATTRIBUTES                                            \
  TYPE __enzyme_fprt_original_##FROM_TYPE##_##OP_TYPE##_##LLVM_OP_NAME(        \
      TYPE a, TYPE b, TYPE c);                                                 \
  __ENZYME_MPFR_ATTRIBUTES                                                     \
  TYPE __enzyme_fprt_##FROM_TYPE##_intr_##LLVM_OP_NAME##_##LLVM_TYPE(          \
      TYPE a, TYPE b, TYPE c, int64_t exponent, int64_t significand,           \
      int64_t mode, const char *loc) {                                         \
    TYPE res =                                                                 \
        __enzyme_fprt_original_##FROM_TYPE##_##OP_TYPE##_##LLVM_OP_NAME(       \
            __enzyme_fprt_double_to_ptr(a)->v,                                 \
            __enzyme_fprt_double_to_ptr(b)->v,                                 \
            __enzyme_fprt_double_to_ptr(c)->v);                                \
    __enzyme_fp *intermediate = __enzyme_fprt_64_52_new_intermediate(          \
        exponent, significand, mode, loc);                                     \
    intermediate->v = res;                                                     \
    double ret = __enzyme_fprt_ptr_to_double(intermediate);                    \
    __enzyme_fprt_trace_flop({a, b, c}, res, #LLVM_OP_NAME);                   \
    return ret;                                                                \
  }

#define __ENZYME_MPFR_FCMP_IMPL(NAME, ORDERED, CMP, FROM_TYPE, TYPE, MPFR_GET, \
                                ROUNDING_MODE)                                 \
  __ENZYME_MPFR_ORIGINAL_ATTRIBUTES                                            \
  bool __enzyme_fprt_original_##FROM_TYPE##_fcmp_##NAME(TYPE a, TYPE b);       \
  __ENZYME_MPFR_ATTRIBUTES                                                     \
  bool __enzyme_fprt_##FROM_TYPE##_fcmp_##NAME(                                \
      TYPE a, TYPE b, int64_t exponent, int64_t significand, int64_t mode,     \
      const char *loc) {                                                       \
    bool res = __enzyme_fprt_original_##FROM_TYPE##_fcmp_##NAME(               \
        __enzyme_fprt_double_to_ptr(a)->v, __enzyme_fprt_double_to_ptr(b)->v); \
    __enzyme_fprt_trace_flop<TYPE>({a, b}, "fcmp_" #NAME);                     \
    return res;                                                                \
  }

__ENZYME_MPFR_ORIGINAL_ATTRIBUTES
bool __enzyme_fprt_original_64_52_intr_llvm_is_fpclass_f64(double a,
                                                           int32_t tests);
__ENZYME_MPFR_ATTRIBUTES bool __enzyme_fprt_64_52_intr_llvm_is_fpclass_f64(
    double a, int32_t tests, int64_t exponent, int64_t significand,
    int64_t mode, const char *loc) {
  __enzyme_fprt_trace_flop<double>({a}, "llvm_is_fpclass_f64");
  return __enzyme_fprt_original_64_52_intr_llvm_is_fpclass_f64(
      __enzyme_fprt_double_to_ptr(a)->v, tests);
}

#include "enzyme/fprt/flops.def"

} // extern "C"
