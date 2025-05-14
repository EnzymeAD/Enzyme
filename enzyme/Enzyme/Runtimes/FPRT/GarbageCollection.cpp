//===- Trace.cpp - FLOP Garbage collection wrappers
//---------------------------------===//
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

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <list>
#include <llvm/Support/ErrorHandling.h>
#include <stdint.h>
#include <stdlib.h>

#define ENZYME_FPRT_ENABLE_GARBAGE_COLLECTION

#include <enzyme/fprt/fprt.h>
#include <enzyme/fprt/mpfr.h>

struct GCFloatTy {
  __enzyme_fp fp;
  bool seen;
  GCFloatTy() : seen(false) {}
  ~GCFloatTy() {}
};
struct {
  std::list<GCFloatTy> all;
  void clear() { all.clear(); }

} __enzyme_mpfr_fps;

double __enzyme_fprt_64_52_get(double _a, int64_t exponent, int64_t significand,
                               int64_t mode, const char *loc) {
  __enzyme_fp *a = __enzyme_fprt_double_to_ptr(_a);
  return mpfr_get_d(a->result, __ENZYME_MPFR_DEFAULT_ROUNDING_MODE);
}

double __enzyme_fprt_64_52_new(double _a, int64_t exponent, int64_t significand,
                               int64_t mode, const char *loc) {
  __enzyme_mpfr_fps.all.push_back({});
  __enzyme_fp *a = &__enzyme_mpfr_fps.all.back().fp;
  mpfr_init2(a->result, significand);
  mpfr_set_d(a->result, _a, __ENZYME_MPFR_DEFAULT_ROUNDING_MODE);
  return __enzyme_fprt_ptr_to_double(a);
}

double __enzyme_fprt_64_52_const(double _a, int64_t exponent,
                                 int64_t significand, int64_t mode,
                                 const char *loc) {
  // TODO This should really be called only once for an appearance in the code,
  // currently it is called every time a flop uses a constant.
  return __enzyme_fprt_64_52_new(_a, exponent, significand, mode, loc);
}

__enzyme_fp *__enzyme_fprt_64_52_new_intermediate(int64_t exponent,
                                                  int64_t significand,
                                                  int64_t mode,
                                                  const char *loc) {
  __enzyme_mpfr_fps.all.push_back({});
  __enzyme_fp *a = &__enzyme_mpfr_fps.all.back().fp;
  mpfr_init2(a->result, significand);
  return a;
}

void __enzyme_fprt_64_52_delete(double a, int64_t exponent, int64_t significand,
                                int64_t mode, const char *loc) {
  // ignore for now
}

void enzyme_fprt_gc_dump_status() {
  std::cerr << "Currently " << __enzyme_mpfr_fps.all.size()
            << " floats allocated." << std::endl;
}

void enzyme_fprt_gc_clear_seen() {
  for (auto &gcfp : __enzyme_mpfr_fps.all)
    gcfp.seen = false;
}

double enzyme_fprt_gc_mark_seen(double a) {
  __enzyme_fp *fp = __enzyme_fprt_double_to_ptr(a);
  if (!fp)
    return a;
  intptr_t offset = (char *)&(((GCFloatTy *)nullptr)->fp) - (char *)nullptr;
  GCFloatTy *gcfp = (GCFloatTy *)((char *)fp - offset);
  gcfp->seen = true;
  return a;
}

void enzyme_fprt_gc_doit() {
  for (auto it = __enzyme_mpfr_fps.all.begin();
       it != __enzyme_mpfr_fps.all.end();) {
    if (!it->seen) {
      mpfr_clear(it->fp.result);
      it = __enzyme_mpfr_fps.all.erase(it);
    } else {
      ++it;
    }
  }
  enzyme_fprt_gc_clear_seen();
}
