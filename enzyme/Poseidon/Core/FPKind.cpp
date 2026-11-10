//=- FPKind.cpp - Number formats and rounding to them ---------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FPKind.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace poseidon {

const char *fpKindName(FPKind k) {
  switch (k) {
  case FPKind::F16:
    return "f16";
  case FPKind::BF16:
    return "bf16";
  case FPKind::TF32:
    return "tf32";
  case FPKind::F32:
    return "f32";
  case FPKind::F64:
    return "f64";
  case FPKind::S8:
    return "s8";
  case FPKind::S32:
    return "s32";
  case FPKind::Invalid:
    return "invalid";
  }
  llvm_unreachable("unknown FPKind value");
}

// nullptr for kinds with no FP rounding semantics (F64, Invalid, S8/S32).
static const fltSemantics *semanticsForKind(FPKind k) {
  switch (k) {
  case FPKind::F16:
    return &APFloat::IEEEhalf();
  case FPKind::BF16:
    return &APFloat::BFloat();
  case FPKind::TF32:
    return &APFloat::FloatTF32();
  case FPKind::F32:
    return &APFloat::IEEEsingle();
  case FPKind::F64:
  case FPKind::Invalid:
  case FPKind::S8:
  case FPKind::S32:
    return nullptr;
  }
  return nullptr;
}

double roundToPrec(double x, FPKind k) {
  if (k == FPKind::S8 || k == FPKind::S32)
    report_fatal_error(
        "roundToPrec: integer FPKind (S8/S32) has no FP rounding semantics");
  const fltSemantics *sem = semanticsForKind(k);
  if (!sem)
    return x;
  bool losesInfo;
  APFloat ap(x);
  ap.convert(*sem, APFloat::rmNearestTiesToEven, &losesInfo);
  ap.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven, &losesInfo);
  return ap.convertToDouble();
}

double minSubnormalForKind(FPKind k) {
  const fltSemantics *sem = semanticsForKind(k);
  if (!sem)
    return 0.0;
  APFloat sm = APFloat::getSmallest(*sem);
  bool li;
  sm.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven, &li);
  return sm.convertToDouble();
}

} // namespace poseidon
