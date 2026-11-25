//===- NVPTXTensorCore.cpp - the NVPTX instructions of an in-kernel raise -===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NVPTXTensorCore.h"

#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"

#include <string>

using namespace llvm;

namespace poseidon {
namespace nvptx {

static Intrinsic::ID resolveWmmaIntrinsic(const Twine &name) {
  Intrinsic::ID id = Intrinsic::lookupIntrinsicID(name.str());
  if (id == Intrinsic::not_intrinsic)
    report_fatal_error("resolveWmmaIntrinsic: intrinsic not found: " + name);
  return id;
}

Value *emitThreadIndex(IRBuilderBase &B, Module *M, unsigned axis) {
  static constexpr Intrinsic::ID ids[3] = {Intrinsic::nvvm_read_ptx_sreg_tid_x,
                                           Intrinsic::nvvm_read_ptx_sreg_tid_y,
                                           Intrinsic::nvvm_read_ptx_sreg_tid_z};
  return B.CreateCall(Intrinsic::getOrInsertDeclaration(M, ids[axis]));
}

Value *emitBlockDim(IRBuilderBase &B, Module *M, unsigned axis) {
  static constexpr Intrinsic::ID ids[3] = {
      Intrinsic::nvvm_read_ptx_sreg_ntid_x,
      Intrinsic::nvvm_read_ptx_sreg_ntid_y,
      Intrinsic::nvvm_read_ptx_sreg_ntid_z};
  return B.CreateCall(Intrinsic::getOrInsertDeclaration(M, ids[axis]));
}

Barrier declareBarrier(Module *M) {
  return {Intrinsic::getOrInsertDeclaration(
              M, Intrinsic::nvvm_barrier_cta_sync_aligned_all),
          {ConstantInt::get(Type::getInt32Ty(M->getContext()), 0)}};
}

Value *emitRoundToTF32(IRBuilderBase &B, Module *M, Value *x) {
  // f2tf32.rna gives the i32 bit pattern.
  Function *f2tf32 =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::nvvm_f2tf32_rna);
  return B.CreateBitCast(B.CreateCall(f2tf32, {x}), x->getType());
}

Wmma declareWmma(Module *M, unsigned m, unsigned n, unsigned k,
                 StringRef operand, StringRef accumulator, bool bRowMajor) {
  const std::string shape =
      ("m" + Twine(m) + "n" + Twine(n) + "k" + Twine(k)).str();
  const std::string a = operand.str(), d = accumulator.str();
  const std::string layoutB = bRowMajor ? "row" : "col";
  // An F16 product names its C/D precision pair; the other operand types
  // imply theirs and name only themselves.
  const std::string mmaSuffix = a == "f16" ? d + "." + d : a;
  Intrinsic::ID loadA = resolveWmmaIntrinsic("llvm.nvvm.wmma." + shape +
                                             ".load.a.row.stride." + a);
  Intrinsic::ID loadB = resolveWmmaIntrinsic(
      "llvm.nvvm.wmma." + shape + ".load.b." + layoutB + ".stride." + a);
  Intrinsic::ID mma = resolveWmmaIntrinsic(
      "llvm.nvvm.wmma." + shape + ".mma.row." + layoutB + "." + mmaSuffix);
  Intrinsic::ID storeD = resolveWmmaIntrinsic("llvm.nvvm.wmma." + shape +
                                              ".store.d.row.stride." + d);
  PointerType *genPtrTy = PointerType::get(M->getContext(), /*addrspace=*/0);
  return {Intrinsic::getOrInsertDeclaration(M, loadA, {genPtrTy}),
          Intrinsic::getOrInsertDeclaration(M, loadB, {genPtrTy}),
          Intrinsic::getOrInsertDeclaration(M, mma),
          Intrinsic::getOrInsertDeclaration(M, storeD, {genPtrTy})};
}

} // namespace nvptx
} // namespace poseidon
