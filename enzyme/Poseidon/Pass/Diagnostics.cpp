//===- Diagnostics.cpp - how the pass reports to the compile --------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Diagnostics.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/LLVMContext.h"

using namespace llvm;

namespace poseidon {

void diagnose(const Function &F, const DiagnosticLocation &Loc,
              const Twine &Message, DiagnosticSeverity Severity) {
  F.getContext().diagnose(DiagnosticInfoUnsupported(
      F, "Poseidon: " + Message.str(), Loc, Severity));
}

char UserError::ID = 0;

Error userError(const Function &F, const DiagnosticLocation &Loc,
                const Twine &Message) {
  return make_error<UserError>(&F, Loc, Message.str());
}

Error userError(const Instruction &I, const Twine &Message) {
  return userError(*I.getFunction(), I.getDebugLoc(), Message);
}

Error userError(const Twine &Message) {
  return make_error<UserError>(nullptr, DiagnosticLocation(), Message.str());
}

void diagnoseUserErrors(LLVMContext &Ctx, Error E) {
  handleAllErrors(
      std::move(E),
      [&](const UserError &U) {
        if (U.F)
          diagnose(*U.F, U.Loc, U.Message, DS_Error);
        else
          Ctx.diagnose(
              DiagnosticInfoGeneric("Poseidon: " + U.Message, DS_Error));
      },
      [&](const ErrorInfoBase &Other) {
        Ctx.diagnose(
            DiagnosticInfoGeneric("Poseidon: " + Other.message(), DS_Error));
      });
}

} // namespace poseidon
