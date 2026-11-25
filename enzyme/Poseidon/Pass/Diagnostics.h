//===- Diagnostics.h - how the pass reports to the compile ----------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_PASS_DIAGNOSTICS_H
#define POSEIDON_PASS_DIAGNOSTICS_H

#include "llvm/ADT/Twine.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace llvm {
class Function;
class Instruction;
class LLVMContext;
} // namespace llvm

namespace poseidon {

/// The pass name of the pass's optimization remarks, which
/// -Rpass=poseidon, -Rpass-missed=poseidon and -Rpass-analysis=poseidon
/// enable.
constexpr const char *kRemarkPass = "poseidon";

/// A diagnostic of `Severity` at `Loc` in `F`, its message prefixed with
/// "Poseidon: ".
void diagnose(const llvm::Function &F, const llvm::DiagnosticLocation &Loc,
              const llvm::Twine &Message, llvm::DiagnosticSeverity Severity);

/// A missed-optimization remark `Name` at `Loc` in `F`: what the pass left as
/// written there, and why.
void remarkMissed(const llvm::Function &F, const llvm::DiagnosticLocation &Loc,
                  llvm::StringRef Name, const llvm::Twine &Message);

/// A mistake in what the compile was given: where it is and what it is.
///
/// A function that finds one returns it, and every caller passes it on, up to
/// the entry of the running pass. There stopOnError (Pass/Session.h) reports it
/// once through diagnoseUserErrors, leaves the module valid, and marks the
/// session stopped, so that the session's later passes leave the module as it
/// is. The compile fails through the diagnostic handler's error count, and the
/// process that loaded the plugin is never ended by the pass. The one place an
/// error is held rather than returned is the DP's search, which cannot unwind:
/// a DeferredError (Solve/DP.h) keeps the first one it meets, a repricing
/// that fails or a cost that leaves the DP's axis, and solveSites returns it
/// after the search.
class UserError : public llvm::ErrorInfo<UserError> {
public:
  static char ID;
  /// At `Loc` in `F`; a mistake no site is concerned in has no function.
  UserError(const llvm::Function *F, llvm::DiagnosticLocation Loc,
            std::string Message)
      : F(F), Loc(Loc), Message(std::move(Message)) {}
  void log(llvm::raw_ostream &OS) const override { OS << Message; }
  std::error_code convertToErrorCode() const override {
    return llvm::inconvertibleErrorCode();
  }

  const llvm::Function *F;
  llvm::DiagnosticLocation Loc;
  std::string Message;
};

llvm::Error userError(const llvm::Function &F,
                      const llvm::DiagnosticLocation &Loc,
                      const llvm::Twine &Message);
llvm::Error userError(const llvm::Instruction &I, const llvm::Twine &Message);
/// A mistake no site is concerned in: a flag value, a plugin or a file of the
/// whole compile.
llvm::Error userError(const llvm::Twine &Message);

/// Reports each error `E` holds as an error diagnostic in `Ctx`: a user error
/// at its location, any other as a mistake of the whole compile.
void diagnoseUserErrors(llvm::LLVMContext &Ctx, llvm::Error E);

} // namespace poseidon
#endif // POSEIDON_PASS_DIAGNOSTICS_H
