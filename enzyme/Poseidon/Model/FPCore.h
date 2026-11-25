//===- FPCore.h - FP expressions as FPCore and as C -----------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The textual forms of the expression graph: FPCore, the language Herbie reads
// and answers in, with its operator names for LLVM instructions, and the C an
// explain record shows a rewrite as.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_MODEL_FP_CORE_H
#define POSEIDON_MODEL_FP_CORE_H

#include "FPNode.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Error.h"

#include <functional>
#include <memory>
#include <mpfr.h>
#include <string>
#include <unordered_map>

namespace llvm {
class Function;
class Instruction;
class Type;
} // namespace llvm

namespace poseidon {

// An operator of Herbie's FPCore as the pass models it: its operand count
// (any positive number for an array) and whether it yields a truth value, and
// how the lowering and the evaluators compute it. A consumer computes an
// operator whose column it reads is empty by its name.
struct FPCoreOperator {
  llvm::StringLiteral Name;
  unsigned Arity = 0;
  bool Boolean = false;
  // The lowering: an intrinsic of Arity operands, or a math-library call.
  llvm::Intrinsic::ID Intrinsic = llvm::Intrinsic::not_intrinsic;
  bool LibraryCall = false;
  // The evaluation in binary64, in binary32 and in MPFR.
  double (*F64Unary)(double) = nullptr;
  float (*F32Unary)(float) = nullptr;
  int (*MPFRUnary)(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t) = nullptr;
  double (*F64Binary)(double, double) = nullptr;
  float (*F32Binary)(float, float) = nullptr;
  int (*MPFRBinary)(mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t) = nullptr;
  // A comparison: its fcmp predicate and the name of the lowered fcmp.
  llvm::CmpInst::Predicate Predicate = llvm::CmpInst::BAD_FCMP_PREDICATE;
  llvm::StringLiteral LoweredName = "";
  // A constant: its binary64 value and its value in MPFR.
  double Constant = 0;
  int (*MPFRConstant)(mpfr_ptr, mpfr_rnd_t) = nullptr;
};

// The row of `Name`, or null for an operator the pass does not model.
const FPCoreOperator *fpcoreOperator(llvm::StringRef Name);

// The rows the model treats by name.
namespace fpcore {
extern const FPCoreOperator &Neg, &Add, &Sub, &Mul, &Div, &Pow, &Sqrt, &Fma,
    &And, &Or, &Not, &True, &False, &If, &Array, &ToBinary32, &ToBinary64;
} // namespace fpcore

// The expression graph of the FPCore expression `expr`, over the values its
// symbols name. An error, saying what and where, for an expression that is not
// one the evaluators and the lowering model: an unknown symbol or operator, an
// operator of an unmodelled format or with the wrong operand count, a truth
// value where a number belongs or the reverse, or a malformed literal, approx
// form or array.
llvm::Expected<std::shared_ptr<FPNode>> parseHerbieExpr(
    llvm::StringRef expr, FPGraph &G,
    llvm::SmallVectorImpl<std::pair<std::string, std::string>> *approxSpecs =
        nullptr,
    const llvm::SetVector<llvm::Value *> *subgraphInputs = nullptr);

std::string dtypeOfType(llvm::Type *T);
// The C type of an FP dtype; "double" for any other.
llvm::StringRef cTypeName(llvm::StringRef dtype);

struct CRendering {
  const llvm::SetVector<llvm::Value *> *inputs = nullptr;
  std::function<std::string(llvm::Value *)> leafName;
  const llvm::DenseMap<const FPNode *, std::string> *dtypes = nullptr;
};

std::string renderC(const FPNode &node, const CRendering &how,
                    llvm::StringRef resultDtype = "");

llvm::SmallVector<std::string, 1> fpcoreToC(
    llvm::StringRef expr, FPGraph &G, const CRendering &how,
    llvm::StringRef resultDtype = "",
    llvm::SmallVectorImpl<std::pair<std::string, std::string>> *approxSpecs =
        nullptr);

std::string getHerbieOperator(const llvm::Instruction &I);
// The FPCore name of `N`'s operation: its operator's, or for an operation of
// the program the operator table does not model, getHerbieOperator's; empty
// for a node that is no operation.
std::string operationName(const FPNode &N);

// The name of intrinsic `Callee` without its "llvm." prefix and type suffixes:
// "sin" for llvm.sin.f64.
llvm::StringRef intrinsicBaseName(const llvm::Function &Callee);

void getUniqueArgs(const std::string &expr,
                   llvm::SmallSet<std::string, 8> &args);

} // namespace poseidon
#endif // POSEIDON_MODEL_FP_CORE_H
