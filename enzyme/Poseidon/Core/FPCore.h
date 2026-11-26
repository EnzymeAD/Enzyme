//===- FPCore.h - FP expressions as FPCore and as C -----------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
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

#ifndef POSEIDON_CORE_FPCORE_H
#define POSEIDON_CORE_FPCORE_H

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

/// An operator of Herbie's FPCore as the pass models it: its operand count
/// and whether it yields a truth value, and how the lowering and the
/// evaluators compute it. A consumer computes an operator whose column it
/// reads is empty by its name.
struct FPCoreOperator {
  /// The FPCore name ("+", "sin", "if").
  llvm::StringLiteral Name;
  /// The operand count; 0 for a constant and for an array, whose operand
  /// count is that of its elements.
  unsigned Arity = 0;
  /// Whether the operator yields a truth value.
  bool Boolean = false;
  /// The intrinsic of Arity operands the operator lowers to; not_intrinsic for
  /// none.
  llvm::Intrinsic::ID Intrinsic = llvm::Intrinsic::not_intrinsic;
  /// Whether the operator lowers to the math-library function of its name.
  bool LibraryCall = false;
  /// The evaluation of a unary operator in binary64.
  double (*F64Unary)(double) = nullptr;
  /// The evaluation of a unary operator in binary32.
  float (*F32Unary)(float) = nullptr;
  /// The evaluation of a unary operator in MPFR.
  int (*MPFRUnary)(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t) = nullptr;
  /// The evaluation of a binary operator in binary64.
  double (*F64Binary)(double, double) = nullptr;
  /// The evaluation of a binary operator in binary32.
  float (*F32Binary)(float, float) = nullptr;
  /// The evaluation of a binary operator in MPFR.
  int (*MPFRBinary)(mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t) = nullptr;
  /// A comparison's fcmp predicate.
  llvm::CmpInst::Predicate Predicate = llvm::CmpInst::BAD_FCMP_PREDICATE;
  /// A comparison's lowered fcmp is named "fpcore." followed by this.
  llvm::StringLiteral LoweredName = "";
  /// A constant's value in binary64.
  double Constant = 0;
  /// A constant's value in MPFR.
  int (*MPFRConstant)(mpfr_ptr, mpfr_rnd_t) = nullptr;
};

/// The row of `Name`, or null for an operator the pass does not model.
const FPCoreOperator *fpcoreOperator(llvm::StringRef Name);

/// The rows the model treats by name.
namespace fpcore {
extern const FPCoreOperator &Neg, &Add, &Sub, &Mul, &Div, &Pow, &Sqrt, &Fma,
    &And, &Or, &Not, &True, &False, &If, &Array, &ToBinary32, &ToBinary64;
} // namespace fpcore

/// The expression graph of the FPCore expression `expr`, over the values its
/// symbols name in `G`, which owns the nodes the parse creates. An error,
/// saying what and where, for an expression that is not one the evaluators and
/// the lowering model: an unknown symbol or operator, an operator of an
/// unmodelled format or with the wrong operand count, a truth value where a
/// number belongs or the reverse, or a malformed literal, approx form or array.
/// Each `#s(approx spec result)` adds its (spec, result) pair to `approxSpecs`
/// when given; with `subgraphInputs`, an operation that recomputes the
/// definition of one of those inputs is that input's node.
llvm::Expected<FPNode *> parseHerbieExpr(
    llvm::StringRef expr, FPGraph &G,
    llvm::SmallVectorImpl<std::pair<std::string, std::string>> *approxSpecs =
        nullptr,
    const llvm::SetVector<llvm::Value *> *subgraphInputs = nullptr);

/// The graph dtype of `T` ("f64"), or empty when `T` is no floating-point type
/// of a format of the pass.
std::string dtypeOfType(llvm::Type *T);
/// The C type of an FP dtype; "double" for any other.
llvm::StringRef cTypeName(llvm::StringRef dtype);

/// How renderC spells an expression graph as C.
struct CRendering {
  /// The values rendered as leaves even when they are computed; none when
  /// null.
  const llvm::SetVector<llvm::Value *> *inputs = nullptr;
  /// The C name of a leaf value.
  std::function<std::string(llvm::Value *)> leafName;
  /// The dtype each node is rendered at, where it differs from its own; none
  /// when null.
  const llvm::DenseMap<const FPNode *, std::string> *dtypes = nullptr;
};

/// `node` as a C expression, cast to `resultDtype` when that is an FP dtype
/// other than its own; empty when the expression is too large to render.
std::string renderC(const FPNode &node, const CRendering &how,
                    llvm::StringRef resultDtype = "");

/// The FPCore expression `expr` as C, one string per element of an array and
/// one otherwise; empty when `expr` names a symbol `G` does not know or does
/// not parse. The parse adds to `G`, as parseHerbieExpr does.
llvm::SmallVector<std::string, 1> fpcoreToC(
    llvm::StringRef expr, FPGraph &G, const CRendering &how,
    llvm::StringRef resultDtype = "",
    llvm::SmallVectorImpl<std::pair<std::string, std::string>> *approxSpecs =
        nullptr);

/// The FPCore name of the operation `I`, which must be a floating-point
/// arithmetic instruction or a call of a function.
std::string getHerbieOperator(const llvm::Instruction &I);
/// The FPCore name of `N`'s operation: its operator's, or for an operation of
/// the program the operator table does not model, getHerbieOperator's; empty
/// for a node that is no operation.
std::string operationName(const FPNode &N);

/// The name of intrinsic `Callee` without its "llvm." prefix and type suffixes:
/// "sin" for llvm.sin.f64.
llvm::StringRef intrinsicBaseName(const llvm::Function &Callee);

/// Adds to `args` each symbol of the form v<N> that `expr` names.
void getUniqueArgs(const std::string &expr,
                   llvm::SmallSet<std::string, 8> &args);

} // namespace poseidon
#endif // POSEIDON_CORE_FPCORE_H
