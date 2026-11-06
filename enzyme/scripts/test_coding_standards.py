#!/usr/bin/env python3
"""Unit and integration tests for Enzyme LLVM/MLIR coding-standards linters and clang-tidy configs."""

import contextlib
from dataclasses import dataclass
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable, List, Optional, Sequence, Set
import unittest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENZYME_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import check_llvm_coding_standards as llvm_lint
import check_mlir_coding_standards as mlir_lint
import run_clang_tidy as tidy_runner


def discover_usable_clang_tidy() -> Optional[str]:
    """Return a `clang-tidy` binary supporting `--experimental-custom-checks`, or None."""
    try:
        cand = tidy_runner.find_clang_tidy()
    except FileNotFoundError:
        return None
    try:
        proc = subprocess.run(
            [cand, "--experimental-custom-checks", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return cand if proc.returncode == 0 else None


CLANG_TIDY_BIN = discover_usable_clang_tidy()

DIAG_RE = re.compile(
    r"^(?P<file>.+?):(?P<line>\d+):(?P<col>\d+):\s+"
    r"(?P<severity>warning|error):\s+(?P<message>.+?)\s+\[(?P<checks>[^\]]+)\]$"
)


def _git_init(repo_dir: str) -> None:
    subprocess.check_call(["git", "init", "-q"], cwd=repo_dir)
    subprocess.check_call(
        ["git", "config", "user.email", "test@example.com"], cwd=repo_dir
    )
    subprocess.check_call(["git", "config", "user.name", "Test User"], cwd=repo_dir)


def _git_commit_all(repo_dir: str, message: str) -> None:
    subprocess.check_call(["git", "add", "-A"], cwd=repo_dir)
    subprocess.check_call(["git", "commit", "-q", "-m", message], cwd=repo_dir)


def _run_main_captured(fn, argv: Sequence[str]):
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
        rc = fn(list(argv))
    return rc, out_buf.getvalue(), err_buf.getvalue()


MOCK_LLVM_MLIR_H = r"""#ifndef MOCK_LLVM_MLIR_ALL_H
#define MOCK_LLVM_MLIR_ALL_H

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace llvm {

class Value {
public:
  virtual ~Value() = default;
};

class StringRef {
  const char *Data = "";
  size_t Length = 0;

public:
  constexpr StringRef() = default;
  StringRef(const char *Str) : Data(Str ? Str : "") {
    if (Str) {
      while (Str[Length] != '\0')
        ++Length;
    }
  }
  StringRef(const std::string &Str) : Data(Str.data()), Length(Str.size()) {}
  const char *data() const { return Data; }
  size_t size() const { return Length; }
  bool empty() const { return Length == 0; }
};

class Twine {
public:
  Twine() = default;
  Twine(const char *) {}
  Twine(const std::string &) {}
  Twine(const StringRef &) {}
  Twine concat(const Twine &) const { return Twine(); }
};

inline Twine operator+(const Twine &LHS, const Twine &RHS) {
  return LHS.concat(RHS);
}

inline Twine operator+(const StringRef &LHS, const char *RHS) {
  return Twine(LHS).concat(Twine(RHS));
}

template <typename T>
class SmallVectorImpl {
protected:
  std::vector<T> Storage;
  SmallVectorImpl() = default;

public:
  using iterator = typename std::vector<T>::iterator;
  using const_iterator = typename std::vector<T>::const_iterator;

  iterator begin() { return Storage.begin(); }
  iterator end() { return Storage.end(); }
  const_iterator begin() const { return Storage.begin(); }
  const_iterator end() const { return Storage.end(); }
  size_t size() const { return Storage.size(); }
  bool empty() const { return Storage.empty(); }
  T &front() { return Storage.front(); }
  const T &front() const { return Storage.front(); }
  T &back() { return Storage.back(); }
  const T &back() const { return Storage.back(); }
  void push_back(const T &Elt) { Storage.push_back(Elt); }
  iterator erase(iterator It) { return Storage.erase(It); }
  void remove(const T &) {}
};

template <typename T, unsigned N = 4>
class SmallVector : public SmallVectorImpl<T> {
public:
  SmallVector() = default;
  SmallVector(const SmallVector &) = default;
  SmallVector(SmallVector &&) noexcept = default;
  SmallVector &operator=(const SmallVector &) = default;
  SmallVector &operator=(SmallVector &&) noexcept = default;
};

template <typename T>
class ArrayRef {
  const T *Data = nullptr;
  size_t Length = 0;

public:
  constexpr ArrayRef() = default;
  ArrayRef(const T &Single) : Data(&Single), Length(1) {}
  ArrayRef(const SmallVectorImpl<T> &Vec)
      : Data(Vec.empty() ? nullptr : &*Vec.begin()), Length(Vec.size()) {}
  ArrayRef(const std::vector<T> &Vec)
      : Data(Vec.data()), Length(Vec.size()) {}
  size_t size() const { return Length; }
  bool empty() const { return Length == 0; }
};

template <typename T>
struct DenseMapInfo {};

template <typename KeyT, typename ValueT>
class DenseMap {
public:
  bool erase(const KeyT &) { return false; }
  void clear() {}
};

template <typename... PTs>
class PointerUnion {
public:
  PointerUnion() = default;
};

class raw_ostream {
public:
  virtual ~raw_ostream() = default;
  raw_ostream &operator<<(char) { return *this; }
  raw_ostream &operator<<(StringRef) { return *this; }
  raw_ostream &operator<<(const char *Str) { return *this << StringRef(Str); }
  raw_ostream &operator<<(const std::string &Str) {
    return *this << StringRef(Str);
  }
  raw_ostream &operator<<(const Twine &) { return *this; }
  raw_ostream &operator<<(int) { return *this; }
  raw_ostream &operator<<(size_t) { return *this; }
};

class raw_fd_ostream : public raw_ostream {};

raw_fd_ostream &errs();
raw_fd_ostream &outs();

template <typename IteratorTy>
inline void sort(IteratorTy Start, IteratorTy End) {
  std::sort(Start, End);
}

template <typename Range>
inline void sort(Range &&R) {
  std::sort(R.begin(), R.end());
}

struct [[nodiscard]] LogicalResult {
  bool IsSuccess = true;
};

inline LogicalResult success(bool IsSuccess = true) { return {IsSuccess}; }
inline LogicalResult failure(bool IsFailure = true) { return {!IsFailure}; }
inline bool succeeded(LogicalResult Res) { return Res.IsSuccess; }
inline bool failed(LogicalResult Res) { return !Res.IsSuccess; }

template <typename T>
class [[nodiscard]] FailureOr : public std::optional<T> {
public:
  FailureOr(LogicalResult) {}
  FailureOr() : FailureOr(failure()) {}
  FailureOr(T &&Val) : std::optional<T>(std::forward<T>(Val)) {}
  FailureOr(const T &Val) : std::optional<T>(Val) {}
  operator LogicalResult() const { return success(this->has_value()); }
};

[[noreturn]] void report_fatal_error(const char *Reason,
                                     bool GenCrashDiag = true);
[[noreturn]] void report_fatal_error(StringRef Reason,
                                     bool GenCrashDiag = true);
[[noreturn]] void report_fatal_error(const Twine &Reason,
                                     bool GenCrashDiag = true);
[[noreturn]] void reportFatalInternalError(const char *Reason);
[[noreturn]] void reportFatalUsageError(const char *Reason);

namespace cl {

template <class DataType>
class parser {};

template <class DataType, bool ExternalStorage = false,
          class ParserClass = parser<DataType>>
class opt {
public:
  explicit opt(const char *) {}
  operator DataType() const { return DataType{}; }
};

template <class DataType, class StorageClass = bool,
          class ParserClass = parser<DataType>>
class list {
public:
  explicit list(const char *) {}
};

} // namespace cl
} // namespace llvm

namespace mlir {

using llvm::ArrayRef;
using llvm::DenseMap;
using llvm::DenseMapInfo;
using llvm::failed;
using llvm::failure;
using llvm::FailureOr;
using llvm::LogicalResult;
using llvm::PointerUnion;
using llvm::SmallVector;
using llvm::SmallVectorImpl;
using llvm::StringRef;
using llvm::succeeded;
using llvm::success;

class MLIRContext {};
class Operation;
class Block;
class Region;

class Type {
public:
  constexpr Type() = default;
  MLIRContext *getContext() const { return nullptr; }
};
class IntegerType : public Type {};
class FloatType : public Type {};

template <typename ConcreteType>
class TypeInterface : public Type {};

class Attribute {
public:
  constexpr Attribute() = default;
  MLIRContext *getContext() const { return nullptr; }
};
class StringAttr : public Attribute {};

template <typename ConcreteAttr>
class AttributeInterface : public Attribute {};

class Location {
public:
  constexpr Location() = default;
  MLIRContext *getContext() const { return nullptr; }
};

struct InFlightDiagnostic {
  operator LogicalResult() const { return failure(); }
};

class Value {
public:
  constexpr Value() = default;
  Value(std::nullptr_t) {}
  Type getType() const { return {}; }
  Location getLoc() const { return {}; }
  MLIRContext *getContext() const { return nullptr; }
  Operation *getDefiningOp() const { return nullptr; }
  template <typename OpTy>
  OpTy getDefiningOp() const {
    return OpTy{};
  }
  void replaceAllUsesWith(Value) const {}
  bool use_empty() const { return true; }
  bool hasOneUse() const { return false; }
  int getUses() const { return 0; }
  int getUsers() const { return 0; }
};

class OpResult : public Value {
public:
  using Value::Value;
};

class BlockArgument : public Value {
public:
  using Value::Value;
};

namespace detail {
template <typename Ty>
class TypedValue : public Value {
public:
  using Value::Value;
  Ty getType() const { return {}; }
};
} // namespace detail

template <typename Ty>
using TypedValue = detail::TypedValue<Ty>;

class OpFoldResult : public PointerUnion<Attribute, Value> {
public:
  OpFoldResult() = default;
  OpFoldResult(Value) {}
  OpFoldResult(Attribute) {}
};

class AffineExpr {};
class AffineMap {};
class IntegerSet {};

class ValueRange {
public:
  ValueRange() = default;
  ValueRange(Value) {}
  size_t size() const { return 0; }
  bool empty() const { return true; }
  Value front() const { return {}; }
  Value back() const { return {}; }
  void replaceAllUsesWith(ValueRange) {}
};

class TypeRange {
public:
  size_t size() const { return 0; }
  bool empty() const { return true; }
};

class OperandRange {
public:
  size_t size() const { return 0; }
  bool empty() const { return true; }
};

class ResultRange {
public:
  size_t size() const { return 0; }
  bool empty() const { return true; }
  void replaceAllUsesWith(ValueRange) {}
  bool use_empty() const { return true; }
  bool hasOneUse() const { return false; }
  int getUses() const { return 0; }
  int getUsers() const { return 0; }
};

class Block {
public:
  int begin() { return 0; }
  void erase() {}
  void moveBefore(Block *) {}
  void remove() {}
};

class Region {
public:
  bool empty() const { return false; }
  void eraseArgument(unsigned) {}
};

struct OperationState {
  OperationState(Location, StringRef) {}
};

class Operation {
public:
  Operation() = default;
  Operation(const Operation &) = delete;
  Operation &operator=(const Operation &) = delete;
  MLIRContext *getContext() { return nullptr; }
  Location getLoc() { return {}; }
  StringRef getName() { return {}; }
  Block *getBlock() { return nullptr; }
  Region *getParentRegion() { return nullptr; }
  unsigned getNumOperands() { return 1; }
  Value getOperand(unsigned) { return {}; }
  unsigned getNumResults() { return 1; }
  OpResult getResult(unsigned) { return {}; }
  ResultRange getResults() { return {}; }
  void erase() {}
  void remove() {}
  void moveBefore(Operation *) {}
  void moveAfter(Operation *) {}
  template <typename ValuesT>
  void replaceAllUsesWith(ValuesT &&) {}
  bool use_empty() { return true; }
  bool hasOneUse() { return false; }
  int getUses() { return 0; }
  int getUsers() { return 0; }
  InFlightDiagnostic emitOpError(const llvm::Twine & = {}) { return {}; }
};

class OpState {
public:
  explicit OpState(Operation *State = nullptr) : State(State) {}
  explicit operator bool() const { return State != nullptr; }
  operator Operation *() const { return State; }
  Operation *operator->() const { return State; }
  Operation *getOperation() const { return State; }
  MLIRContext *getContext() const { return State->getContext(); }
  Location getLoc() const { return State->getLoc(); }
  void erase() { State->erase(); }
  InFlightDiagnostic emitOpError(const llvm::Twine &Msg = {}) {
    return State->emitOpError(Msg);
  }

protected:
  LogicalResult verify() { return success(); }
  LogicalResult verifyRegions() { return success(); }

private:
  Operation *State;
};

namespace OpTrait {
template <typename ConcreteType, template <typename> class TraitType>
class TraitBase {
protected:
  Operation *getOperation() {
    return static_cast<ConcreteType *>(this)->getOperation();
  }
};

template <typename ConcreteType>
class OneResult : public TraitBase<ConcreteType, OneResult> {
public:
  OpResult getResult() { return this->getOperation()->getResult(0); }
  operator Value() { return getResult(); }
  void replaceAllUsesWith(Value NewValue) {
    this->getOperation()->getResult(0).replaceAllUsesWith(NewValue);
  }
};
} // namespace OpTrait

template <typename ConcreteType, template <typename T> class... Traits>
class Op : public OpState, public Traits<ConcreteType>... {
public:
  using OpState::getOperation;
  using OpState::OpState;
};

template <typename ConcreteInterface>
class OpInterface : public OpState {
public:
  using OpState::OpState;
};
class FunctionOpInterface : public OpInterface<FunctionOpInterface> {};

class Builder {
public:
  explicit Builder(MLIRContext *Ctx) : Context(Ctx) {}

protected:
  MLIRContext *Context;
};

class OpBuilder : public Builder {
public:
  struct Listener {};
  class InsertionGuard {
  public:
    explicit InsertionGuard(OpBuilder &) {}
    ~InsertionGuard() = default;
    InsertionGuard(const InsertionGuard &) = delete;
    InsertionGuard &operator=(const InsertionGuard &) = delete;
  };

  explicit OpBuilder(MLIRContext *Ctx, Listener * = nullptr) : Builder(Ctx) {}
  explicit OpBuilder(Operation *Op, Listener * = nullptr)
      : Builder(Op ? Op->getContext() : nullptr) {}
  OpBuilder(Block *, int, Listener * = nullptr) : Builder(nullptr) {}

  static OpBuilder atBlockBegin(Block *Blk, Listener *L = nullptr) {
    return OpBuilder(Blk, Blk ? Blk->begin() : 0, L);
  }

  void setInsertionPoint(Operation *) {}
  Operation *create(const OperationState &) { return nullptr; }

  template <typename OpTy, typename... Args>
  OpTy create(Location, Args &&...) {
    return OpTy{};
  }
};

class ImplicitLocOpBuilder : public OpBuilder {
public:
  template <typename... T>
  ImplicitLocOpBuilder(Location, T &&...Operands)
      : OpBuilder(std::forward<T>(Operands)...) {}
  using OpBuilder::create;
  template <typename OpTy, typename... Args>
  OpTy create(Args &&...) {
    return OpTy{};
  }
};

class RewriterBase : public OpBuilder {
public:
  virtual void replaceOp(Operation *, ValueRange) {}
  virtual void eraseOp(Operation *) {}
  void moveOpBefore(Operation *, Operation *) {}
  void moveOpAfter(Operation *, Operation *) {}
  virtual void replaceAllUsesWith(Value, Value) {}

protected:
  explicit RewriterBase(MLIRContext *Ctx) : OpBuilder(Ctx) {}
  explicit RewriterBase(Operation *Op) : OpBuilder(Op) {}
  virtual ~RewriterBase() = default;
};

class IRRewriter : public RewriterBase {
public:
  explicit IRRewriter(MLIRContext *Ctx) : RewriterBase(Ctx) {}
  explicit IRRewriter(Operation *Op) : RewriterBase(Op) {}
};

class PatternRewriter : public RewriterBase {
public:
  explicit PatternRewriter(MLIRContext *Ctx) : RewriterBase(Ctx) {}
};

class ConversionPatternRewriter : public PatternRewriter {
public:
  explicit ConversionPatternRewriter(MLIRContext *Ctx) : PatternRewriter(Ctx) {}
};

template <typename SourceOp>
struct OpRewritePattern {
  explicit OpRewritePattern(MLIRContext * = nullptr) {}
  virtual ~OpRewritePattern() = default;
  virtual LogicalResult matchAndRewrite(SourceOp Op,
                                        PatternRewriter &Rewriter) const = 0;
};

namespace detail {
class PassOptions {
public:
  template <typename DataType>
  class Option : public llvm::cl::opt<DataType> {
  public:
    template <typename PassT>
    Option(PassT &, const char *) : llvm::cl::opt<DataType>("opt") {}
  };
  template <typename DataType>
  class ListOption : public llvm::cl::list<DataType> {
  public:
    template <typename PassT>
    ListOption(PassT &, const char *) : llvm::cl::list<DataType>("list") {}
  };
};
} // namespace detail

class Pass {
public:
  template <typename DataType>
  using Option = detail::PassOptions::Option<DataType>;
  template <typename DataType>
  using ListOption = detail::PassOptions::ListOption<DataType>;

  virtual ~Pass() = default;
  Operation *getOperation() { return nullptr; }
  void signalPassFailure() {}
  virtual void runOnOperation() = 0;
};

template <typename OpTy = void>
class OperationPass : public Pass {};

template <typename TargetT>
TargetT dyn_cast(Operation *) {
  return TargetT{};
}

class FooOp : public Op<FooOp, OpTrait::OneResult> {
public:
  using Op::Op;
  unsigned getNumOperands() { return getOperation()->getNumOperands(); }
  Value getInput() { return getOperation()->getOperand(0); }
  LogicalResult verify();
  LogicalResult verifyRegions();
  static FooOp create(OpBuilder &, Location) { return FooOp{}; }
};

class MyOp : public Op<MyOp, OpTrait::OneResult> {
public:
  using Op::Op;
  Value getInput() { return getOperation()->getOperand(0); }
  LogicalResult verify();
  LogicalResult verifyRegions();
};

class UnrealizedConversionCastOp
    : public Op<UnrealizedConversionCastOp, OpTrait::OneResult> {
public:
  using Op::Op;
  OpResult getResult(unsigned Idx = 0) { return getOperation()->getResult(Idx); }
};

namespace arith {
class ConstantIndexOp : public Op<ConstantIndexOp, OpTrait::OneResult> {
public:
  using Op::Op;
  static ConstantIndexOp create(OpBuilder &, Location, int) {
    return ConstantIndexOp{};
  }
};
} // namespace arith

} // namespace mlir

#endif // MOCK_LLVM_MLIR_ALL_H
"""


@dataclass(frozen=True)
class TidyFinding:
    file: str
    line: int
    col: int
    severity: str
    message: str
    check: str


# ===========================================================================
# 1. Non-AST Python Linter Tests (always run, zero external binary needed)
# ===========================================================================


class TestNonASTMLIRCodingStandards(unittest.TestCase):
    """Positive, negative, and suppression tests for `check_mlir_coding_standards.py`."""

    POSITIVE_CASES = {
        "mlir-lit-stderr-devnull": (
            "test/MLIR/foo.mlir",
            "// RUN: %eopt %s 2>/dev/null | FileCheck %s\n",
        ),
        "mlir-lit-negative-func-name": (
            "test/MLIR/invalid.mlir",
            "// RUN: %eopt -verify-diagnostics -split-input-file %s\n"
            "func.func @bad_operand_fail() {\n"
            "  return\n"
            "}\n",
        ),
        "mlir-lit-pinned-ssa": (
            "test/MLIR/foo.mlir",
            "// RUN: %eopt %s | FileCheck %s\n"
            "// CHECK: %0 = arith.addf %arg0, %arg1 : f64\n",
        ),
        "mlir-lit-closing-brace": (
            "test/MLIR/foo.mlir",
            "// RUN: %eopt %s | FileCheck %s\n"
            "// CHECK-LABEL: func.func @foo\n"
            "// CHECK-NEXT: return\n"
            "// CHECK-NEXT: }\n",
        ),
        "mlir-doxygen-comment": (
            "Enzyme/MLIR/Foo.h",
            "// Convert the given value.\nValue convertValue(Value input);\n",
        ),
    }

    def test_every_registered_mlir_check_has_positive_case(self) -> None:
        registered = {spec.name for spec in mlir_lint.CHECK_SPECS}
        self.assertEqual(registered, set(self.POSITIVE_CASES.keys()))

    def test_positive_cases_trigger_expected_check(self) -> None:
        for expected_check, (path, code) in self.POSITIVE_CASES.items():
            with self.subTest(check=expected_check):
                findings = mlir_lint.check_text(path, code)
                self.assertIn(expected_check, {f.check for f in findings})

    def test_idiomatic_mlir_lit_and_doxygen_produce_zero_findings(self) -> None:
        idiomatic_mlir = (
            "// RUN: %eopt -verify-diagnostics -split-input-file %s | FileCheck %s\n"
            "\n"
            "// CHECK-LABEL: func.func @forward_square(\n"
            "// CHECK-SAME:    %[[X:.*]]: f64) -> f64 {\n"
            "// CHECK-NEXT:    %[[PROD:.*]] = arith.mulf %[[X]], %[[X]] : f64\n"
            "// CHECK-NEXT:    return %[[PROD]] : f64\n"
            "func.func @forward_square(%x: f64) -> f64 {\n"
            "  %0 = arith.mulf %x, %x : f64\n"
            "  return %0 : f64\n"
            "}\n"
            "\n"
            "// -----\n"
            "\n"
            "func.func @negative_mismatched_shadow_type(%x: f64) {\n"
            "  return\n"
            "}\n"
        )
        self.assertEqual(
            mlir_lint.check_text("test/MLIR/forward_square.mlir", idiomatic_mlir), []
        )

        idiomatic_cpp = (
            "/// Convert the given value.\n"
            "Value convertValue(Value input) {\n"
            "  // Inline body comment above helper call is fine.\n"
            "  return helperCall(input);\n"
            "}\n"
            "// Suppressed doc comment.\n"
            "Value suppressedFn(Value input); // NOLINT(mlir-doxygen-comment)\n"
        )
        # Note: NOLINTNEXTLINE or same-line NOLINT on comment line:
        idiomatic_cpp_suppressed = (
            "/// Convert the given value.\n"
            "Value convertValue(Value input) {\n"
            "  // Inline body comment above helper call is fine.\n"
            "  return helperCall(input);\n"
            "}\n"
            "// Plain comment with NOLINT(mlir-doxygen-comment)\n"
            "Value suppressedFn(Value input);\n"
        )
        self.assertEqual(
            mlir_lint.check_text("Enzyme/MLIR/Clean.cpp", idiomatic_cpp_suppressed), []
        )


class TestNonASTLLVMCodingStandards(unittest.TestCase):
    """Positive, negative, and suppression tests for `check_llvm_coding_standards.py`."""

    POSITIVE_CASES = {
        "llvm-lit-stderr-devnull": (
            "test/Enzyme/ReverseMode/foo.ll",
            '; RUN: %opt < %s %OPloadEnzyme -passes="enzyme" -S 2>/dev/null | FileCheck %s\n',
        ),
        "llvm-lit-pinned-ssa": (
            "test/Enzyme/ReverseMode/foo.ll",
            '; RUN: %opt < %s %OPloadEnzyme -passes="enzyme" -S | FileCheck %s\n'
            "; CHECK: %0 = fmul double %x, %x\n",
        ),
        "llvm-assert-false": (
            "Enzyme/Foo.cpp",
            'void unreachableCase() { assert(false && "should not happen"); }\n',
        ),
        "llvm-assert-no-message": (
            "Enzyme/Foo.cpp",
            "void checkPtr(int *Ptr) { assert(Ptr != nullptr); }\n",
        ),
        "llvm-include-comment": (
            "Enzyme/Foo.cpp",
            '#include "llvm/ADT/SmallVector.h" // Needed for SmallVector\n',
        ),
        "llvm-non-ascii": (
            "Enzyme/Foo.cpp",
            "// Em-dash \u2014 and unicode arrow \u2192 are non-ASCII.\n",
        ),
    }

    def test_every_registered_llvm_check_has_positive_case(self) -> None:
        registered = {spec.name for spec in llvm_lint.CHECK_SPECS}
        self.assertEqual(registered, set(self.POSITIVE_CASES.keys()))

    def test_positive_cases_trigger_expected_check(self) -> None:
        for expected_check, (path, code) in self.POSITIVE_CASES.items():
            with self.subTest(check=expected_check):
                findings = llvm_lint.check_text(path, code)
                self.assertIn(expected_check, {f.check for f in findings})

    def test_idiomatic_llvm_cpp_and_ll_lit_produce_zero_findings(self) -> None:
        clean_cpp = (
            '#include "llvm/ADT/ArrayRef.h" // IWYU pragma: keep\n'
            "/// Never write assert(false) or assert(ptr); in real code.\n"
            "void ok(int *Ptr) {\n"
            '  assert(Ptr && "Ptr must be non-null");\n'
            "  assert(Ptr != nullptr); // NOLINT(llvm-assert-no-message)\n"
            "}\n"
        )
        self.assertEqual(llvm_lint.check_text("Enzyme/Clean.cpp", clean_cpp), [])

        idiomatic_ll = (
            '; RUN: %opt < %s %OPloadEnzyme -passes="enzyme" -S | FileCheck %s\n'
            "\n"
            "; CHECK-LABEL: define internal double @fwddiffe_square(\n"
            "; CHECK-NEXT: entry:\n"
            "; CHECK-NEXT:   %[[MUL:.+]] = fmul fast double %{{.+}}, 2.000000e+00\n"
            "; CHECK-NEXT:   ret double %[[MUL]]\n"
        )
        self.assertEqual(
            llvm_lint.check_text("test/Enzyme/ForwardMode/square.ll", idiomatic_ll), []
        )

    def test_diff_base_and_live_enzyme_repository_cleanliness(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _git_init(tmpdir)
            ll_path = os.path.join(tmpdir, "sample.ll")
            with open(ll_path, "w", encoding="utf-8") as fh:
                fh.write(
                    '; RUN: %opt < %s -S | FileCheck %s\n'
                    "; CHECK: %0 = fadd double %x, %x\n"
                )
            _git_commit_all(tmpdir, "baseline")

            rc, _, _ = _run_main_captured(
                llvm_lint.main, ["--diff-base", "HEAD", ll_path]
            )
            self.assertEqual(rc, 0)

            with open(ll_path, "w", encoding="utf-8") as fh:
                fh.write(
                    '; RUN: %opt < %s -S | FileCheck %s\n'
                    "; CHECK: %0 = fadd double %x, %x\n"
                    "; CHECK: %1 = fmul double %x, %x\n"
                )
            rc_added, out_added, _ = _run_main_captured(
                llvm_lint.main, ["--diff-base", "HEAD", ll_path]
            )
            self.assertEqual(rc_added, 1)
            self.assertIn(":3: error: [llvm-lit-pinned-ssa]", out_added)
            self.assertNotIn(":2: error:", out_added)

        rc_mlir, _, err_mlir = _run_main_captured(
            mlir_lint.main,
            [
                os.path.join(ENZYME_DIR, "Enzyme", "MLIR"),
                os.path.join(ENZYME_DIR, "test", "MLIR"),
            ],
        )
        self.assertEqual(rc_mlir, 0, err_mlir)

        rc_llvm, _, err_llvm = _run_main_captured(
            llvm_lint.main,
            [
                os.path.join(ENZYME_DIR, "Enzyme"),
                os.path.join(ENZYME_DIR, "BCLoad"),
                os.path.join(ENZYME_DIR, "tools"),
                os.path.join(ENZYME_DIR, "test"),
            ],
        )
        self.assertEqual(rc_llvm, 0, err_llvm)


class TestRunClangTidyPurePythonLogic(unittest.TestCase):
    """Pure-Python unit tests for `run_clang_tidy.py` (no clang-tidy binary required)."""

    def test_whole_tree_checks_constant_completeness(self) -> None:
        self.assertEqual(
            set(tidy_runner.WHOLE_TREE_CHECKS),
            {
                "llvm-twine-local",
                "custom-llvm-twine-field-or-return",
                "portability-restrict-system-includes",
                "performance-avoid-endl",
                "custom-llvm-no-rtti-exceptions",
                "custom-llvm-no-iostream",
                "custom-llvm-std-sort",
                "custom-mlir-const-ir-pointer",
                "custom-mlir-builder-by-value",
                "custom-mlir-verifier-non-local",
                "custom-mlir-bool-error-return",
            },
        )

    def test_parse_and_filter_diagnostics_deduplicates_and_scopes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            enz_dir = os.path.join(tmpdir, "enzyme", "Enzyme")
            scev_dir = os.path.join(enz_dir, "SCEV")
            os.makedirs(scev_dir)
            foo_cpp = os.path.realpath(os.path.join(enz_dir, "Foo.cpp"))
            scev_cpp = os.path.realpath(os.path.join(scev_dir, "Ignored.cpp"))
            for p in (foo_cpp, scev_cpp):
                with open(p, "w", encoding="utf-8") as fh:
                    fh.write("int x;\n")

            sample_out = (
                f"{foo_cpp}:10:3: warning: bad twine [llvm-twine-local]\n"
                "   10 |   Twine t;\n"
                "      |   ^\n"
                f"{foo_cpp}:10:3: warning: bad twine [llvm-twine-local]\n"
                f"{foo_cpp}:20:5: warning: else after return [llvm-else-after-return]\n"
                f"{foo_cpp}:30:5: warning: added else [llvm-else-after-return]\n"
                f"{scev_cpp}:5:3: warning: bad twine in SCEV [llvm-twine-local]\n"
            )
            parsed = tidy_runner.parse_clang_tidy_output(
                sample_out, tu_dir=tmpdir, cwd=tmpdir
            )
            filtered = tidy_runner.filter_diagnostics(
                parsed,
                target_roots=[os.path.realpath(enz_dir)],
                build_dir=None,
                git_top=os.path.realpath(tmpdir),
                all_checks=False,
                changed_lines={foo_cpp: {30}},
            )
            self.assertEqual(
                [(d.line, d.check) for d in filtered],
                [(10, "llvm-twine-local"), (30, "llvm-else-after-return")],
            )


# ===========================================================================
# 2. Live `clang-tidy` Tests for All 25 C++ AST Rules & `run_clang_tidy.py`
# ===========================================================================


@unittest.skipIf(
    CLANG_TIDY_BIN is None,
    "clang-tidy >= 21 with --experimental-custom-checks is not installed",
)
class TestLiveClangTidyConfigsAnd25Checks(unittest.TestCase):
    """End-to-end tests of all 25 C++ AST checks using the actual repo `.clang-tidy` hierarchy."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir_obj = tempfile.TemporaryDirectory(prefix="enzyme_tidy_suite_")
        cls.tmpdir = cls._tmpdir_obj.name

        # Mirror the exact Enzyme `.clang-tidy` directory hierarchy in `cls.tmpdir`.
        cls.enz_root = os.path.join(cls.tmpdir, "enzyme")
        cls.enz_cpp_dir = os.path.join(cls.enz_root, "Enzyme")
        cls.mlir_dir = os.path.join(cls.enz_cpp_dir, "MLIR")
        cls.passes_dir = os.path.join(cls.mlir_dir, "Passes")
        cls.mlir_tools_dir = os.path.join(cls.mlir_dir, "Tools")
        for d in (cls.passes_dir, cls.mlir_tools_dir):
            os.makedirs(d, exist_ok=True)

        shutil.copyfile(
            os.path.join(ENZYME_DIR, ".clang-tidy"),
            os.path.join(cls.enz_root, ".clang-tidy"),
        )
        shutil.copyfile(
            os.path.join(ENZYME_DIR, "Enzyme", "MLIR", ".clang-tidy"),
            os.path.join(cls.mlir_dir, ".clang-tidy"),
        )
        shutil.copyfile(
            os.path.join(ENZYME_DIR, "Enzyme", "MLIR", "Passes", ".clang-tidy"),
            os.path.join(cls.passes_dir, ".clang-tidy"),
        )

        # Populate simulated external `include/llvm/` and `include/mlir/` headers
        # included via `-I` (not `-isystem`) to exercise `ExcludeHeaderFilterRegex`.
        cls.ext_inc = os.path.join(cls.tmpdir, "llvm-project", "include")
        mock_headers = {
            "llvm/MockAll.h": MOCK_LLVM_MLIR_H,
            "llvm/ADT/ArrayRef.h": '#ifndef H1\n#define H1\n#include "llvm/MockAll.h"\n#endif\n',
            "llvm/ADT/SmallVector.h": '#ifndef H2\n#define H2\n#include "llvm/MockAll.h"\n#endif\n',
            "llvm/ADT/STLExtras.h": '#ifndef H3\n#define H3\n#include "llvm/MockAll.h"\n#endif\n',
            "llvm/ADT/StringRef.h": '#ifndef H4\n#define H4\n#include "llvm/MockAll.h"\n#endif\n',
            "llvm/ADT/Twine.h": '#ifndef H5\n#define H5\n#include "llvm/MockAll.h"\n#endif\n',
            "llvm/IR/Value.h": '#ifndef H6\n#define H6\n#include "llvm/MockAll.h"\n#endif\n',
            "llvm/Support/raw_ostream.h": '#ifndef H7\n#define H7\n#include "llvm/MockAll.h"\n#endif\n',
            "mlir/MockMLIR.h": '#ifndef H8\n#define H8\n#include "llvm/MockAll.h"\n#endif\n',
        }
        for rel, content in mock_headers.items():
            full = os.path.join(cls.ext_inc, rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w", encoding="utf-8") as fh:
                fh.write(content)

        cls._counter = 0

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir_obj.cleanup()

    def _run_tidy(
        self,
        rel_to_enzyme: str,
        code: str,
        *,
        compile_flags: Sequence[str] = (),
    ) -> List[TidyFinding]:
        type(self)._counter += 1
        target_path = os.path.join(self.enz_root, rel_to_enzyme)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as fh:
            fh.write(code)

        flags: List[str] = ["-std=c++17", "-Wno-unused-function", f"-I{self.ext_inc}"]
        if target_path.endswith((".h", ".hpp")):
            flags.insert(0, "-xc++-header")
        flags.extend(compile_flags)

        cmd = [
            CLANG_TIDY_BIN,
            "--experimental-custom-checks",
            "--quiet",
            target_path,
            "--",
            *flags,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        self.assertNotIn(
            "[clang-tidy-config]",
            combined,
            f"Configuration error in .clang-tidy:\n{combined}",
        )

        findings: List[TidyFinding] = []
        for line in combined.splitlines():
            m = DIAG_RE.match(line.strip())
            if not m:
                continue
            for chk in m.group("checks").split(","):
                chk = chk.strip()
                findings.append(
                    TidyFinding(
                        file=m.group("file"),
                        line=int(m.group("line")),
                        col=int(m.group("col")),
                        severity=m.group("severity"),
                        message=m.group("message"),
                        check=chk,
                    )
                )

        compiler_errors = [
            f for f in findings if f.check.startswith("clang-diagnostic-")
        ]
        self.assertEqual(
            compiler_errors,
            [],
            f"Unexpected compiler error(s) in {rel_to_enzyme}:\n{combined}\nCode:\n{code}",
        )
        return findings

    @staticmethod
    def _check_set(findings: Iterable[TidyFinding]) -> Set[str]:
        return {f.check for f in findings}

    def test_1_all_11_llvm_ast_rules_positive_and_negative(self) -> None:
        cases = [
            # 1. Twine local + field/return
            (
                "Enzyme/TwineCase.cpp",
                '#include "llvm/ADT/Twine.h"\n'
                "struct Holder { llvm::Twine Field; };\n"
                "void f(llvm::StringRef Base) {\n"
                '  llvm::Twine Full = Base + "_x";\n'
                "  (void)Full;\n"
                "}\n",
                (),
                {"llvm-twine-local", "custom-llvm-twine-field-or-return"},
            ),
            # 2. else after return
            (
                "Enzyme/ElseCase.cpp",
                "int classify(int X) {\n"
                "  if (X < 0) {\n"
                "    return -1;\n"
                "  } else {\n"
                "    return 1;\n"
                "  }\n"
                "}\n",
                (),
                {"llvm-else-after-return"},
            ),
            # 3. anonymous namespace function
            (
                "Enzyme/AnonCase.cpp",
                "namespace {\nint computeLocal(int X) { return X + 1; }\n}\n",
                (),
                {"llvm-prefer-static-over-anonymous-namespace"},
            ),
            # 4. #pragma once
            (
                "Enzyme/PragmaCase.h",
                "#pragma once\nint helper();\n",
                (),
                {"portability-avoid-pragma-once"},
            ),
            # 5. <iostream> / std::cerr / std::endl
            (
                "Enzyme/IostreamCase.cpp",
                "#include <iostream>\nvoid dump() { std::cerr << 42 << std::endl; }\n",
                (),
                {
                    "portability-restrict-system-includes",
                    "performance-avoid-endl",
                    "custom-llvm-no-iostream",
                },
            ),
            # 6. RTTI / exceptions
            (
                "Enzyme/RttiCase.cpp",
                "struct Base { virtual ~Base() = default; };\n"
                "struct Derived : Base {};\n"
                "void bad(Base *B) {\n"
                "  if (auto *D = dynamic_cast<Derived *>(B)) { (void)D; throw 1; }\n"
                "}\n",
                ("-frtti", "-fcxx-exceptions"),
                {"custom-llvm-no-rtti-exceptions"},
            ),
            # 7. std::sort
            (
                "Enzyme/SortCase.cpp",
                '#include "llvm/ADT/SmallVector.h"\n'
                "void sortVec(llvm::SmallVectorImpl<int> &Vals) {\n"
                "  std::sort(Vals.begin(), Vals.end());\n"
                "}\n",
                (),
                {"custom-llvm-std-sort"},
            ),
            # 8. std::vector<bool>
            (
                "Enzyme/VecBoolCase.cpp",
                "#include <vector>\nvoid flags() { std::vector<bool> ActiveFlags; (void)ActiveFlags; }\n",
                (),
                {"custom-llvm-vector-bool"},
            ),
            # 9. SmallVector<T, N> & parameter
            (
                "Enzyme/SmallVecParamCase.h",
                "#ifndef SV_CASE_H\n#define SV_CASE_H\n"
                '#include "llvm/ADT/SmallVector.h"\n'
                "void collect(llvm::SmallVector<int, 8> &Out);\n"
                "#endif\n",
                (),
                {"custom-llvm-smallvector-param"},
            ),
            # 10. const std::string & parameter
            (
                "Enzyme/StdStringParamCase.h",
                "#ifndef STR_CASE_H\n#define STR_CASE_H\n"
                "#include <string>\n"
                "bool hasPrefix(const std::string &Name);\n"
                "#endif\n",
                (),
                {"custom-llvm-const-std-string-param"},
            ),
            # 11. raw_ostream << "\n"
            (
                "Enzyme/OstreamNewlineCase.cpp",
                '#include "llvm/Support/raw_ostream.h"\n'
                'void logMsg() { llvm::errs() << "warning" << "\\n"; }\n',
                (),
                {"custom-llvm-ostream-string-newline"},
            ),
        ]

        for rel_path, code, extra_flags, expected_checks in cases:
            with self.subTest(path=rel_path):
                findings = self._run_tidy(
                    rel_path, code, compile_flags=extra_flags
                )
                self.assertEqual(self._check_set(findings), expected_checks)

        # Idiomatic LLVM C++ (including `llvm::sort` instantiating `std::sort` via `-I`
        # header, `#ifndef ENZYME_FOO_H`, `if constexpr`, and `struct` in anon namespace)
        idiomatic_llvm = (
            "#ifndef ENZYME_IDIOMATIC_H\n"
            "#define ENZYME_IDIOMATIC_H\n"
            '#include "llvm/ADT/ArrayRef.h"\n'
            '#include "llvm/ADT/SmallVector.h"\n'
            '#include "llvm/ADT/STLExtras.h"\n'
            '#include "llvm/ADT/StringRef.h"\n'
            '#include "llvm/ADT/Twine.h"\n'
            '#include "llvm/Support/raw_ostream.h"\n'
            "\n"
            "namespace {\n"
            "struct LocalVisitor {\n"
            "  int visit(int X) { return X + 1; }\n"
            "};\n"
            "} // namespace\n"
            "\n"
            "static int helper(llvm::StringRef Name,\n"
            "                  llvm::SmallVectorImpl<int> &Out,\n"
            "                  const llvm::Twine &Label) {\n"
            "  (void)Name;\n"
            "  llvm::sort(Out);\n"
            "  if constexpr (sizeof(void *) == 8) {\n"
            "    return LocalVisitor{}.visit(0);\n"
            "  } else {\n"
            "    return -1;\n"
            "  }\n"
            "  llvm::errs() << Label << '\\n';\n"
            "  return static_cast<int>(Out.size());\n"
            "}\n"
            "#endif\n"
        )
        self.assertEqual(self._run_tidy("Enzyme/IdiomaticLLVM.h", idiomatic_llvm), [])

    def test_2_all_14_mlir_ast_rules_and_inheritance_positive_and_negative(self) -> None:
        hdr = '#include "mlir/MockMLIR.h"\nusing namespace mlir;\n'
        cases = [
            # 12. custom-mlir-const-ir-pointer
            (
                "Enzyme/MLIR/ConstPtr.cpp",
                hdr + "void inspect(const Operation *op, Block const *blk) { (void)op; (void)blk; }\n",
                {"custom-mlir-const-ir-pointer"},
            ),
            # 13. custom-mlir-const-ir-value
            (
                "Enzyme/MLIR/ConstVal.cpp",
                hdr + "void inspect() { const Value val = nullptr; (void)val; }\n",
                {"custom-mlir-const-ir-value"},
            ),
            # 14. custom-mlir-pass-ir-by-value (including uninstantiated TypedValue)
            (
                "Enzyme/MLIR/PassByVal.h",
                "#ifndef PBV_H\n#define PBV_H\n"
                + hdr
                + "LogicalResult helper(const Value &val, const TypedValue<FloatType> &tv);\n"
                "#endif\n",
                {"custom-mlir-pass-ir-by-value"},
            ),
            # 15. custom-mlir-builder-by-value
            (
                "Enzyme/MLIR/BuilderByVal.h",
                "#ifndef BBV_H\n#define BBV_H\n"
                + hdr
                + "void buildOps(OpBuilder builder, Location loc);\n"
                "#endif\n",
                {"custom-mlir-builder-by-value"},
            ),
            # 16. custom-mlir-bool-error-return
            (
                "Enzyme/MLIR/BoolErr.h",
                "#ifndef BERR_H\n#define BERR_H\n"
                + hdr
                + "FailureOr<bool> trySimplify(Operation *op);\n"
                "#endif\n",
                {"custom-mlir-bool-error-return"},
            ),
            # 17. custom-mlir-auto-overuse
            (
                "Enzyme/MLIR/AutoOveruse.cpp",
                hdr + "void inspect(Operation *op) { auto operand = op->getOperand(0); (void)operand; }\n",
                {"custom-mlir-auto-overuse"},
            ),
            # 18. llvm-use-new-mlir-op-builder
            (
                "Enzyme/MLIR/OpCreate.cpp",
                hdr
                + "void emit(OpBuilder &builder, Location loc) {\n"
                "  builder.create<arith::ConstantIndexOp>(loc, 0);\n"
                "}\n",
                {"llvm-use-new-mlir-op-builder"},
            ),
            # 19. readability-redundant-inline-specifier
            (
                "Enzyme/MLIR/RedundantInline.cpp",
                "static inline bool isZero(int val) { return val == 0; }\n",
                {"readability-redundant-inline-specifier"},
            ),
            # 20. readability-identifier-naming (camelBack)
            (
                "Enzyme/MLIR/CamelBack.cpp",
                hdr + "void compute(Operation *op) { Value InputVal = op->getOperand(0); (void)InputVal; }\n",
                {"readability-identifier-naming"},
            ),
            # 21. custom-mlir-raw-builder-in-rewrite
            (
                "Enzyme/MLIR/RawBuilder.cpp",
                hdr
                + "struct Pat : OpRewritePattern<FooOp> {\n"
                "  LogicalResult matchAndRewrite(FooOp op, PatternRewriter &rewriter) const override {\n"
                "    (void)rewriter;\n"
                "    OpBuilder localBuilder(op);\n"
                "    (void)localBuilder;\n"
                "    return success();\n"
                "  }\n"
                "};\n",
                {"custom-mlir-raw-builder-in-rewrite"},
            ),
            # 22. custom-mlir-direct-ir-mutation (both op->erase() and OneResult op.replaceAllUsesWith)
            (
                "Enzyme/MLIR/DirectMut.cpp",
                hdr
                + "struct Pat : OpRewritePattern<FooOp> {\n"
                "  LogicalResult matchAndRewrite(FooOp op, PatternRewriter &rewriter) const override {\n"
                "    (void)rewriter;\n"
                "    op.replaceAllUsesWith(op.getInput());\n"
                "    op->erase();\n"
                "    return success();\n"
                "  }\n"
                "};\n",
                {"custom-mlir-direct-ir-mutation"},
            ),
            # 23. custom-mlir-verifier-non-local
            (
                "Enzyme/MLIR/VerifierNonLocal.cpp",
                hdr
                + "LogicalResult MyOp::verify() {\n"
                "  Operation *def = getInput().getDefiningOp();\n"
                '  return def ? success() : emitOpError("bad");\n'
                "}\n",
                {"custom-mlir-verifier-non-local"},
            ),
            # 24. custom-mlir-pass-cl-opt (enabled inside `Enzyme/MLIR/Passes/`)
            (
                "Enzyme/MLIR/Passes/BadPassOpt.cpp",
                hdr + 'static llvm::cl::opt<bool> enableFoo("enable-foo");\n',
                {"custom-mlir-pass-cl-opt"},
            ),
            # 25. custom-mlir-pass-no-abort
            (
                "Enzyme/MLIR/Passes/BadPassAbort.cpp",
                hdr
                + "struct BadPass : OperationPass<> {\n"
                "  void runOnOperation() override {\n"
                '    llvm::report_fatal_error("boom");\n'
                "  }\n"
                "};\n",
                {"custom-mlir-pass-no-abort"},
            ),
            # Inherited parent LLVM check inside `Enzyme/MLIR/`!
            (
                "Enzyme/MLIR/InheritedParentCheck.cpp",
                hdr
                + "void sortVals(SmallVectorImpl<int> &vals) {\n"
                "  std::sort(vals.begin(), vals.end());\n"
                "}\n",
                {"custom-llvm-std-sort"},
            ),
        ]

        for rel_path, code, expected_checks in cases:
            with self.subTest(path=rel_path):
                findings = self._run_tidy(rel_path, code)
                self.assertEqual(self._check_set(findings), expected_checks)

        # Verify `llvm::cl::opt` in `Enzyme/MLIR/Tools/` (outside `Passes/`) is NOT flagged!
        tools_opt = hdr + 'static llvm::cl::opt<bool> toolFlag("tool-flag");\n'
        self.assertEqual(
            self._run_tidy("Enzyme/MLIR/Tools/DriverFlag.cpp", tools_opt), []
        )

        # Verify idiomatic MLIR C++ inside `Enzyme/MLIR/Passes/` passes BOTH parent
        # LLVM checks AND all 14 child MLIR checks with ZERO warnings!
        idiomatic_mlir_cpp = (
            hdr
            + "namespace {\n"
            "struct LowerFooPattern : public OpRewritePattern<FooOp> {\n"
            "  using OpRewritePattern::OpRewritePattern;\n"
            "  LogicalResult matchAndRewrite(FooOp op,\n"
            "                                PatternRewriter &rewriter) const override {\n"
            "    OpBuilder::InsertionGuard guard(rewriter);\n"
            "    DenseMap<Block *, unsigned> blockCounts;\n"
            "    blockCounts.erase(op->getBlock());\n"
            "    Value firstOperand = op->getOperand(0);\n"
            "    auto constOp = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 1);\n"
            "    if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(firstOperand.getDefiningOp()))\n"
            "      rewriter.replaceAllUsesWith(castOp.getResult(0), constOp);\n"
            "    rewriter.replaceOp(op, constOp.getResult());\n"
            "    return success();\n"
            "  }\n"
            "};\n"
            "\n"
            "struct DemoPass : public OperationPass<> {\n"
            '  Pass::Option<bool> validOpt{*this, "valid-opt"};\n'
            "  void runOnOperation() override {\n"
            "    Operation *rootOp = getOperation();\n"
            "    if (!rootOp) {\n"
            "      signalPassFailure();\n"
            "      return;\n"
            "    }\n"
            "  }\n"
            "};\n"
            "} // namespace\n"
            "\n"
            "static void cApiBridge(const mlir::Value *valueArray, size_t count,\n"
            "                       OpBuilder &builder, Value scalarVal, Type resultTy) {\n"
            "  (void)valueArray;\n"
            "  (void)count;\n"
            "  (void)builder;\n"
            "  (void)scalarVal;\n"
            "  (void)resultTy;\n"
            "}\n"
            "\n"
            "LogicalResult FooOp::verify() {\n"
            "  if (getNumOperands() == 0)\n"
            '    return emitOpError("expected at least one operand");\n'
            "  return success();\n"
            "}\n"
            "\n"
            "static const char *kDoc = R\"mlir(\n"
            "  const Operation *bad = nullptr;\n"
            "  builder.create<FooOp>(loc);\n"
            ")mlir\";\n"
            "\n"
            "bool entryPoint(OpBuilder &builder) {\n"
            "  cApiBridge(nullptr, 0, builder, Value{}, Type{});\n"
            "  // NOLINTNEXTLINE(llvm-use-new-mlir-op-builder)\n"
            "  builder.create<FooOp>(Location{});\n"
            "  return kDoc != nullptr;\n"
            "}\n"
        )
        self.assertEqual(
            self._run_tidy("Enzyme/MLIR/Passes/IdiomaticPass.cpp", idiomatic_mlir_cpp),
            [],
        )


@unittest.skipIf(
    CLANG_TIDY_BIN is None,
    "clang-tidy >= 21 with --experimental-custom-checks is not installed",
)
class TestRunClangTidyIntegration(unittest.TestCase):
    """End-to-end tests for `run_clang_tidy.py` with `compile_commands.json` and `git diff`."""

    def test_two_tier_diff_base_header_promotion_dedup_and_cross_dir_scoping(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = os.path.join(tmpdir, "repo")
            enz_root = os.path.join(repo, "enzyme")
            enz_dir = os.path.join(enz_root, "Enzyme")
            mlir_dir = os.path.join(enz_dir, "MLIR")
            scev_dir = os.path.join(enz_dir, "SCEV")
            bld_dir = os.path.join(repo, "enzyme-build")
            ext_inc = os.path.join(repo, "llvm-project", "include")
            for d in (mlir_dir, scev_dir, bld_dir, os.path.join(ext_inc, "mlir")):
                os.makedirs(d, exist_ok=True)

            _git_init(repo)

            shutil.copyfile(
                os.path.join(ENZYME_DIR, ".clang-tidy"),
                os.path.join(enz_root, ".clang-tidy"),
            )
            shutil.copyfile(
                os.path.join(ENZYME_DIR, "Enzyme", "MLIR", ".clang-tidy"),
                os.path.join(mlir_dir, ".clang-tidy"),
            )
            with open(
                os.path.join(ext_inc, "mlir", "MockMLIR.h"), "w", encoding="utf-8"
            ) as fh:
                fh.write(MOCK_LLVM_MLIR_H)

            parent_utils_h = os.path.join(enz_dir, "Utils.h")
            core_cpp = os.path.join(enz_dir, "Core.cpp")
            mlir_cpp = os.path.join(mlir_dir, "Rewrite.cpp")
            scev_cpp = os.path.join(scev_dir, "ScevIgnored.cpp")

            with open(parent_utils_h, "w", encoding="utf-8") as fh:
                fh.write(
                    "#ifndef ENZYME_UTILS_H\n"
                    "#define ENZYME_UTILS_H\n"
                    "inline int legacyClassify(int X) {\n"
                    "  if (X < 0) {\n"
                    "    return -1;\n"
                    "  } else {\n"
                    "    return 1;\n"
                    "  }\n"
                    "}\n"
                    "#endif\n"
                )
            with open(core_cpp, "w", encoding="utf-8") as fh:
                fh.write('#include "Utils.h"\nint runCore() { return legacyClassify(1); }\n')
            with open(mlir_cpp, "w", encoding="utf-8") as fh:
                fh.write(
                    '#include "../Utils.h"\n'
                    '#include "mlir/MockMLIR.h"\n'
                    "int runMlir() { return legacyClassify(2); }\n"
                )
            with open(scev_cpp, "w", encoding="utf-8") as fh:
                fh.write(
                    '#include "mlir/MockMLIR.h"\n'
                    "void scevBad(llvm::SmallVectorImpl<int> &V) {\n"
                    "  std::sort(V.begin(), V.end());\n"
                    "}\n"
                )

            compdb = []
            # Duplicate TUs in `compile_commands.json` (like LLVMEnzyme + ClangEnzyme).
            for tu in (core_cpp, mlir_cpp, scev_cpp):
                for mode in ("A", "B"):
                    compdb.append(
                        {
                            "directory": bld_dir,
                            "command": f"/usr/bin/clang++ -DMODE_{mode} -std=c++17 -I{ext_inc} -I{enz_dir} -c {tu}",
                            "file": tu,
                        }
                    )
            with open(
                os.path.join(bld_dir, "compile_commands.json"), "w", encoding="utf-8"
            ) as fh:
                json.dump(compdb, fh)

            _git_commit_all(repo, "baseline")

            # 1. Whole-tree mode passes (legacy `else` after `return` ignored; `SCEV/` ignored).
            rc, out, err = _run_main_captured(
                tidy_runner.main,
                ["--clang-tidy-binary", CLANG_TIDY_BIN, "-p", bld_dir, enz_dir],
            )
            self.assertEqual(rc, 0, f"stdout={out}\nstderr={err}")

            # 2. Modify `Enzyme/Utils.h` (parent non-MLIR header, `#include`d by BOTH
            #    `Enzyme/Core.cpp` and `Enzyme/MLIR/Rewrite.cpp`!) to add an idiomatic
            #    LLVM `UpperCamelCase` variable `UpperCamelVal`.
            #    `MLIR/Rewrite.cpp` MUST NOT bleed `readability-identifier-naming` (`camelBack`)
            #    onto `Enzyme/Utils.h`!
            with open(parent_utils_h, "w", encoding="utf-8") as fh:
                fh.write(
                    "#ifndef ENZYME_UTILS_H\n"
                    "#define ENZYME_UTILS_H\n"
                    "inline int legacyClassify(int X) {\n"
                    "  if (X < 0) {\n"
                    "    return -1;\n"
                    "  } else {\n"
                    "    return 1;\n"
                    "  }\n"
                    "}\n"
                    "inline int cleanLLVMHelper(int InputVal) {\n"
                    "  int UpperCamelVal = InputVal + 1;\n"
                    "  return UpperCamelVal;\n"
                    "}\n"
                    "#endif\n"
                )
            rc, out, err = _run_main_captured(
                tidy_runner.main,
                [
                    "--clang-tidy-binary",
                    CLANG_TIDY_BIN,
                    "-p",
                    bld_dir,
                    "--diff-base",
                    "HEAD",
                    enz_dir,
                ],
            )
            self.assertEqual(rc, 0, f"Unexpected cross-dir bleed:\nstdout={out}\nstderr={err}")

            # 3. Now add a real LLVM diff-scoped violation (`std::vector<bool>`) to
            #    `Enzyme/Utils.h` (without touching any `.cpp`!). Verify header-only
            #    diff promotion catches it AND deduplicates across `Core.cpp` & `Rewrite.cpp`!
            with open(parent_utils_h, "w", encoding="utf-8") as fh:
                fh.write(
                    "#ifndef ENZYME_UTILS_H\n"
                    "#define ENZYME_UTILS_H\n"
                    "#include <vector>\n"
                    "inline int legacyClassify(int X) {\n"
                    "  if (X < 0) {\n"
                    "    return -1;\n"
                    "  } else {\n"
                    "    return 1;\n"
                    "  }\n"
                    "}\n"
                    "inline void badAdded() { std::vector<bool> BadFlags; (void)BadFlags; }\n"
                    "#endif\n"
                )
            rc, out, err = _run_main_captured(
                tidy_runner.main,
                [
                    "--clang-tidy-binary",
                    CLANG_TIDY_BIN,
                    "-p",
                    bld_dir,
                    "--diff-base",
                    "HEAD",
                    "--github",
                    enz_dir,
                ],
            )
            self.assertEqual(rc, 1)
            gh_errors = [
                line for line in out.splitlines() if line.startswith("::error ")
            ]
            self.assertEqual(
                len(gh_errors),
                1,
                f"Expected exactly 1 deduplicated violation on Utils.h, got:\n{out}",
            )
            self.assertIn("custom-llvm-vector-bool", gh_errors[0])


if __name__ == "__main__":
    unittest.main()
