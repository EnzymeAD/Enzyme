#!/usr/bin/env python3
"""Unit tests for MLIR and LLVM coding standards linters."""

import os
import subprocess
import sys
import tempfile
import unittest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import check_llvm_coding_standards as llvm_lint
import check_mlir_coding_standards as mlir_lint


class TestMLIRCodingStandards(unittest.TestCase):
    """Positive, negative, and edge-case tests for every MLIR coding standard check."""

    POSITIVE_CASES = {
        "mlir-const-ir-pointer": (
            "Enzyme/MLIR/Foo.cpp",
            "void inspect(const Operation *op, Block const *block) {}\n",
        ),
        "mlir-const-ir-value": (
            "Enzyme/MLIR/Foo.cpp",
            "void inspect(OpBuilder &b) {\n  const Value v = nullptr;\n}\n",
        ),
        "mlir-pass-ir-by-value": (
            "Enzyme/MLIR/Foo.h",
            "/// Helper.\nLogicalResult helper(const Value &val, const Type &ty);\n",
        ),
        "mlir-builder-by-value": (
            "Enzyme/MLIR/Foo.h",
            "/// Build ops.\nvoid buildOps(OpBuilder builder, Location loc);\n",
        ),
        "mlir-raw-builder-in-rewrite": (
            "Enzyme/MLIR/Foo.cpp",
            "struct Pat {\n"
            "  LogicalResult matchAndRewrite(FooOp op, PatternRewriter &rewriter) const override {\n"
            "    OpBuilder localBuilder(op);\n"
            "    return success();\n"
            "  }\n"
            "};\n",
        ),
        "mlir-direct-ir-mutation": (
            "Enzyme/MLIR/Foo.cpp",
            "struct Pat {\n"
            "  LogicalResult matchAndRewrite(FooOp op, PatternRewriter &rewriter) const override {\n"
            "    op->erase();\n"
            "    return success();\n"
            "  }\n"
            "};\n",
        ),
        "mlir-op-create": (
            "Enzyme/MLIR/Foo.cpp",
            "void emit(OpBuilder &builder, Location loc) {\n"
            "  builder.create<arith::ConstantIndexOp>(loc, 0);\n"
            "}\n",
        ),
        "mlir-verifier-non-local": (
            "Enzyme/MLIR/Dialect/Ops.cpp",
            "LogicalResult MyOp::verify() {\n"
            "  Operation *def = getInput().getDefiningOp();\n"
            "  return def ? success() : emitOpError(\"bad\");\n"
            "}\n",
        ),
        "mlir-pass-cl-opt": (
            "Enzyme/MLIR/Passes/MyPass.cpp",
            "static llvm::cl::opt<bool> enableFoo(\"enable-foo\");\n",
        ),
        "mlir-pass-no-abort": (
            "Enzyme/MLIR/Passes/MyPass.cpp",
            "struct MyPass {\n"
            "  void runOnOperation() override {\n"
            "    llvm::report_fatal_error(\"boom\");\n"
            "  }\n"
            "};\n",
        ),
        "mlir-bool-error-return": (
            "Enzyme/MLIR/Foo.h",
            "/// Fallible helper.\nFailureOr<bool> trySimplify(Operation *op);\n",
        ),
        "mlir-explicit-inline": (
            "Enzyme/MLIR/Foo.cpp",
            "/// Local helper.\nstatic inline bool isZero(int x) { return x == 0; }\n",
        ),
        "mlir-camelback-naming": (
            "Enzyme/MLIR/Foo.cpp",
            "/// Compute.\nvoid compute(Operation *op) {\n"
            "  Value InputVal = op->getOperand(0);\n"
            "}\n",
        ),
        "mlir-doxygen-comment": (
            "Enzyme/MLIR/Foo.h",
            "// Convert the given value.\nValue convertValue(Value input);\n",
        ),
        "mlir-auto-overuse": (
            "Enzyme/MLIR/Foo.cpp",
            "/// Inspect.\nvoid inspect(Operation *op) {\n"
            "  auto operand = op->getOperand(0);\n"
            "}\n",
        ),
        "mlir-lit-pinned-ssa": (
            "test/MLIR/foo.mlir",
            "// RUN: %eopt %s | FileCheck %s\n"
            "// CHECK: %0 = arith.addf %arg0, %arg1 : f64\n",
        ),
        "mlir-lit-stderr-devnull": (
            "test/MLIR/foo.mlir",
            "// RUN: %eopt %s 2>/dev/null | FileCheck %s\n",
        ),
        "mlir-lit-closing-brace": (
            "test/MLIR/foo.mlir",
            "// RUN: %eopt %s | FileCheck %s\n"
            "// CHECK-LABEL: func.func @foo\n"
            "// CHECK-NEXT: return\n"
            "// CHECK-NEXT: }\n",
        ),
        "mlir-lit-negative-func-name": (
            "test/MLIR/invalid.mlir",
            "// RUN: %eopt -verify-diagnostics -split-input-file %s\n"
            "func.func @bad_operand_fail() {\n"
            "  return\n"
            "}\n",
        ),
    }

    def test_every_registered_mlir_check_has_positive_case(self):
        registered = {spec.name for spec in mlir_lint.CHECK_SPECS}
        self.assertEqual(
            registered,
            set(self.POSITIVE_CASES.keys()),
            "Every MLIR check in CHECK_SPECS must have an explicit test case.",
        )

    def test_positive_cases_trigger_expected_check(self):
        for expected_check, (path, code) in self.POSITIVE_CASES.items():
            with self.subTest(check=expected_check):
                findings = mlir_lint.check_text(path, code)
                checks = {f.check for f in findings}
                self.assertIn(
                    expected_check,
                    checks,
                    f"Expected `{expected_check}` in findings for:\n{code}\nGot: {findings}",
                )

    def test_idiomatic_mlir_cpp_produces_zero_findings(self):
        idiomatic_cpp = (
            "/// Rewrite `FooOp` into `arith::ConstantIndexOp`.\n"
            "struct LowerFooPattern : public OpRewritePattern<FooOp> {\n"
            "  using OpRewritePattern::OpRewritePattern;\n"
            "\n"
            "  LogicalResult matchAndRewrite(FooOp op,\n"
            "                                PatternRewriter &rewriter) const override {\n"
            "    // Track visited blocks in a local map; container `.erase(key)` is fine.\n"
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
            "/// C-API array of `const mlir::Value *` is a pointer-to-const array, not const Value.\n"
            "static void cApiBridge(const mlir::Value *valueArray, size_t count,\n"
            "                       OpBuilder &builder, Value scalarVal, Type resultTy) {\n"
            "  (void)valueArray;\n"
            "  (void)count;\n"
            "  (void)builder;\n"
            "  (void)scalarVal;\n"
            "  (void)resultTy;\n"
            "}\n"
            "\n"
            "/// Local-only op verifier.\n"
            "LogicalResult FooOp::verify() {\n"
            "  if (getNumOperands() == 0)\n"
            "    return emitOpError(\"expected at least one operand\");\n"
            "  return success();\n"
            "}\n"
            "\n"
            "/// Pass signaling failure properly without aborting.\n"
            "struct DemoPass {\n"
            "  void runOnOperation() override {\n"
            "    Operation *rootOp = getOperation();\n"
            "    if (!rootOp) {\n"
            "      signalPassFailure();\n"
            "      return;\n"
            "    }\n"
            "  }\n"
            "};\n"
        )
        findings = mlir_lint.check_text("Enzyme/MLIR/Passes/DemoPass.cpp", idiomatic_cpp)
        self.assertEqual(findings, [])

    def test_idiomatic_mlir_lit_test_produces_zero_findings(self):
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
        findings = mlir_lint.check_text("test/MLIR/forward_square.mlir", idiomatic_mlir)
        self.assertEqual(findings, [])

    def test_direct_rauw_in_rewrite_is_flagged_but_rewriter_rauw_is_not(self):
        bad = (
            "struct Pat {\n"
            "  LogicalResult matchAndRewrite(FooOp op, PatternRewriter &rewriter) const override {\n"
            "    op.replaceAllUsesWith(newVal);\n"
            "    return success();\n"
            "  }\n"
            "};\n"
        )
        findings = mlir_lint.check_text("Enzyme/MLIR/Foo.cpp", bad)
        self.assertEqual([f.check for f in findings], ["mlir-direct-ir-mutation"])

    def test_comments_and_raw_strings_do_not_false_positive(self):
        code = (
            "/// Example showing why `const Operation *` and `builder.create<FooOp>` are avoided.\n"
            "static const char *kDoc = R\"mlir(\n"
            "  const Operation *bad = nullptr;\n"
            "  builder.create<FooOp>(loc);\n"
            ")mlir\";\n"
        )
        findings = mlir_lint.check_text("Enzyme/MLIR/Doc.cpp", code)
        self.assertEqual(findings, [])

    def test_nolint_and_nolintnextline_suppression(self):
        code = (
            "void f(const Operation *op) {} // NOLINT(mlir-const-ir-pointer)\n"
            "// NOLINTNEXTLINE(mlir-op-create)\n"
            "void g(OpBuilder &b, Location loc) { b.create<FooOp>(loc); }\n"
        )
        findings = mlir_lint.check_text("Enzyme/MLIR/Suppress.cpp", code)
        self.assertEqual(findings, [])


class TestLLVMCodingStandards(unittest.TestCase):
    """Positive, negative, and edge-case tests for every LLVM coding standard check."""

    POSITIVE_CASES = {
        "llvm-no-iostream": (
            "Enzyme/Foo.cpp",
            "#include <iostream>\nvoid dump() { std::cerr << 42 << std::endl; }\n",
        ),
        "llvm-no-rtti-exceptions": (
            "Enzyme/Foo.cpp",
            "void bad(Base *B) {\n"
            "  if (auto *D = dynamic_cast<Derived *>(B))\n"
            "    throw 1;\n"
            "}\n",
        ),
        "llvm-twine-local": (
            "Enzyme/Foo.cpp",
            "void buildName(StringRef Base) {\n"
            "  Twine FullName = Base + \"_shadow\";\n"
            "}\n",
        ),
        "llvm-std-sort": (
            "Enzyme/Foo.cpp",
            "void sortVec(SmallVectorImpl<int> &Vals) {\n"
            "  std::sort(Vals.begin(), Vals.end());\n"
            "}\n",
        ),
        "llvm-vector-bool": (
            "Enzyme/Foo.cpp",
            "void flags() { std::vector<bool> ActiveFlags; }\n",
        ),
        "llvm-smallvector-param": (
            "Enzyme/Foo.h",
            "#ifndef FOO_H\n#define FOO_H\n"
            "void collect(SmallVector<Value *, 8> &Out);\n"
            "#endif\n",
        ),
        "llvm-const-std-string-param": (
            "Enzyme/Foo.h",
            "#ifndef FOO_H\n#define FOO_H\n"
            "bool hasPrefix(const std::string &Name);\n"
            "#endif\n",
        ),
        "llvm-assert-false": (
            "Enzyme/Foo.cpp",
            "void unreachableCase() { assert(false && \"should not happen\"); }\n",
        ),
        "llvm-assert-no-message": (
            "Enzyme/Foo.cpp",
            "void checkPtr(int *Ptr) { assert(Ptr != nullptr); }\n",
        ),
        "llvm-pragma-once": (
            "Enzyme/Foo.h",
            "#pragma once\nint helper();\n",
        ),
        "llvm-else-after-return": (
            "Enzyme/Foo.cpp",
            "int classify(int X) {\n"
            "  if (X < 0) {\n"
            "    return -1;\n"
            "  } else {\n"
            "    return 1;\n"
            "  }\n"
            "}\n",
        ),
        "llvm-anon-namespace-func": (
            "Enzyme/Foo.cpp",
            "namespace {\n"
            "int computeLocal(int X) {\n"
            "  return X + 1;\n"
            "}\n"
            "} // namespace\n",
        ),
        "llvm-include-comment": (
            "Enzyme/Foo.cpp",
            "#include \"llvm/ADT/SmallVector.h\" // Needed for SmallVector\n",
        ),
        "llvm-ostream-string-newline": (
            "Enzyme/Foo.cpp",
            "void logMsg() { llvm::errs() << \"warning\" << \"\\n\"; }\n",
        ),
        "llvm-non-ascii": (
            "Enzyme/Foo.cpp",
            "// Em-dash \u2014 and unicode arrow \u2192 are non-ASCII.\n",
        ),
        "llvm-lit-stderr-devnull": (
            "test/Enzyme/ReverseMode/foo.ll",
            "; RUN: %opt < %s %OPloadEnzyme -passes=\"enzyme\" -S 2>/dev/null | FileCheck %s\n",
        ),
        "llvm-lit-pinned-ssa": (
            "test/Enzyme/ReverseMode/foo.ll",
            "; RUN: %opt < %s %OPloadEnzyme -passes=\"enzyme\" -S | FileCheck %s\n"
            "; CHECK: %0 = fmul double %x, %x\n",
        ),
    }

    def test_every_registered_llvm_check_has_positive_case(self):
        registered = {spec.name for spec in llvm_lint.CHECK_SPECS}
        self.assertEqual(
            registered,
            set(self.POSITIVE_CASES.keys()),
            "Every LLVM check in CHECK_SPECS must have an explicit test case.",
        )

    def test_positive_cases_trigger_expected_check(self):
        for expected_check, (path, code) in self.POSITIVE_CASES.items():
            with self.subTest(check=expected_check):
                findings = llvm_lint.check_text(path, code)
                checks = {f.check for f in findings}
                self.assertIn(
                    expected_check,
                    checks,
                    f"Expected `{expected_check}` in findings for:\n{code}\nGot: {findings}",
                )

    def test_idiomatic_llvm_cpp_produces_zero_findings(self):
        idiomatic_cpp = (
            "#ifndef ENZYME_FOO_H\n"
            "#define ENZYME_FOO_H\n"
            "#include \"llvm/ADT/ArrayRef.h\" // IWYU pragma: keep\n"
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
            "  assert(!Name.empty() && \"expected non-empty name\");\n"
            "  llvm::sort(Out);\n"
            "  if constexpr (sizeof(void *) == 8) {\n"
            "    return 0;\n"
            "  } else {\n"
            "    return -1;\n"
            "  }\n"
            "  if (Out.size() > 1) {\n"
            "    if (Out.front() == 0)\n"
            "      return 0;\n"
            "  } else {\n"
            "    Out.push_back(1);\n"
            "  }\n"
            "  llvm::errs() << Label << '\\n';\n"
            "  return static_cast<int>(Out.size());\n"
            "}\n"
            "#endif // ENZYME_FOO_H\n"
        )
        findings = llvm_lint.check_text("Enzyme/Foo.h", idiomatic_cpp)
        self.assertEqual(findings, [])

    def test_idiomatic_ll_lit_test_produces_zero_findings(self):
        idiomatic_ll = (
            "; RUN: %opt < %s %OPloadEnzyme -passes=\"enzyme\" -S | FileCheck %s\n"
            "\n"
            "; CHECK-LABEL: define internal double @fwddiffe_square(\n"
            "; CHECK-NEXT: entry:\n"
            "; CHECK-NEXT:   %[[MUL:.+]] = fmul fast double %{{.+}}, 2.000000e+00\n"
            "; CHECK-NEXT:   ret double %[[MUL]]\n"
        )
        findings = llvm_lint.check_text("test/Enzyme/ForwardMode/square.ll", idiomatic_ll)
        self.assertEqual(findings, [])


class TestDiffBaseAndLiveRepoIntegration(unittest.TestCase):
    """End-to-end tests for `--diff-base` line scoping and live repository cleanliness."""

    def test_diff_base_flags_added_lines_only_for_diff_scoped_checks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.check_call(["git", "init", "-q"], cwd=tmpdir)
            subprocess.check_call(
                ["git", "config", "user.email", "test@example.com"], cwd=tmpdir
            )
            subprocess.check_call(
                ["git", "config", "user.name", "Test User"], cwd=tmpdir
            )

            cpp_path = os.path.join(tmpdir, "Sample.cpp")
            # Commit baseline with a legacy diff-scoped issue (`static inline`).
            with open(cpp_path, "w", encoding="utf-8") as fh:
                fh.write(
                    "/// Legacy baseline function.\n"
                    "static inline int legacyHelper(int x) { return x; }\n"
                )
            subprocess.check_call(["git", "add", "Sample.cpp"], cwd=tmpdir)
            subprocess.check_call(["git", "commit", "-q", "-m", "baseline"], cwd=tmpdir)

            # Unmodified diff against HEAD must pass cleanly (0 findings).
            rc_clean = mlir_lint.main(["--diff-base", "HEAD", cpp_path])
            self.assertEqual(rc_clean, 0)

            # Add a new diff-scoped violation on a modified line (`builder.create<...>`).
            with open(cpp_path, "w", encoding="utf-8") as fh:
                fh.write(
                    "/// Legacy baseline function.\n"
                    "static inline int legacyHelper(int x) { return x; }\n"
                    "/// Newly added bad function.\n"
                    "void emitBad(OpBuilder &b, Location loc) { b.create<FooOp>(loc); }\n"
                )
            changed = mlir_lint.compute_changed_lines("HEAD", cwd=tmpdir)
            with open(cpp_path, encoding="utf-8") as fh:
                raw_findings = mlir_lint.check_text(cpp_path, fh.read())
            filtered = mlir_lint.filter_findings(
                raw_findings, all_checks=False, changed_lines=changed
            )
            self.assertEqual([f.check for f in filtered], ["mlir-op-create"])

            # Whole-tree check (`const Operation *`) is caught even on untouched baseline!
            whole_tree_path = os.path.join(tmpdir, "WholeTree.cpp")
            with open(whole_tree_path, "w", encoding="utf-8") as fh:
                fh.write("/// Bad.\nvoid bad(const Operation *op) {}\n")
            subprocess.check_call(["git", "add", "WholeTree.cpp"], cwd=tmpdir)
            subprocess.check_call(
                ["git", "commit", "-q", "-m", "baseline with whole-tree bug"],
                cwd=tmpdir,
            )
            rc_whole = mlir_lint.main(["--diff-base", "HEAD", whole_tree_path])
            self.assertEqual(rc_whole, 1)

    def test_enzyme_repository_passes_whole_tree_checks(self):
        enzyme_dir = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
        mlir_rc = mlir_lint.main(
            [
                os.path.join(enzyme_dir, "Enzyme", "MLIR"),
                os.path.join(enzyme_dir, "test", "MLIR"),
            ]
        )
        self.assertEqual(mlir_rc, 0)

        llvm_rc = llvm_lint.main(
            [
                os.path.join(enzyme_dir, "Enzyme"),
                os.path.join(enzyme_dir, "BCLoad"),
                os.path.join(enzyme_dir, "tools"),
                os.path.join(enzyme_dir, "test"),
            ]
        )
        self.assertEqual(llvm_rc, 0)


if __name__ == "__main__":
    unittest.main()
