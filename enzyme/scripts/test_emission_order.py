#!/usr/bin/env python3
"""Tests for check_emission_order.py."""

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "check_emission_order.py"
)

HELPER = """
static Value *loadField(IRBuilder<> &B, Type *T, Value *p) {
  return B.CreateLoad(T, p);
}
"""


class EmissionOrderTest(unittest.TestCase):
    def check(self, source):
        """Runs the checker with --include-llvm-builder on one file."""
        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root, "t.cpp"), "w") as f:
                f.write(textwrap.dedent(source))
            return subprocess.run(
                [sys.executable, SCRIPT, "--include-llvm-builder", root],
                capture_output=True,
                text=True,
                check=False,
            )

    def assertFlags(self, source, callee):
        proc = self.check(source)
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn(f"{callee}(...) builds two or more ops", proc.stdout)

    def assertClean(self, source):
        proc = self.check(source)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_two_builder_calls(self):
        self.assertFlags(
            """
            Value *f(IRBuilder<> &B, Value *a, Value *b) {
              return B.CreateAdd(B.CreateMul(a, b), B.CreateMul(b, a));
            }
            """,
            "CreateAdd",
        )

    def test_two_helper_calls(self):
        self.assertFlags(
            HELPER
            + """
            Value *f(IRBuilder<> &B, Type *T, Value *p, Value *q) {
              return B.CreateICmpEQ(loadField(B, T, p), loadField(B, T, q));
            }
            """,
            "CreateICmpEQ",
        )

    def test_lambda_forwarding_to_a_helper(self):
        self.assertFlags(
            HELPER
            + """
            Value *f(IRBuilder<> &B, Type *T, Value *p, Value *q) {
              auto value = [&](Value *v) { return loadField(B, T, v); };
              return B.CreateICmpEQ(value(p), value(q));
            }
            """,
            "CreateICmpEQ",
        )

    def test_lambda_forwarding_to_a_forwarding_lambda(self):
        self.assertFlags(
            HELPER
            + """
            Value *f(IRBuilder<> &B, Type *T, Value *p, Value *q) {
              auto value = [&](Value *v) { return loadField(B, T, v); };
              auto field = [&](Value *v) { return value(v); };
              return B.CreateICmpEQ(field(p), field(q));
            }
            """,
            "CreateICmpEQ",
        )

    def test_forwarded_reads_bound_to_locals(self):
        self.assertClean(
            HELPER
            + """
            Value *f(IRBuilder<> &B, Type *T, Value *p, Value *q) {
              auto value = [&](Value *v) { return loadField(B, T, v); };
              Value *lhs = value(p);
              Value *rhs = value(q);
              return B.CreateICmpEQ(lhs, rhs);
            }
            """
        )


if __name__ == "__main__":
    unittest.main()
