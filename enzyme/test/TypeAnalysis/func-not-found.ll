; RUN: not %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=nonexistent -S 2>&1 | FileCheck %s

define void @foo(i64* %x) {
entry:
  ret void
}

; CHECK: Function 'nonexistent' specified in -type-analysis-func not found in module
