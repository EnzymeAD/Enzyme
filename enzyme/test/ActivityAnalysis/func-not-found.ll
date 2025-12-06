; RUN: not %opt < %s %newLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=nonexistent -S 2>&1 | FileCheck %s

define void @foo(i64* %x) {
entry:
  ret void
}

; CHECK: Enzyme: Function 'nonexistent' specified in -activity-analysis-func not found in module
