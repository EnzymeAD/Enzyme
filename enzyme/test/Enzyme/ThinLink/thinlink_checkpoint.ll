; RUN: if [ %llvmver -ge 16 ]; then rm -rf %t && mkdir -p %t && %opt < %s %OPnewLoadEnzyme -passes="enzyme-summary" -enzyme-summary-out=%t/main.json -disable-output && %opt < %S/Inputs/thinlink_step.ll.in %OPnewLoadEnzyme -passes="enzyme-summary" -enzyme-summary-out=%t/step.json -disable-output && python3 %S/../../../scripts/enzyme_thinlink.py --inactive both --out %t/plan %t/main.json %t/step.json > /dev/null; fi
; RUN: if [ %llvmver -ge 16 ]; then FileCheck %s --check-prefix=EXPORTS < %t/plan/step.exports; fi
; RUN: if [ %llvmver -ge 16 ]; then FileCheck %s --check-prefix=INACTIVE < %t/plan/inactive.c; fi
; RUN: if [ %llvmver -ge 16 ]; then FileCheck %s --check-prefix=GLOBALS < %t/plan/checkpoint_globals.txt; fi

; A function that hands the step of a checkpointed loop to
; __enzyme_checkpoint_for moves no floating-point data itself, but it is
; where the loop's derivative happens: it is not inferred inactive (nor are
; its callers), and the step it passes is part of the differentiated call
; graph, exported from its module. The snapshots of the loop hold the
; globals the step writes, which its own module's checkpointing pass cannot
; see beyond the calls into other modules (checkpoint_globals.txt).

declare void @step(ptr)
declare void @__enzyme_checkpoint_for(...)
declare void @__enzyme_autodiff(...)

define void @loop_shim(ptr %x, ptr %n) {
  call void (...) @__enzyme_checkpoint_for(ptr @step, ptr %n, ptr %x)
  ret void
}

define void @run(ptr %x, ptr %n) {
  call void @loop_shim(ptr %x, ptr %n)
  ret void
}

define void @caller(ptr %x, ptr %dx, ptr %n) {
  call void (...) @__enzyme_autodiff(ptr @run, ptr %x, ptr %dx, ptr %n, ptr %n)
  ret void
}

; EXPORTS: step reverse

; INACTIVE-NOT: loop_shim
; INACTIVE-NOT: {{[^_]}}run

; GLOBALS: step state_ 16
