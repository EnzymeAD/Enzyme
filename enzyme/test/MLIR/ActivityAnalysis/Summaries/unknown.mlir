// RUN: %eopt --print-activity-analysis='relative verbose' --split-input-file %s | FileCheck %s

// @callee writes through its pointer and then does something the analysis
// cannot see past, so its dense origins map goes unknown. serializeMapOfSetsNaive
// used to say that as a one-element list, ["<unknown>"], while every reader of
// these summaries -- and serializeSetNaive, which writes the sparse ones --
// says and expects the bare marker. The list read as a set of one known origin,
// whose element was then cast from a StringAttr to an OriginAttr and indexed
// with whatever came out.

func.func @callee(%p: !llvm.ptr, %v: f64) {
  llvm.store %v, %p : f64, !llvm.ptr
  llvm.inline_asm has_side_effects "nop", "" : () -> ()
  return
}

// CHECK-LABEL: processing function @callee
// CHECK: forward value origins:
// CHECK-NEXT: originates from "<unknown>"

func.func @caller(%p: !llvm.ptr, %v: f64) -> f64 {
  call @callee(%p, %v) : (!llvm.ptr, f64) -> ()
  %r = llvm.load %p {tag = "reload"} : !llvm.ptr -> f64
  return %r : f64
}

// The caller reads what the callee wrote, so what it loads is of unknown
// origin too -- not of no origin, which is what a dropped marker would say.

// CHECK-LABEL: processing function @caller
// CHECK: forward value origins:
// CHECK-NEXT: originates from "<unknown>"
// CHECK: "reload"(#0)
// CHECK-NEXT: sources: "<unknown>"
