; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="print<enzyme-function-summary>" -disable-output | FileCheck %s; fi

; The flow and points-to matrices of the function summary, through local
; memory, phis and calls. Sources and sinks: "a<i>" argument i (its value or
; the memory reachable from it), "g" any global, "ret" the return value.

@g = global double 0.0

declare void @callee(ptr, double)
declare ptr @ret_ptr(ptr)

; x by value flows to y's memory and the return value; z is only read.
define double @simple(double %x, ptr %y, ptr %z) {
  %v = load double, ptr %z
  %m = fmul double %x, %v
  store double %m, ptr %y
  ret double %x
}
; CHECK-LABEL: enzyme-function-summary simple:
; CHECK-SAME: "args":[{"escape":false,"read":false,"write":false},{"escape":false,"read":false,"write":true},{"escape":false,"read":true,"write":false}]
; CHECK-SAME: "edges":[]
; CHECK-SAME: "flow":{"a0":["a1","ret"],"a2":["a1"]}
; CHECK-SAME: "globals_read":[]
; CHECK-SAME: "globals_write":[]
; CHECK-SAME: "pts":{}
; CHECK-SAME: "unknown":false

; global data reaches the return value; arg memory reaches the global
define double @globals(ptr %a) {
  %v = load double, ptr %a
  store double %v, ptr @g
  %w = load double, ptr @g
  ret double %w
}
; CHECK-LABEL: enzyme-function-summary globals:
; CHECK-SAME: "args":[{"escape":false,"read":true,"write":false}]
; CHECK-SAME: "edges":[]
; CHECK-SAME: "flow":{"a0":["g"],"g":["ret"]}
; CHECK-SAME: "globals_read":["g"]
; CHECK-SAME: "globals_write":["g"]
; CHECK-SAME: "pts":{}
; CHECK-SAME: "unknown":false

; data through a stack slot, and a pointer through a stack slot
define void @through_alloca(double %x, ptr %p, ptr %q) {
  %slot = alloca double
  store double %x, ptr %slot
  %v = load double, ptr %slot
  %pslot = alloca ptr
  store ptr %p, ptr %pslot
  %pp = load ptr, ptr %pslot
  store double %v, ptr %pp
  ret void
}
; CHECK-LABEL: enzyme-function-summary through_alloca:
; CHECK-SAME: "args":[{"escape":false,"read":false,"write":false},{"escape":false,"read":false,"write":true},{"escape":false,"read":false,"write":false}]
; CHECK-SAME: "edges":[]
; CHECK-SAME: "flow":{"a0":["a1"]}
; CHECK-SAME: "globals_read":[]
; CHECK-SAME: "globals_write":[]
; CHECK-SAME: "pts":{}
; CHECK-SAME: "unknown":false

; a pointer to b's memory is stored into a's: b escapes, and is reachable
; from a
define void @publish(ptr %a, ptr %b) {
  store ptr %b, ptr %a
  ret void
}
; CHECK-LABEL: enzyme-function-summary publish:
; CHECK-SAME: "args":[{"escape":false,"read":false,"write":false},{"escape":true,"read":false,"write":false}]
; CHECK-SAME: "edges":[]
; CHECK-SAME: "flow":{"a1":["a0"]}
; CHECK-SAME: "globals_read":[]
; CHECK-SAME: "globals_write":[]
; CHECK-SAME: "pts":{"a1":["a0"]}
; CHECK-SAME: "unknown":false

; pointer returned
define ptr @ident(ptr %a) {
  ret ptr %a
}
; CHECK-LABEL: enzyme-function-summary ident:
; CHECK-SAME: "args":[{"escape":true,"read":false,"write":false}]
; CHECK-SAME: "edges":[]
; CHECK-SAME: "flow":{"a0":["ret"]}
; CHECK-SAME: "globals_read":[]
; CHECK-SAME: "globals_write":[]
; CHECK-SAME: "pts":{"a0":["ret"]}
; CHECK-SAME: "unknown":false

; by-value data given to a callee may reach what the callee may write; the
; memory of %p is passed on as an edge
define void @calls(double %x, ptr %p) {
  call void @callee(ptr %p, double %x)
  ret void
}
; CHECK-LABEL: enzyme-function-summary calls:
; CHECK-SAME: "args":[{"escape":false,"read":false,"write":false},{"escape":false,"read":false,"write":false}]
; CHECK-SAME: "edges":{{\[\[}}"a1","callee",0{{\]\]}}
; CHECK-SAME: "flow":{"a0":["a1","g"]}
; CHECK-SAME: "globals_read":[]
; CHECK-SAME: "globals_write":[]
; CHECK-SAME: "pts":{}
; CHECK-SAME: "unknown":false

; a callee fills a local buffer (sret): data and pointers in it come from
; what the callee can reach
define double @sret(ptr %p) {
  %buf = alloca double
  call void @callee(ptr %buf, double 0.0)
  %v = load double, ptr %buf
  ret double %v
}
; CHECK-LABEL: enzyme-function-summary sret:
; CHECK-SAME: "args":[{"escape":false,"read":false,"write":false}]
; CHECK-SAME: "edges":{{\[\[}}"g*","callee",0{{\]\]}}
; CHECK-SAME: "flow":{"g":["ret","g"]}
; CHECK-SAME: "globals_read":[]
; CHECK-SAME: "globals_write":[]
; CHECK-SAME: "pts":{}
; CHECK-SAME: "unknown":false

; a pointer a callee returned points into its arguments' memory or a global
define void @callee_ptr(ptr %p, double %x) {
  %r = call ptr @ret_ptr(ptr %p)
  store double %x, ptr %r
  ret void
}
; CHECK-LABEL: enzyme-function-summary callee_ptr:
; CHECK-SAME: "args":[{"escape":false,"read":false,"write":true},{"escape":false,"read":false,"write":false}]
; CHECK-SAME: "edges":{{\[\[}}"a0","ret_ptr",0{{\]\]}}
; CHECK-SAME: "flow":{"a1":["a0","g"]}
; CHECK-SAME: "globals_read":[]
; CHECK-SAME: "globals_write":["*"]
; CHECK-SAME: "pts":{}
; CHECK-SAME: "unknown":false

; phi cycle
define void @loop(ptr %a, ptr %b, double %x, i1 %c) {
entry:
  br label %l
l:
  %p = phi ptr [ %a, %entry ], [ %q, %l ]
  %q = phi ptr [ %b, %entry ], [ %p, %l ]
  store double %x, ptr %p
  br i1 %c, label %l, label %e
e:
  ret void
}
; CHECK-LABEL: enzyme-function-summary loop:
; CHECK-SAME: "args":[{"escape":false,"read":false,"write":true},{"escape":false,"read":false,"write":true},{"escape":false,"read":false,"write":false},{"escape":false,"read":false,"write":false}]
; CHECK-SAME: "edges":[]
; CHECK-SAME: "flow":{"a2":["a0","a1"]}
; CHECK-SAME: "globals_read":[]
; CHECK-SAME: "globals_write":[]
; CHECK-SAME: "pts":{}
; CHECK-SAME: "unknown":false

