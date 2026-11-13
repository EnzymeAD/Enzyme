; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s; fi

@vtable = internal constant [1 x ptr] [ptr @callee]
@enzyme_dup = external global i32

define internal void @callee(ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer, [-1,8]:Float@double}" %ctx, ptr %out) {
entry:
  %n = load i64, ptr %ctx, align 8
  %n1 = add i64 %n, 1
  store i64 %n1, ptr %ctx, align 8
  %p = getelementptr inbounds i8, ptr %ctx, i64 8
  %v = load double, ptr %p, align 8
  %sq = fmul double %v, %v
  store double %sq, ptr %out, align 8
  ret void
}

define void @caller(ptr %ctx, ptr %out) {
entry:
  %fp = load ptr, ptr @vtable, align 8
  call void %fp(ptr %ctx, ptr %out)
  ret void
}

declare void @__enzyme_fwddiff(ptr, ...)
define void @dcaller(ptr %ctx, ptr %dctx, ptr %out, ptr %dout) {
entry:
  call void (ptr, ...) @__enzyme_fwddiff(ptr @caller, ptr @enzyme_dup, ptr %ctx, ptr %dctx, ptr @enzyme_dup, ptr %out, ptr %dout)
  ret void
}

; CHECK: @vtable_shadow = internal constant [1 x ptr] [ptr @"_enzyme_forward_callee'"]
; CHECK: define internal void @fwddiffecallee(ptr {{.*}}%ctx, ptr {{.*}}%"ctx'", ptr {{.*}}%out, ptr {{.*}}%"out'")
; CHECK:   %n = load i64, ptr %ctx
; CHECK:   %n1 = add i64 %n, 1
; CHECK:   store i64 %n1, ptr %"ctx'"
; CHECK:   store i64 %n1, ptr %ctx
; CHECK:   %"v'ipl" = load double, ptr %"p'ipg"
; CHECK:   %v = load double, ptr %p
; CHECK:   %[[m0:.+]] = fmul fast double %"v'ipl", %v
; CHECK:   %[[m1:.+]] = fmul fast double %"v'ipl", %v
; CHECK:   %[[sum:.+]] = fadd fast double %[[m0]], %[[m1]]
; CHECK:   store double %[[sum]], ptr %"out'"
; CHECK:   ret void
