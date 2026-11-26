; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s --check-prefix=NOREQ; fi
; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -S | FileCheck %s --check-prefix=NOREQ; fi

; Outer requests whose targets reach inner requests only through ordinary
; calls (h -> wrap -> df -> __enzyme_autodiff). Callers are defined before
; their callees (except @pre_df), so the inner requests must be lowered while
; preparing the outer one.

@enzyme_dup = external global ptr

@edge_alias = alias double (double), ptr @edge_df

; Defined first, so the module traversal lowers it before @edge_h.
define double @pre_df(double %x) {
entry:
  %r = call double (...) @__enzyme_autodiff(ptr @cube, double %x)
  ret double %r
}

define double @h(double %x, double %dx) {
entry:
  %r = call double (...) @__enzyme_fwddiff(ptr @wrap, double %x, double %dx)
  ret double %r
}

define double @multi_h(double %x, double %dx) {
entry:
  %r = call double (...) @__enzyme_fwddiff(ptr @multi, double %x, double %dx)
  ret double %r
}

define double @rec_h(double %x, double %dx, i64 %n) {
entry:
  %r = call double (...) @__enzyme_fwddiff(ptr @rec, double %x, double %dx, metadata !"enzyme_const", i64 %n)
  ret double %r
}

define double @direct_h(double %x, double %dx) {
entry:
  %a = call double @br_df(double %x)
  %r = call double (...) @__enzyme_fwddiff(ptr @br_wrap, double %x, double %dx)
  %s = fadd double %a, %r
  ret double %s
}

define void @rust_h(ptr %x, ptr %dx, ptr %g, ptr %dg) {
entry:
  call void (...) @__enzyme_fwddiff(ptr @rust_wrap, ptr @enzyme_dup, ptr %x, ptr %dx, ptr @enzyme_dup, ptr %g, ptr %dg)
  ret void
}

define double @rev_h(double %x) {
entry:
  %r = call double (...) @__enzyme_autodiff(ptr @rev_wrap, double %x)
  ret double %r
}

define double @fwd_h(double %x, double %dx) {
entry:
  %r = call double (...) @__enzyme_fwddiff(ptr @fwd_wrap, double %x, double %dx)
  ret double %r
}

define double @edge_h(double %x, double %dx) {
entry:
  %r = call double (...) @__enzyme_fwddiff(ptr @edge_wrap, double %x, double %dx)
  ret double %r
}

define double @multi(double %x) {
entry:
  %a = call double @wrap(double %x)
  %b = call double @wrap2(double %x)
  %m = fmul double %b, %x
  %r = fadd double %a, %m
  ret double %r
}

define double @rec(double %x, i64 %n) {
entry:
  %cmp = icmp eq i64 %n, 0
  br i1 %cmp, label %base, label %step

base:
  %d = call double @df(double %x)
  ret double %d

step:
  %n1 = sub i64 %n, 1
  %r = call double @rec(double %x, i64 %n1)
  ret double %r
}

define double @wrap2(double %x) {
entry:
  %r = call double @wrap(double %x)
  ret double %r
}

define double @wrap(double %x) {
entry:
  %r = call double @df(double %x)
  ret double %r
}

define double @df(double %x) {
entry:
  %r = call double (...) @__enzyme_autodiff(ptr @sq, double %x)
  ret double %r
}

define double @sq(double %x) {
entry:
  %m = fmul double %x, %x
  ret double %m
}

define double @br_wrap(double %x) {
entry:
  %r = call double @br_df(double %x)
  ret double %r
}

define double @br_df(double %x) {
entry:
  %c = fcmp ogt double %x, 0.000000e+00
  br i1 %c, label %pos, label %neg

pos:
  %r = call double (...) @__enzyme_autodiff(ptr @sq, double %x)
  ret double %r

neg:
  %r2 = call double (...) @__enzyme_autodiff(ptr @sq, double %x)
  %r3 = fmul double %r2, 3.000000e+00
  ret double %r3
}

define double @cube(double %x) {
entry:
  %m = fmul double %x, %x
  %c = fmul double %m, %x
  ret double %c
}

define double @rev_wrap(double %x) {
entry:
  %r = call double @rev_df(double %x)
  ret double %r
}

define double @rev_df(double %x) {
entry:
  %r = call double (...) @__enzyme_autodiff(ptr @cube, double %x)
  ret double %r
}

define double @fwd_wrap(double %x) {
entry:
  %r = call double @fwd_df(double %x)
  ret double %r
}

define double @fwd_df(double %x) {
entry:
  %r = call double (...) @__enzyme_fwddiff(ptr @cube, double %x, double 1.000000e+00)
  ret double %r
}

; Reaches a declaration, a request behind an alias and the lowered @pre_df.
define double @edge_wrap(double %x) {
entry:
  call void @trace()
  %a = call double @edge_alias(double %x)
  %b = call double @pre_df(double %x)
  %r = fadd double %a, %b
  ret double %r
}

define double @edge_df(double %x) {
entry:
  %r = call double (...) @__enzyme_autodiff(ptr @sq, double %x)
  ret double %r
}

declare void @trace() "enzyme_inactive"

define void @rust_wrap(ptr "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" %x, ptr "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" %g) {
entry:
  call void @rust_df(ptr %x, ptr %g)
  ret void
}

define void @rust_df(ptr %x, ptr %g) {
entry:
  %out = alloca double, align 8
  %dout = alloca double, align 8
  store double 1.000000e+00, ptr %dout, align 8
  call void (...) @__enzyme_autodiff(ptr @rust_f, ptr @enzyme_dup, ptr %x, ptr %g, ptr @enzyme_dup, ptr %out, ptr %dout)
  ret void
}

; f(x) = x[0] * x[0] + x[1] * x[0]
define void @rust_f(ptr "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" %x, ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double}" %out) {
entry:
  %x0 = load double, ptr %x, align 8
  %x1p = getelementptr inbounds double, ptr %x, i64 1
  %x1 = load double, ptr %x1p, align 8
  %sq = fmul double %x0, %x0
  %prod = fmul double %x1, %x0
  %sum = fadd double %sq, %prod
  store double %sum, ptr %out, align 8
  ret void
}

declare double @__enzyme_fwddiff(...)

declare double @__enzyme_autodiff(...)
; NOREQ: define double @pre_df(
; NOREQ-NOT: call {{.*}}@__enzyme_fwddiff(
; NOREQ-NOT: call {{.*}}@__enzyme_autodiff(
; NOREQ-NOT: fwddiffe__enzyme_

; CHECK: define double @pre_df(double %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %{{.+}} = call {{(fast )?}}{ double } @diffecube(double %x, double 1.000000e+00)

; CHECK: define double @h(double %x, double %dx)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[r:[^ ]+]] = call fast double @fwddiffewrap(double %x, double %dx)
; CHECK-NEXT:   ret double %[[r]]

; CHECK: define double @multi_h(double %x, double %dx)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[r:[^ ]+]] = call fast double @fwddiffemulti(double %x, double %dx)
; CHECK-NEXT:   ret double %[[r]]

; CHECK: define double @rec_h(double %x, double %dx, i64 %n)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[r:[^ ]+]] = call fast double @fwddifferec(double %x, double %dx, i64 %n)
; CHECK-NEXT:   ret double %[[r]]

; @br_df is inlined here with its gradients 2x and 3 * 2x.
; CHECK: define double @direct_h(double %x, double %dx)
; CHECK: pos.i:
; CHECK-NEXT:   %[[pos:[^ ]+]] = fadd fast double %x, %x
; CHECK: neg.i:
; CHECK-NEXT:   %[[neg2:[^ ]+]] = fadd fast double %x, %x
; CHECK:   %[[neg:[^ ]+]] = fmul double %[[neg2]], 3.000000e+00
; CHECK: br_df.exit:
; CHECK-NEXT:   %[[a:[^ ]+]] = phi double [ %[[pos]], %pos.i ], [ %[[neg]], %neg.i ]
; CHECK-NEXT:   %[[r:[^ ]+]] = call fast double @fwddiffebr_wrap(double %x, double %dx)
; CHECK-NEXT:   %s = fadd double %[[a]], %[[r]]
; CHECK-NEXT:   ret double %s

; CHECK: define void @rust_h(ptr %x, ptr %dx, ptr %g, ptr %dg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @fwddifferust_wrap(ptr %x, ptr %dx, ptr %g, ptr %dg)
; CHECK-NEXT:   ret void

; CHECK: define double @rev_h(double %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %{{.+}} = call {{(fast )?}}{ double } @differev_wrap(double %x, double 1.000000e+00)

; CHECK: define double @fwd_h(double %x, double %dx)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[r:[^ ]+]] = call fast double @fwddiffefwd_wrap(double %x, double %dx)
; CHECK-NEXT:   ret double %[[r]]

; CHECK: define double @edge_h(double %x, double %dx)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[r:[^ ]+]] = call fast double @fwddiffeedge_wrap(double %x, double %dx)
; CHECK-NEXT:   ret double %[[r]]

; CHECK: define internal double @fwddiffewrap(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[r:[^ ]+]] = fadd fast double %"x'", %"x'"
; CHECK-NEXT:   ret double %[[r]]

; d/dx (2x + 2x * x) = 2 + 4x
; CHECK: define internal double @fwddiffemulti(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[a:[^ ]+]] = fadd fast double %"x'", %"x'"
; CHECK-NEXT:   %[[w:[^ ]+]] = call {{(fast )?}}{ double, double } @fwddiffewrap2(double %x, double %"x'")
; CHECK-NEXT:   %[[b:[^ ]+]] = extractvalue { double, double } %[[w]], 0
; CHECK-NEXT:   %[[db:[^ ]+]] = extractvalue { double, double } %[[w]], 1
; CHECK-NEXT:   %[[p:[^ ]+]] = fmul fast double %[[db]], %x
; CHECK-NEXT:   %[[q:[^ ]+]] = fmul fast double %"x'", %[[b]]
; CHECK-NEXT:   %[[pq:[^ ]+]] = fadd fast double %[[p]], %[[q]]
; CHECK-NEXT:   %[[r:[^ ]+]] = fadd fast double %[[a]], %[[pq]]
; CHECK-NEXT:   ret double %[[r]]

; CHECK: define internal { double, double } @fwddiffewrap2(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[v:[^ ]+]] = fadd fast double %x, %x
; CHECK-NEXT:   %[[dv:[^ ]+]] = fadd fast double %"x'", %"x'"
; CHECK-NEXT:   %[[s:[^ ]+]] = insertvalue { double, double } undef, double %[[v]], 0
; CHECK-NEXT:   %[[s2:[^ ]+]] = insertvalue { double, double } %[[s]], double %[[dv]], 1
; CHECK-NEXT:   ret { double, double } %[[s2]]

; CHECK: define internal double @fwddifferec(double %x, double %"x'", i64 %n)
; CHECK: base:
; CHECK-NEXT:   %[[r:[^ ]+]] = fadd fast double %"x'", %"x'"
; CHECK-NEXT:   ret double %[[r]]
; CHECK: step:
; CHECK-NEXT:   %n1 = sub i64 %n, 1
; CHECK-NEXT:   %[[r:[^ ]+]] = call fast double @fwddifferec.1(double %x, double %"x'", i64 %n1)
; CHECK-NEXT:   ret double %[[r]]

; CHECK: define internal double @fwddiffebr_wrap(double %x, double %"x'")
; CHECK: pos.i:
; CHECK-NEXT:   %[[pos:[^ ]+]] = fadd fast double %"x'", %"x'"
; CHECK: neg.i:
; CHECK-NEXT:   %[[neg2:[^ ]+]] = fadd fast double %"x'", %"x'"
; CHECK-NEXT:   %[[neg:[^ ]+]] = fmul fast double %[[neg2]], 3.000000e+00
; CHECK: br_df.exit:
; CHECK-NEXT:   %[[r:[^ ]+]] = phi fast double [ %[[pos]], %pos.i ], [ %[[neg]], %neg.i ]
; CHECK-NEXT:   ret double %[[r]]

; dg += H * dx, where H = [[2, 1], [1, 0]] is the Hessian of @rust_f.
; CHECK: define internal void @fwddifferust_wrap(ptr {{.*}}%x, ptr {{.*}}%"x'", ptr {{.*}}%g, ptr {{.*}}%"g'")
; CHECK:   %[[dx0:[^ ]+]] = load double, ptr %"x'"
; CHECK:   %[[dg1p:[^ ]+]] = getelementptr inbounds double, ptr %"g'", i64 1
; CHECK:   %[[dx1p:[^ ]+]] = getelementptr inbounds double, ptr %"x'", i64 1
; CHECK:   %[[dx1:[^ ]+]] = load double, ptr %[[dx1p]]
; CHECK:   %[[t0:[^ ]+]] = fmul fast double {{.*}}%[[dx0]]{{.*}}
; CHECK:   %[[t1:[^ ]+]] = fmul fast double {{.*}}%[[dx1]]{{.*}}
; CHECK:   %[[s0:[^ ]+]] = fadd fast double %[[t1]], %[[t0]]
; CHECK:   %[[s1:[^ ]+]] = fadd fast double %[[s0]], %[[t0]]
; CHECK:   %[[dg1:[^ ]+]] = load double, ptr %[[dg1p]]
; CHECK:   %[[ndg1:[^ ]+]] = fadd fast double %[[dg1]], %[[t0]]
; CHECK:   store double %[[ndg1]], ptr %[[dg1p]]
; CHECK:   %[[dg0:[^ ]+]] = load double, ptr %"g'"
; CHECK:   %[[ndg0:[^ ]+]] = fadd fast double %[[dg0]], %[[s1]]
; CHECK:   store double %[[ndg0]], ptr %"g'"

; Reverse over reverse: only check that the inner gradient was inlined.
; CHECK: define internal { double } @differev_wrap(double %x, double %differeturn)
; CHECK-NOT: call
; CHECK: ret { double }

; d/dx 3x^2 = 6x
; CHECK: define internal double @fwddiffefwd_wrap(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[a:[^ ]+]] = fmul fast double %"x'", %x
; CHECK-NEXT:   %[[b:[^ ]+]] = fmul fast double %"x'", %x
; CHECK-NEXT:   %[[ab:[^ ]+]] = fadd fast double %[[a]], %[[b]]
; CHECK-NEXT:   %[[x2:[^ ]+]] = fadd fast double %x, %x
; CHECK-NEXT:   %[[d2:[^ ]+]] = fadd fast double %"x'", %"x'"
; CHECK-NEXT:   %[[p:[^ ]+]] = fmul fast double %[[d2]], %x
; CHECK-NEXT:   %[[q:[^ ]+]] = fmul fast double %"x'", %[[x2]]
; CHECK-NEXT:   %[[pq:[^ ]+]] = fadd fast double %[[p]], %[[q]]
; CHECK-NEXT:   %[[r:[^ ]+]] = fadd fast double %[[pq]], %[[ab]]
; CHECK-NEXT:   ret double %[[r]]

; CHECK: define internal double @fwddiffeedge_wrap(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @trace()
; CHECK-NEXT:   %[[a:[^ ]+]] = call fast double @fwddiffeedge_df(double %x, double %"x'")
; CHECK-NEXT:   %[[b:[^ ]+]] = call fast double @fwddiffepre_df(double %x, double %"x'")
; CHECK-NEXT:   %[[r:[^ ]+]] = fadd fast double %[[a]], %[[b]]
; CHECK-NEXT:   ret double %[[r]]

; CHECK: define internal double @fwddiffeedge_df(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[r:[^ ]+]] = fadd fast double %"x'", %"x'"
; CHECK-NEXT:   ret double %[[r]]

; CHECK: define internal double @fwddiffepre_df(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %{{.+}} = call {{(fast )?}}{ double } @fwddiffediffecube(double %x, double %"x'", double 1.000000e+00)
