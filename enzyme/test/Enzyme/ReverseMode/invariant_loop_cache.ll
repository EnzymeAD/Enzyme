; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -S | %lli -

declare ptr @__enzyme_augmentfwd(ptr, ...)
declare void @__enzyme_reverse(ptr, ...)
declare void @__enzyme_fwdsplit(ptr, ...)
declare void @__enzyme_autodiff(ptr, ...)
declare double @llvm.fabs.f64(double)

; 1. Split-mode 3-loop matmul (BT=3, OC=5, C=4):
;    inp[bt, i] is invariant in o -> 2D cache 3*4*8 = 96 bytes (not 480)
;    weight[o, i] is invariant in bt -> 2D cache 5*4*8 = 160 bytes (not 480)
define void @matmul_split(ptr noalias %out, ptr noalias %inp, ptr noalias %weight) {
entry:
  br label %loop.bt

loop.bt:
  %bt = phi i64 [ 0, %entry ], [ %bt.next, %loop.bt.latch ]
  %bt.c = mul nuw nsw i64 %bt, 4
  %bt.oc = mul nuw nsw i64 %bt, 5
  br label %loop.o

loop.o:
  %o = phi i64 [ 0, %loop.bt ], [ %o.next, %loop.o.latch ]
  %o.c = mul nuw nsw i64 %o, 4
  br label %loop.i

loop.i:
  %i = phi i64 [ 0, %loop.o ], [ %i.next, %loop.i ]
  %acc = phi double [ 0.0, %loop.o ], [ %acc.next, %loop.i ]
  %inp.idx = add nuw nsw i64 %bt.c, %i
  %inp.ptr = getelementptr inbounds double, ptr %inp, i64 %inp.idx
  %inp.val = load double, ptr %inp.ptr, align 8
  %w.idx = add nuw nsw i64 %o.c, %i
  %w.ptr = getelementptr inbounds double, ptr %weight, i64 %w.idx
  %w.val = load double, ptr %w.ptr, align 8
  %prod = fmul double %inp.val, %w.val
  %acc.next = fadd double %acc, %prod
  %i.next = add nuw nsw i64 %i, 1
  %i.cmp = icmp eq i64 %i.next, 4
  br i1 %i.cmp, label %loop.o.latch, label %loop.i

loop.o.latch:
  %out.idx = add nuw nsw i64 %bt.oc, %o
  %out.ptr = getelementptr inbounds double, ptr %out, i64 %out.idx
  store double %acc.next, ptr %out.ptr, align 8
  %o.next = add nuw nsw i64 %o, 1
  %o.cmp = icmp eq i64 %o.next, 5
  br i1 %o.cmp, label %loop.bt.latch, label %loop.o

loop.bt.latch:
  %bt.next = add nuw nsw i64 %bt, 1
  %bt.cmp = icmp eq i64 %bt.next, 3
  br i1 %bt.cmp, label %exit, label %loop.bt

exit:
  ret void
}

; 2. Split-mode triangular attention scores (T=4, H=3):
;    q[t, h] is invariant in t2 -> flat 4*3*8 = 96 bytes (collapsing 2-chunk cache)
;    k[t2, h] depends on t2 whose bound depends on t -> stays 2-chunk (32 + dynamic)
define void @triangular_attn_split(ptr noalias %out, ptr noalias %q, ptr noalias %k) {
entry:
  br label %loop.t

loop.t:
  %t = phi i64 [ 0, %entry ], [ %t.next, %loop.t.latch ]
  %t.h = mul nuw nsw i64 %t, 3
  %t.T = mul nuw nsw i64 %t, 4
  %t.lim = add nuw nsw i64 %t, 1
  br label %loop.t2

loop.t2:
  %t2 = phi i64 [ 0, %loop.t ], [ %t2.next, %loop.t2.latch ]
  %t2.h = mul nuw nsw i64 %t2, 3
  br label %loop.h

loop.h:
  %h = phi i64 [ 0, %loop.t2 ], [ %h.next, %loop.h ]
  %acc = phi double [ 0.0, %loop.t2 ], [ %acc.next, %loop.h ]
  %q.idx = add nuw nsw i64 %t.h, %h
  %q.ptr = getelementptr inbounds double, ptr %q, i64 %q.idx
  %q.val = load double, ptr %q.ptr, align 8
  %k.idx = add nuw nsw i64 %t2.h, %h
  %k.ptr = getelementptr inbounds double, ptr %k, i64 %k.idx
  %k.val = load double, ptr %k.ptr, align 8
  %prod = fmul double %q.val, %k.val
  %acc.next = fadd double %acc, %prod
  %h.next = add nuw nsw i64 %h, 1
  %h.cmp = icmp eq i64 %h.next, 3
  br i1 %h.cmp, label %loop.t2.latch, label %loop.h

loop.t2.latch:
  %out.idx = add nuw nsw i64 %t.T, %t2
  %out.ptr = getelementptr inbounds double, ptr %out, i64 %out.idx
  store double %acc.next, ptr %out.ptr, align 8
  %t2.next = add nuw nsw i64 %t2, 1
  %t2.cmp = icmp eq i64 %t2.next, %t.lim
  br i1 %t2.cmp, label %loop.t.latch, label %loop.t2

loop.t.latch:
  %t.next = add nuw nsw i64 %t, 1
  %t.cmp = icmp eq i64 %t.next, 4
  br i1 %t.cmp, label %exit, label %loop.t

exit:
  ret void
}

; 3. Split-mode matmul where iteration o == 0 conditionally skips loading inp:
;    BT=3, OC=4, C=2 -> inp omits o (48 bytes, not 192), weight omits bt (64 bytes, not 192).
;    Unconditional cache store on o >= 1 ensures valid cached values even when o == 0 skips.
define void @skip_iter0(ptr noalias %out, ptr noalias %inp, ptr noalias %weight) {
entry:
  br label %loop.bt

loop.bt:
  %bt = phi i64 [ 0, %entry ], [ %bt.next, %loop.bt.latch ]
  %bt.c = mul nuw nsw i64 %bt, 2
  %bt.oc = mul nuw nsw i64 %bt, 4
  br label %loop.o

loop.o:
  %o = phi i64 [ 0, %loop.bt ], [ %o.next, %loop.o.latch ]
  %o.c = mul nuw nsw i64 %o, 2
  %skip = icmp eq i64 %o, 0
  br i1 %skip, label %loop.o.skip, label %loop.i

loop.o.skip:
  %out.skip.idx = add nuw nsw i64 %bt.oc, %o
  %out.skip.ptr = getelementptr inbounds double, ptr %out, i64 %out.skip.idx
  store double 0.0, ptr %out.skip.ptr, align 8
  br label %loop.o.latch

loop.i:
  %i = phi i64 [ 0, %loop.o ], [ %i.next, %loop.i ]
  %acc = phi double [ 0.0, %loop.o ], [ %acc.next, %loop.i ]
  %inp.idx = add nuw nsw i64 %bt.c, %i
  %inp.ptr = getelementptr inbounds double, ptr %inp, i64 %inp.idx
  %inp.val = load double, ptr %inp.ptr, align 8
  %w.idx = add nuw nsw i64 %o.c, %i
  %w.ptr = getelementptr inbounds double, ptr %weight, i64 %w.idx
  %w.val = load double, ptr %w.ptr, align 8
  %prod = fmul double %inp.val, %w.val
  %acc.next = fadd double %acc, %prod
  %i.next = add nuw nsw i64 %i, 1
  %i.cmp = icmp eq i64 %i.next, 2
  br i1 %i.cmp, label %loop.o.store, label %loop.i

loop.o.store:
  %out.idx = add nuw nsw i64 %bt.oc, %o
  %out.ptr = getelementptr inbounds double, ptr %out, i64 %out.idx
  store double %acc.next, ptr %out.ptr, align 8
  br label %loop.o.latch

loop.o.latch:
  %o.next = add nuw nsw i64 %o, 1
  %o.cmp = icmp eq i64 %o.next, 4
  br i1 %o.cmp, label %loop.bt.latch, label %loop.o

loop.bt.latch:
  %bt.next = add nuw nsw i64 %bt, 1
  %bt.cmp = icmp eq i64 %bt.next, 3
  br i1 %bt.cmp, label %exit, label %loop.bt

exit:
  ret void
}

; 4. Negative soundness test (cross-iteration memory mutation):
;    A[i] address is invariant in o, but A[o & 1] is mutated in loop.o -> MUST NOT omit o.
;    Cache must stay full 4*3*8 = 96 bytes (not 24).
define void @cross_iter_alias(ptr noalias %out, ptr noalias %A) {
entry:
  br label %loop.o

loop.o:
  %o = phi i64 [ 0, %entry ], [ %o.next, %loop.o.latch ]
  %acc.o = phi double [ 0.0, %entry ], [ %acc.next, %loop.o.latch ]
  br label %loop.i

loop.i:
  %i = phi i64 [ 0, %loop.o ], [ %i.next, %loop.i ]
  %acc = phi double [ %acc.o, %loop.o ], [ %acc.next, %loop.i ]
  %a.ptr = getelementptr inbounds double, ptr %A, i64 %i
  %v = load double, ptr %a.ptr, align 8
  %sq = fmul double %v, %v
  %acc.next = fadd double %acc, %sq
  %i.next = add nuw nsw i64 %i, 1
  %i.cmp = icmp eq i64 %i.next, 3
  br i1 %i.cmp, label %loop.o.latch, label %loop.i

loop.o.latch:
  %mod = and i64 %o, 1
  %am.ptr = getelementptr inbounds double, ptr %A, i64 %mod
  %old = load double, ptr %am.ptr, align 8
  %newv = fadd double %old, 5.000000e-01
  store double %newv, ptr %am.ptr, align 8
  %o.next = add nuw nsw i64 %o, 1
  %o.cmp = icmp eq i64 %o.next, 4
  br i1 %o.cmp, label %exit, label %loop.o

exit:
  store double %acc.next, ptr %out, align 8
  ret void
}

define i32 @test_matmul() {
entry:
  %out = alloca [15 x double], align 16
  %dout = alloca [15 x double], align 16
  %inp = alloca [12 x double], align 16
  %dinp = alloca [12 x double], align 16
  %w = alloca [20 x double], align 16
  %dw = alloca [20 x double], align 16
  %inp.orig = alloca [12 x double], align 16
  %w.orig = alloca [20 x double], align 16
  br label %init.inp

init.inp:
  %i0 = phi i64 [ 0, %entry ], [ %i0.next, %init.inp ]
  %i0.d = sitofp i64 %i0 to double
  %v0 = fadd double %i0.d, 1.0
  %p.inp = getelementptr inbounds double, ptr %inp, i64 %i0
  store double %v0, ptr %p.inp, align 8
  %p.inp.o = getelementptr inbounds double, ptr %inp.orig, i64 %i0
  store double %v0, ptr %p.inp.o, align 8
  %p.dinp = getelementptr inbounds double, ptr %dinp, i64 %i0
  store double 0.0, ptr %p.dinp, align 8
  %i0.next = add nuw nsw i64 %i0, 1
  %c0 = icmp eq i64 %i0.next, 12
  br i1 %c0, label %init.w, label %init.inp

init.w:
  %i1 = phi i64 [ 0, %init.inp ], [ %i1.next, %init.w ]
  %i1.d = sitofp i64 %i1 to double
  %v1 = fmul double %i1.d, 2.500000e-01
  %v1b = fadd double %v1, 5.000000e-01
  %p.w = getelementptr inbounds double, ptr %w, i64 %i1
  store double %v1b, ptr %p.w, align 8
  %p.w.o = getelementptr inbounds double, ptr %w.orig, i64 %i1
  store double %v1b, ptr %p.w.o, align 8
  %p.dw = getelementptr inbounds double, ptr %dw, i64 %i1
  store double 0.0, ptr %p.dw, align 8
  %i1.next = add nuw nsw i64 %i1, 1
  %c1 = icmp eq i64 %i1.next, 20
  br i1 %c1, label %init.out, label %init.w

init.out:
  %i2 = phi i64 [ 0, %init.w ], [ %i2.next, %init.out ]
  %i2.d = sitofp i64 %i2 to double
  %v2 = fadd double %i2.d, 1.0
  %p.out = getelementptr inbounds double, ptr %out, i64 %i2
  store double 0.0, ptr %p.out, align 8
  %p.dout = getelementptr inbounds double, ptr %dout, i64 %i2
  store double %v2, ptr %p.dout, align 8
  %i2.next = add nuw nsw i64 %i2, 1
  %c2 = icmp eq i64 %i2.next, 15
  br i1 %c2, label %run.split, label %init.out

run.split:
  %tape = call ptr (ptr, ...) @__enzyme_augmentfwd(ptr @matmul_split, metadata !"enzyme_dup", ptr %out, ptr %dout, metadata !"enzyme_dup", ptr %inp, ptr %dinp, metadata !"enzyme_dup", ptr %w, ptr %dw)
  ; Overwrite primal inp and w before reverse to prove cached values are used.
  br label %clobber.inp

clobber.inp:
  %ci = phi i64 [ 0, %run.split ], [ %ci.next, %clobber.inp ]
  %pci = getelementptr inbounds double, ptr %inp, i64 %ci
  store double -9.990000e+02, ptr %pci, align 8
  %ci.next = add nuw nsw i64 %ci, 1
  %cc = icmp eq i64 %ci.next, 12
  br i1 %cc, label %clobber.w, label %clobber.inp

clobber.w:
  %cw = phi i64 [ 0, %clobber.inp ], [ %cw.next, %clobber.w ]
  %pcw = getelementptr inbounds double, ptr %w, i64 %cw
  store double -9.990000e+02, ptr %pcw, align 8
  %cw.next = add nuw nsw i64 %cw, 1
  %ccw = icmp eq i64 %cw.next, 20
  br i1 %ccw, label %do.rev, label %clobber.w

do.rev:
  call void (ptr, ...) @__enzyme_reverse(ptr @matmul_split, metadata !"enzyme_dup", ptr %out, ptr %dout, metadata !"enzyme_dup", ptr %inp, ptr %dinp, metadata !"enzyme_dup", ptr %w, ptr %dw, ptr %tape)
  br label %chk.dinp.bt

chk.dinp.bt:
  %bt = phi i64 [ 0, %do.rev ], [ %bt.n, %chk.dinp.bt.l ]
  br label %chk.dinp.i

chk.dinp.i:
  %ii = phi i64 [ 0, %chk.dinp.bt ], [ %ii.n, %chk.dinp.i.ok ]
  br label %chk.dinp.o

chk.dinp.o:
  %oo = phi i64 [ 0, %chk.dinp.i ], [ %oo.n, %chk.dinp.o ]
  %exp = phi double [ 0.0, %chk.dinp.i ], [ %exp.n, %chk.dinp.o ]
  %bt5 = mul nuw nsw i64 %bt, 5
  %do.idx = add nuw nsw i64 %bt5, %oo
  %do.d = sitofp i64 %do.idx to double
  %do.v = fadd double %do.d, 1.0
  %oo4 = mul nuw nsw i64 %oo, 4
  %wo.idx = add nuw nsw i64 %oo4, %ii
  %wo.p = getelementptr inbounds double, ptr %w.orig, i64 %wo.idx
  %wo.v = load double, ptr %wo.p, align 8
  %term = fmul double %do.v, %wo.v
  %exp.n = fadd double %exp, %term
  %oo.n = add nuw nsw i64 %oo, 1
  %oo.c = icmp eq i64 %oo.n, 5
  br i1 %oo.c, label %chk.dinp.i.l, label %chk.dinp.o

chk.dinp.i.l:
  %bt4 = mul nuw nsw i64 %bt, 4
  %di.idx = add nuw nsw i64 %bt4, %ii
  %di.p = getelementptr inbounds double, ptr %dinp, i64 %di.idx
  %di.v = load double, ptr %di.p, align 8
  %diff = fsub double %di.v, %exp.n
  %adiff = call double @llvm.fabs.f64(double %diff)
  %bad = fcmp ogt double %adiff, 1.000000e-09
  br i1 %bad, label %fail1, label %chk.dinp.i.ok

chk.dinp.i.ok:
  %ii.n = add nuw nsw i64 %ii, 1
  %ii.c = icmp eq i64 %ii.n, 4
  br i1 %ii.c, label %chk.dinp.bt.l, label %chk.dinp.i

chk.dinp.bt.l:
  %bt.n = add nuw nsw i64 %bt, 1
  %bt.c = icmp eq i64 %bt.n, 3
  br i1 %bt.c, label %chk.dw.o, label %chk.dinp.bt

chk.dw.o:
  %wo = phi i64 [ 0, %chk.dinp.bt.l ], [ %wo.n, %chk.dw.o.l ]
  br label %chk.dw.i

chk.dw.i:
  %wi = phi i64 [ 0, %chk.dw.o ], [ %wi.n, %chk.dw.i.ok ]
  br label %chk.dw.bt

chk.dw.bt:
  %wbt = phi i64 [ 0, %chk.dw.i ], [ %wbt.n, %chk.dw.bt ]
  %wexp = phi double [ 0.0, %chk.dw.i ], [ %wexp.n, %chk.dw.bt ]
  %wbt5 = mul nuw nsw i64 %wbt, 5
  %wdo.idx = add nuw nsw i64 %wbt5, %wo
  %wdo.d = sitofp i64 %wdo.idx to double
  %wdo.v = fadd double %wdo.d, 1.0
  %wbt4 = mul nuw nsw i64 %wbt, 4
  %io.idx = add nuw nsw i64 %wbt4, %wi
  %io.p = getelementptr inbounds double, ptr %inp.orig, i64 %io.idx
  %io.v = load double, ptr %io.p, align 8
  %wterm = fmul double %wdo.v, %io.v
  %wexp.n = fadd double %wexp, %wterm
  %wbt.n = add nuw nsw i64 %wbt, 1
  %wbt.c = icmp eq i64 %wbt.n, 3
  br i1 %wbt.c, label %chk.dw.i.l, label %chk.dw.bt

chk.dw.i.l:
  %wo4 = mul nuw nsw i64 %wo, 4
  %dw.idx = add nuw nsw i64 %wo4, %wi
  %dw.p = getelementptr inbounds double, ptr %dw, i64 %dw.idx
  %dw.v = load double, ptr %dw.p, align 8
  %wdiff = fsub double %dw.v, %wexp.n
  %wadiff = call double @llvm.fabs.f64(double %wdiff)
  %wbad = fcmp ogt double %wadiff, 1.000000e-09
  br i1 %wbad, label %fail2, label %chk.dw.i.ok

chk.dw.i.ok:
  %wi.n = add nuw nsw i64 %wi, 1
  %wi.c = icmp eq i64 %wi.n, 4
  br i1 %wi.c, label %chk.dw.o.l, label %chk.dw.i

chk.dw.o.l:
  %wo.n = add nuw nsw i64 %wo, 1
  %wo.c = icmp eq i64 %wo.n, 5
  br i1 %wo.c, label %test.fwdsplit, label %chk.dw.o

test.fwdsplit:
  ; Also verify ForwardModeSplit unpacks the pruned tape cleanly.
  %fwd.dout = alloca [15 x double], align 16
  %fwd.dinp = alloca [12 x double], align 16
  %fwd.dw = alloca [20 x double], align 16
  br label %init.fwd.inp

init.fwd.inp:
  %fi0 = phi i64 [ 0, %test.fwdsplit ], [ %fi0.n, %init.fwd.inp ]
  %p.io = getelementptr inbounds double, ptr %inp.orig, i64 %fi0
  %vio = load double, ptr %p.io, align 8
  %p.i1 = getelementptr inbounds double, ptr %inp, i64 %fi0
  store double %vio, ptr %p.i1, align 8
  %p.fi = getelementptr inbounds double, ptr %fwd.dinp, i64 %fi0
  store double 1.0, ptr %p.fi, align 8
  %fi0.n = add nuw nsw i64 %fi0, 1
  %fc0 = icmp eq i64 %fi0.n, 12
  br i1 %fc0, label %init.fwd.w, label %init.fwd.inp

init.fwd.w:
  %fi1 = phi i64 [ 0, %init.fwd.inp ], [ %fi1.n, %init.fwd.w ]
  %p.wo = getelementptr inbounds double, ptr %w.orig, i64 %fi1
  %vwo = load double, ptr %p.wo, align 8
  %p.w1 = getelementptr inbounds double, ptr %w, i64 %fi1
  store double %vwo, ptr %p.w1, align 8
  %p.fw = getelementptr inbounds double, ptr %fwd.dw, i64 %fi1
  store double 2.0, ptr %p.fw, align 8
  %fi1.n = add nuw nsw i64 %fi1, 1
  %fc1 = icmp eq i64 %fi1.n, 20
  br i1 %fc1, label %run.fwdsplit, label %init.fwd.w

run.fwdsplit:
  %tape2 = call ptr (ptr, ...) @__enzyme_augmentfwd(ptr @matmul_split, metadata !"enzyme_dup", ptr %out, ptr %fwd.dout, metadata !"enzyme_dup", ptr %inp, ptr %fwd.dinp, metadata !"enzyme_dup", ptr %w, ptr %fwd.dw)
  br label %clobber.fwd

clobber.fwd:
  %cf = phi i64 [ 0, %run.fwdsplit ], [ %cf.n, %clobber.fwd ]
  %pcf1 = getelementptr inbounds double, ptr %inp, i64 %cf
  store double -8.880000e+02, ptr %pcf1, align 8
  %cf.n = add nuw nsw i64 %cf, 1
  %cfc = icmp eq i64 %cf.n, 12
  br i1 %cfc, label %do.fwdsplit, label %clobber.fwd

do.fwdsplit:
  call void (ptr, ...) @__enzyme_fwdsplit(ptr @matmul_split, metadata !"enzyme_nofree", metadata !"enzyme_dup", ptr %out, ptr %fwd.dout, metadata !"enzyme_dup", ptr %inp, ptr %fwd.dinp, metadata !"enzyme_dup", ptr %w, ptr %fwd.dw, ptr %tape2)
  ; Check fwd.dout[bt=2, o=3] = sum_{i=0..3} (1.0 * w[3,i] + inp[2,i] * 2.0)
  br label %chk.fwd

chk.fwd:
  %fi = phi i64 [ 0, %do.fwdsplit ], [ %fi.n, %chk.fwd ]
  %fexp = phi double [ 0.0, %do.fwdsplit ], [ %fexp.n, %chk.fwd ]
  %f.iidx = add nuw nsw i64 8, %fi
  %f.ip = getelementptr inbounds double, ptr %inp.orig, i64 %f.iidx
  %f.iv = load double, ptr %f.ip, align 8
  %f.widx = add nuw nsw i64 12, %fi
  %f.wp = getelementptr inbounds double, ptr %w.orig, i64 %f.widx
  %f.wv = load double, ptr %f.wp, align 8
  %f.t1 = fmul double %f.iv, 2.0
  %f.t2 = fadd double %f.wv, %f.t1
  %fexp.n = fadd double %fexp, %f.t2
  %fi.n = add nuw nsw i64 %fi, 1
  %fc2 = icmp eq i64 %fi.n, 4
  br i1 %fc2, label %chk.fwd.done, label %chk.fwd

chk.fwd.done:
  %fd.p = getelementptr inbounds double, ptr %fwd.dout, i64 13
  %fd.v = load double, ptr %fd.p, align 8
  %fdiff = fsub double %fd.v, %fexp.n
  %fadiff = call double @llvm.fabs.f64(double %fdiff)
  %fbad = fcmp ogt double %fadiff, 1.000000e-09
  br i1 %fbad, label %fail3, label %ok

fail1:
  ret i32 1
fail2:
  ret i32 2
fail3:
  ret i32 3
ok:
  ret i32 0
}

define i32 @test_triangular_attn() {
entry:
  %out = alloca [16 x double], align 16
  %dout = alloca [16 x double], align 16
  %q = alloca [12 x double], align 16
  %dq = alloca [12 x double], align 16
  %k = alloca [12 x double], align 16
  %dk = alloca [12 x double], align 16
  %q.orig = alloca [12 x double], align 16
  %k.orig = alloca [12 x double], align 16
  br label %init.qk

init.qk:
  %i = phi i64 [ 0, %entry ], [ %i.n, %init.qk ]
  %id = sitofp i64 %i to double
  %qv = fadd double %id, 1.0
  %kv = fmul double %id, 5.000000e-01
  %kv2 = fadd double %kv, 2.500000e-01
  %pq = getelementptr inbounds double, ptr %q, i64 %i
  store double %qv, ptr %pq, align 8
  %pqo = getelementptr inbounds double, ptr %q.orig, i64 %i
  store double %qv, ptr %pqo, align 8
  %pdq = getelementptr inbounds double, ptr %dq, i64 %i
  store double 0.0, ptr %pdq, align 8
  %pk = getelementptr inbounds double, ptr %k, i64 %i
  store double %kv2, ptr %pk, align 8
  %pko = getelementptr inbounds double, ptr %k.orig, i64 %i
  store double %kv2, ptr %pko, align 8
  %pdk = getelementptr inbounds double, ptr %dk, i64 %i
  store double 0.0, ptr %pdk, align 8
  %i.n = add nuw nsw i64 %i, 1
  %ic = icmp eq i64 %i.n, 12
  br i1 %ic, label %init.out, label %init.qk

init.out:
  %j = phi i64 [ 0, %init.qk ], [ %j.n, %init.out ]
  %jd = sitofp i64 %j to double
  %ov = fadd double %jd, 1.0
  %po = getelementptr inbounds double, ptr %out, i64 %j
  store double 0.0, ptr %po, align 8
  %pdo = getelementptr inbounds double, ptr %dout, i64 %j
  store double %ov, ptr %pdo, align 8
  %j.n = add nuw nsw i64 %j, 1
  %jc = icmp eq i64 %j.n, 16
  br i1 %jc, label %run, label %init.out

run:
  %tape = call ptr (ptr, ...) @__enzyme_augmentfwd(ptr @triangular_attn_split, metadata !"enzyme_dup", ptr %out, ptr %dout, metadata !"enzyme_dup", ptr %q, ptr %dq, metadata !"enzyme_dup", ptr %k, ptr %dk)
  br label %clobber

clobber:
  %c = phi i64 [ 0, %run ], [ %c.n, %clobber ]
  %cpq = getelementptr inbounds double, ptr %q, i64 %c
  store double -7.770000e+02, ptr %cpq, align 8
  %cpk = getelementptr inbounds double, ptr %k, i64 %c
  store double -7.770000e+02, ptr %cpk, align 8
  %c.n = add nuw nsw i64 %c, 1
  %cc = icmp eq i64 %c.n, 12
  br i1 %cc, label %rev, label %clobber

rev:
  call void (ptr, ...) @__enzyme_reverse(ptr @triangular_attn_split, metadata !"enzyme_dup", ptr %out, ptr %dout, metadata !"enzyme_dup", ptr %q, ptr %dq, metadata !"enzyme_dup", ptr %k, ptr %dk, ptr %tape)
  br label %chk.t

chk.t:
  %t = phi i64 [ 0, %rev ], [ %t.n, %chk.t.l ]
  %tlim = add nuw nsw i64 %t, 1
  br label %chk.h

chk.h:
  %h = phi i64 [ 0, %chk.t ], [ %h.n, %chk.h.ok ]
  br label %chk.t2

chk.t2:
  %t2 = phi i64 [ 0, %chk.h ], [ %t2.n, %chk.t2 ]
  %qexp = phi double [ 0.0, %chk.h ], [ %qexp.n, %chk.t2 ]
  %t4 = mul nuw nsw i64 %t, 4
  %oidx = add nuw nsw i64 %t4, %t2
  %od = sitofp i64 %oidx to double
  %ov2 = fadd double %od, 1.0
  %t23 = mul nuw nsw i64 %t2, 3
  %kidx = add nuw nsw i64 %t23, %h
  %kp = getelementptr inbounds double, ptr %k.orig, i64 %kidx
  %kv3 = load double, ptr %kp, align 8
  %qt = fmul double %ov2, %kv3
  %qexp.n = fadd double %qexp, %qt
  %t2.n = add nuw nsw i64 %t2, 1
  %t2c = icmp eq i64 %t2.n, %tlim
  br i1 %t2c, label %chk.h.l, label %chk.t2

chk.h.l:
  %t3 = mul nuw nsw i64 %t, 3
  %qidx = add nuw nsw i64 %t3, %h
  %dqp = getelementptr inbounds double, ptr %dq, i64 %qidx
  %dqv = load double, ptr %dqp, align 8
  %qd = fsub double %dqv, %qexp.n
  %qa = call double @llvm.fabs.f64(double %qd)
  %qbad = fcmp ogt double %qa, 1.000000e-09
  br i1 %qbad, label %fail1, label %chk.h.ok

chk.h.ok:
  %h.n = add nuw nsw i64 %h, 1
  %hc = icmp eq i64 %h.n, 3
  br i1 %hc, label %chk.t.l, label %chk.h

chk.t.l:
  %t.n = add nuw nsw i64 %t, 1
  %tc = icmp eq i64 %t.n, 4
  br i1 %tc, label %chk.dk.t2, label %chk.t

chk.dk.t2:
  %kt2 = phi i64 [ 0, %chk.t.l ], [ %kt2.n, %chk.dk.t2.l ]
  br label %chk.dk.h

chk.dk.h:
  %kh = phi i64 [ 0, %chk.dk.t2 ], [ %kh.n, %chk.dk.h.ok ]
  br label %chk.dk.t

chk.dk.t:
  %kt = phi i64 [ %kt2, %chk.dk.h ], [ %kt.n, %chk.dk.t ]
  %kexp = phi double [ 0.0, %chk.dk.h ], [ %kexp.n, %chk.dk.t ]
  %kt4 = mul nuw nsw i64 %kt, 4
  %koidx = add nuw nsw i64 %kt4, %kt2
  %kod = sitofp i64 %koidx to double
  %kov = fadd double %kod, 1.0
  %kt3 = mul nuw nsw i64 %kt, 3
  %kqidx = add nuw nsw i64 %kt3, %kh
  %kqp = getelementptr inbounds double, ptr %q.orig, i64 %kqidx
  %kqv = load double, ptr %kqp, align 8
  %kterm = fmul double %kov, %kqv
  %kexp.n = fadd double %kexp, %kterm
  %kt.n = add nuw nsw i64 %kt, 1
  %ktc = icmp eq i64 %kt.n, 4
  br i1 %ktc, label %chk.dk.h.l, label %chk.dk.t

chk.dk.h.l:
  %kt23 = mul nuw nsw i64 %kt2, 3
  %dkidx = add nuw nsw i64 %kt23, %kh
  %dkp = getelementptr inbounds double, ptr %dk, i64 %dkidx
  %dkv = load double, ptr %dkp, align 8
  %kd = fsub double %dkv, %kexp.n
  %ka = call double @llvm.fabs.f64(double %kd)
  %kbad = fcmp ogt double %ka, 1.000000e-09
  br i1 %kbad, label %fail2, label %chk.dk.h.ok

chk.dk.h.ok:
  %kh.n = add nuw nsw i64 %kh, 1
  %khc = icmp eq i64 %kh.n, 3
  br i1 %khc, label %chk.dk.t2.l, label %chk.dk.h

chk.dk.t2.l:
  %kt2.n = add nuw nsw i64 %kt2, 1
  %kt2c = icmp eq i64 %kt2.n, 4
  br i1 %kt2c, label %ok, label %chk.dk.t2

fail1:
  ret i32 4
fail2:
  ret i32 5
ok:
  ret i32 0
}

define i32 @test_skip_and_alias() {
entry:
  %out = alloca [12 x double], align 16
  %dout = alloca [12 x double], align 16
  %inp = alloca [6 x double], align 16
  %dinp = alloca [6 x double], align 16
  %w = alloca [8 x double], align 16
  %dw = alloca [8 x double], align 16
  %inp.orig = alloca [6 x double], align 16
  %w.orig = alloca [8 x double], align 16
  br label %init1

init1:
  %i = phi i64 [ 0, %entry ], [ %i.n, %init1 ]
  %id = sitofp i64 %i to double
  %iv = fadd double %id, 1.0
  %pi = getelementptr inbounds double, ptr %inp, i64 %i
  store double %iv, ptr %pi, align 8
  %pio = getelementptr inbounds double, ptr %inp.orig, i64 %i
  store double %iv, ptr %pio, align 8
  %pdi = getelementptr inbounds double, ptr %dinp, i64 %i
  store double 0.0, ptr %pdi, align 8
  %i.n = add nuw nsw i64 %i, 1
  %ic = icmp eq i64 %i.n, 6
  br i1 %ic, label %init2, label %init1

init2:
  %j = phi i64 [ 0, %init1 ], [ %j.n, %init2 ]
  %jd = sitofp i64 %j to double
  %jv = fadd double %jd, 2.0
  %pw = getelementptr inbounds double, ptr %w, i64 %j
  store double %jv, ptr %pw, align 8
  %pwo = getelementptr inbounds double, ptr %w.orig, i64 %j
  store double %jv, ptr %pwo, align 8
  %pdw = getelementptr inbounds double, ptr %dw, i64 %j
  store double 0.0, ptr %pdw, align 8
  %j.n = add nuw nsw i64 %j, 1
  %jc = icmp eq i64 %j.n, 8
  br i1 %jc, label %init3, label %init2

init3:
  %k = phi i64 [ 0, %init2 ], [ %k.n, %init3 ]
  %kd = sitofp i64 %k to double
  %kv = fadd double %kd, 1.0
  %po = getelementptr inbounds double, ptr %out, i64 %k
  store double 0.0, ptr %po, align 8
  %pdo = getelementptr inbounds double, ptr %dout, i64 %k
  store double %kv, ptr %pdo, align 8
  %k.n = add nuw nsw i64 %k, 1
  %kc = icmp eq i64 %k.n, 12
  br i1 %kc, label %run.skip, label %init3

run.skip:
  %stape = call ptr (ptr, ...) @__enzyme_augmentfwd(ptr @skip_iter0, metadata !"enzyme_dup", ptr %out, ptr %dout, metadata !"enzyme_dup", ptr %inp, ptr %dinp, metadata !"enzyme_dup", ptr %w, ptr %dw)
  br label %clobber.s

clobber.s:
  %cs = phi i64 [ 0, %run.skip ], [ %cs.n, %clobber.s ]
  %pcs1 = getelementptr inbounds double, ptr %inp, i64 %cs
  store double -6.660000e+02, ptr %pcs1, align 8
  %pcs2 = getelementptr inbounds double, ptr %w, i64 %cs
  store double -6.660000e+02, ptr %pcs2, align 8
  %cs.n = add nuw nsw i64 %cs, 1
  %csc = icmp eq i64 %cs.n, 6
  br i1 %csc, label %rev.skip, label %clobber.s

rev.skip:
  call void (ptr, ...) @__enzyme_reverse(ptr @skip_iter0, metadata !"enzyme_dup", ptr %out, ptr %dout, metadata !"enzyme_dup", ptr %inp, ptr %dinp, metadata !"enzyme_dup", ptr %w, ptr %dw, ptr %stape)
  ; Check dinp[bt=1, i=0] = sum_{o=1..3} dout[1*4 + o] * w[o*2 + 0]
  ;   = 6*4 + 7*6 + 8*8 = 24 + 42 + 64 = 130.0
  %di.p = getelementptr inbounds double, ptr %dinp, i64 2
  %di.v = load double, ptr %di.p, align 8
  %di.d = fsub double %di.v, 1.300000e+02
  %di.a = call double @llvm.fabs.f64(double %di.d)
  %di.bad = fcmp ogt double %di.a, 1.000000e-09
  br i1 %di.bad, label %fail1, label %run.alias

run.alias:
  %aout = alloca double, align 8
  %adout = alloca double, align 8
  %A = alloca [4 x double], align 16
  %dA = alloca [4 x double], align 16
  %A.plus = alloca [4 x double], align 16
  %A.minus = alloca [4 x double], align 16
  %out.plus = alloca double, align 8
  %out.minus = alloca double, align 8
  store double 0.0, ptr %aout, align 8
  store double 1.0, ptr %adout, align 8
  %pA0 = getelementptr inbounds double, ptr %A, i64 0
  %pA1 = getelementptr inbounds double, ptr %A, i64 1
  %pA2 = getelementptr inbounds double, ptr %A, i64 2
  %pA3 = getelementptr inbounds double, ptr %A, i64 3
  store double 1.0, ptr %pA0, align 8
  store double 2.0, ptr %pA1, align 8
  store double 3.0, ptr %pA2, align 8
  store double 4.0, ptr %pA3, align 8
  %pdA0 = getelementptr inbounds double, ptr %dA, i64 0
  %pdA1 = getelementptr inbounds double, ptr %dA, i64 1
  %pdA2 = getelementptr inbounds double, ptr %dA, i64 2
  %pdA3 = getelementptr inbounds double, ptr %dA, i64 3
  store double 0.0, ptr %pdA0, align 8
  store double 0.0, ptr %pdA1, align 8
  store double 0.0, ptr %pdA2, align 8
  store double 0.0, ptr %pdA3, align 8
  call void (ptr, ...) @__enzyme_autodiff(ptr @cross_iter_alias, metadata !"enzyme_dup", ptr %aout, ptr %adout, metadata !"enzyme_dup", ptr %A, ptr %dA)
  ; Verify dA[0] and dA[1] against central finite differences (eps = 1e-6).
  br label %fd.loop

fd.loop:
  %idx = phi i64 [ 0, %run.alias ], [ %idx.n, %fd.ok ]
  %pp0 = getelementptr inbounds double, ptr %A.plus, i64 0
  %pp1 = getelementptr inbounds double, ptr %A.plus, i64 1
  %pp2 = getelementptr inbounds double, ptr %A.plus, i64 2
  %pp3 = getelementptr inbounds double, ptr %A.plus, i64 3
  store double 1.0, ptr %pp0, align 8
  store double 2.0, ptr %pp1, align 8
  store double 3.0, ptr %pp2, align 8
  store double 4.0, ptr %pp3, align 8
  %pm0 = getelementptr inbounds double, ptr %A.minus, i64 0
  %pm1 = getelementptr inbounds double, ptr %A.minus, i64 1
  %pm2 = getelementptr inbounds double, ptr %A.minus, i64 2
  %pm3 = getelementptr inbounds double, ptr %A.minus, i64 3
  store double 1.0, ptr %pm0, align 8
  store double 2.0, ptr %pm1, align 8
  store double 3.0, ptr %pm2, align 8
  store double 4.0, ptr %pm3, align 8
  %ppk = getelementptr inbounds double, ptr %A.plus, i64 %idx
  %oldp = load double, ptr %ppk, align 8
  %newp = fadd double %oldp, 1.000000e-06
  store double %newp, ptr %ppk, align 8
  %pmk = getelementptr inbounds double, ptr %A.minus, i64 %idx
  %oldm = load double, ptr %pmk, align 8
  %newm = fsub double %oldm, 1.000000e-06
  store double %newm, ptr %pmk, align 8
  call void @cross_iter_alias(ptr %out.plus, ptr %A.plus)
  call void @cross_iter_alias(ptr %out.minus, ptr %A.minus)
  %vp = load double, ptr %out.plus, align 8
  %vm = load double, ptr %out.minus, align 8
  %vdiff = fsub double %vp, %vm
  %fd = fdiv double %vdiff, 2.000000e-06
  %dAk.p = getelementptr inbounds double, ptr %dA, i64 %idx
  %dAk = load double, ptr %dAk.p, align 8
  %err = fsub double %dAk, %fd
  %aerr = call double @llvm.fabs.f64(double %err)
  %abad = fcmp ogt double %aerr, 1.000000e-05
  br i1 %abad, label %fail2, label %fd.ok

fd.ok:
  %idx.n = add nuw nsw i64 %idx, 1
  %idxc = icmp eq i64 %idx.n, 3
  br i1 %idxc, label %ok, label %fd.loop

fail1:
  ret i32 6
fail2:
  ret i32 7
ok:
  ret i32 0
}

define i32 @main() {
entry:
  %r1 = call i32 @test_matmul()
  %c1 = icmp ne i32 %r1, 0
  br i1 %c1, label %ret1, label %next2

ret1:
  ret i32 %r1

next2:
  %r2 = call i32 @test_triangular_attn()
  %c2 = icmp ne i32 %r2, 0
  br i1 %c2, label %ret2, label %next3

ret2:
  ret i32 %r2

next3:
  %r3 = call i32 @test_skip_and_alias()
  ret i32 %r3
}

; CHECK-LABEL: define internal ptr @augmented_matmul_split(
; CHECK: entry:
; CHECK-NOT: @malloc(i64 480)
; CHECK-DAG: @malloc(i64 96)
; CHECK-DAG: @malloc(i64 160)
; CHECK-NOT: @malloc(i64 480)
; CHECK: ret ptr

; CHECK-LABEL: define internal void @diffematmul_split(
; CHECK-NOT: @malloc(
; CHECK: ret void

; CHECK-LABEL: define internal void @fwddiffematmul_split(
; CHECK-NOT: @malloc(
; CHECK: ret void

; CHECK-LABEL: define internal ptr @augmented_triangular_attn_split(
; CHECK: entry:
; CHECK-DAG: @malloc(i64 96)
; CHECK-DAG: @malloc(i64 32)
; CHECK-NOT: @malloc(i64 32)
; CHECK: loop.t:
; CHECK: @malloc(i64
; CHECK-NOT: @malloc(i64
; CHECK: ret ptr

; CHECK-LABEL: define internal ptr @augmented_skip_iter0(
; CHECK: entry:
; CHECK-NOT: @malloc(i64 192)
; CHECK-DAG: @malloc(i64 48)
; CHECK-DAG: @malloc(i64 64)
; CHECK-NOT: @malloc(i64 192)
; CHECK: ret ptr

; CHECK-LABEL: define internal void @diffecross_iter_alias(
; CHECK: entry:
; CHECK-NOT: @malloc(i64 24)
; CHECK: @malloc(i64 96)
; CHECK: ret void
