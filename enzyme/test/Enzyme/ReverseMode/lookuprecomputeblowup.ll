; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -enzyme-lookup-recompute-budget=10 -S | FileCheck %s

; A loop whose body is a chain of diamonds. The reverse pass cannot reuse the
; body's values in place -- they change every iteration and their blocks do not
; dominate the return -- so lookupM tries to recompute each merge phi.
; Rebuilding a phi unwraps it into fresh blocks, one per predecessor, and those
; siblings do not share lookup results, so every diamond multiplies the work.
;
; This is a compile time test: what matters is that it finishes at all. With
; the recompute bound this file differentiates in well under a second; with
; the bound removed from the source it was still running after five minutes,
; growing about 3x per diamond. The last RUN exercises the exhausted-budget
; fallback, where nearly every lookup takes the always-legal caching path.

declare double @__enzyme_autodiff(i8*, ...)

define double @f(double %x, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inext, %latch ]
  %acc = phi double [ %x, %entry ], [ %accnext, %latch ]
  %c0 = fcmp ogt double %acc, 1.000000e+00
  br i1 %c0, label %a0, label %b0

a0:
  %va0 = fmul double %acc, %acc
  br label %m0
b0:
  %vb0 = fmul double %acc, %x
  br label %m0
m0:
  %p0 = phi double [ %va0, %a0 ], [ %vb0, %b0 ]
  %c1 = fcmp ogt double %p0, 1.000000e+00
  br i1 %c1, label %a1, label %b1

a1:
  %va1 = fmul double %p0, %p0
  br label %m1
b1:
  %vb1 = fmul double %p0, %x
  br label %m1
m1:
  %p1 = phi double [ %va1, %a1 ], [ %vb1, %b1 ]
  %c2 = fcmp ogt double %p1, 1.000000e+00
  br i1 %c2, label %a2, label %b2

a2:
  %va2 = fmul double %p1, %p1
  br label %m2
b2:
  %vb2 = fmul double %p1, %x
  br label %m2
m2:
  %p2 = phi double [ %va2, %a2 ], [ %vb2, %b2 ]
  %c3 = fcmp ogt double %p2, 1.000000e+00
  br i1 %c3, label %a3, label %b3

a3:
  %va3 = fmul double %p2, %p2
  br label %m3
b3:
  %vb3 = fmul double %p2, %x
  br label %m3
m3:
  %p3 = phi double [ %va3, %a3 ], [ %vb3, %b3 ]
  %c4 = fcmp ogt double %p3, 1.000000e+00
  br i1 %c4, label %a4, label %b4

a4:
  %va4 = fmul double %p3, %p3
  br label %m4
b4:
  %vb4 = fmul double %p3, %x
  br label %m4
m4:
  %p4 = phi double [ %va4, %a4 ], [ %vb4, %b4 ]
  %c5 = fcmp ogt double %p4, 1.000000e+00
  br i1 %c5, label %a5, label %b5

a5:
  %va5 = fmul double %p4, %p4
  br label %m5
b5:
  %vb5 = fmul double %p4, %x
  br label %m5
m5:
  %p5 = phi double [ %va5, %a5 ], [ %vb5, %b5 ]
  %c6 = fcmp ogt double %p5, 1.000000e+00
  br i1 %c6, label %a6, label %b6

a6:
  %va6 = fmul double %p5, %p5
  br label %m6
b6:
  %vb6 = fmul double %p5, %x
  br label %m6
m6:
  %p6 = phi double [ %va6, %a6 ], [ %vb6, %b6 ]
  %c7 = fcmp ogt double %p6, 1.000000e+00
  br i1 %c7, label %a7, label %b7

a7:
  %va7 = fmul double %p6, %p6
  br label %m7
b7:
  %vb7 = fmul double %p6, %x
  br label %m7
m7:
  %p7 = phi double [ %va7, %a7 ], [ %vb7, %b7 ]
  %c8 = fcmp ogt double %p7, 1.000000e+00
  br i1 %c8, label %a8, label %b8

a8:
  %va8 = fmul double %p7, %p7
  br label %m8
b8:
  %vb8 = fmul double %p7, %x
  br label %m8
m8:
  %p8 = phi double [ %va8, %a8 ], [ %vb8, %b8 ]
  %c9 = fcmp ogt double %p8, 1.000000e+00
  br i1 %c9, label %a9, label %b9

a9:
  %va9 = fmul double %p8, %p8
  br label %m9
b9:
  %vb9 = fmul double %p8, %x
  br label %m9
m9:
  %p9 = phi double [ %va9, %a9 ], [ %vb9, %b9 ]
  %c10 = fcmp ogt double %p9, 1.000000e+00
  br i1 %c10, label %a10, label %b10

a10:
  %va10 = fmul double %p9, %p9
  br label %m10
b10:
  %vb10 = fmul double %p9, %x
  br label %m10
m10:
  %p10 = phi double [ %va10, %a10 ], [ %vb10, %b10 ]
  %c11 = fcmp ogt double %p10, 1.000000e+00
  br i1 %c11, label %a11, label %b11

a11:
  %va11 = fmul double %p10, %p10
  br label %m11
b11:
  %vb11 = fmul double %p10, %x
  br label %m11
m11:
  %p11 = phi double [ %va11, %a11 ], [ %vb11, %b11 ]
  %c12 = fcmp ogt double %p11, 1.000000e+00
  br i1 %c12, label %a12, label %b12

a12:
  %va12 = fmul double %p11, %p11
  br label %m12
b12:
  %vb12 = fmul double %p11, %x
  br label %m12
m12:
  %p12 = phi double [ %va12, %a12 ], [ %vb12, %b12 ]
  %c13 = fcmp ogt double %p12, 1.000000e+00
  br i1 %c13, label %a13, label %b13

a13:
  %va13 = fmul double %p12, %p12
  br label %m13
b13:
  %vb13 = fmul double %p12, %x
  br label %m13
m13:
  %p13 = phi double [ %va13, %a13 ], [ %vb13, %b13 ]
  %c14 = fcmp ogt double %p13, 1.000000e+00
  br i1 %c14, label %a14, label %b14

a14:
  %va14 = fmul double %p13, %p13
  br label %m14
b14:
  %vb14 = fmul double %p13, %x
  br label %m14
m14:
  %p14 = phi double [ %va14, %a14 ], [ %vb14, %b14 ]
  %c15 = fcmp ogt double %p14, 1.000000e+00
  br i1 %c15, label %a15, label %b15

a15:
  %va15 = fmul double %p14, %p14
  br label %m15
b15:
  %vb15 = fmul double %p14, %x
  br label %m15
m15:
  %p15 = phi double [ %va15, %a15 ], [ %vb15, %b15 ]
  %c16 = fcmp ogt double %p15, 1.000000e+00
  br i1 %c16, label %a16, label %b16

a16:
  %va16 = fmul double %p15, %p15
  br label %m16
b16:
  %vb16 = fmul double %p15, %x
  br label %m16
m16:
  %p16 = phi double [ %va16, %a16 ], [ %vb16, %b16 ]
  %c17 = fcmp ogt double %p16, 1.000000e+00
  br i1 %c17, label %a17, label %b17

a17:
  %va17 = fmul double %p16, %p16
  br label %m17
b17:
  %vb17 = fmul double %p16, %x
  br label %m17
m17:
  %p17 = phi double [ %va17, %a17 ], [ %vb17, %b17 ]
  %c18 = fcmp ogt double %p17, 1.000000e+00
  br i1 %c18, label %a18, label %b18

a18:
  %va18 = fmul double %p17, %p17
  br label %m18
b18:
  %vb18 = fmul double %p17, %x
  br label %m18
m18:
  %p18 = phi double [ %va18, %a18 ], [ %vb18, %b18 ]
  %c19 = fcmp ogt double %p18, 1.000000e+00
  br i1 %c19, label %a19, label %b19

a19:
  %va19 = fmul double %p18, %p18
  br label %m19
b19:
  %vb19 = fmul double %p18, %x
  br label %m19
m19:
  %p19 = phi double [ %va19, %a19 ], [ %vb19, %b19 ]
  br label %latch

latch:
  %accnext = fmul double %p19, %x
  %inext = add i64 %i, 1
  %cond = icmp slt i64 %inext, %n
  br i1 %cond, label %loop, label %exit

exit:
  ret double %acc
}

define double @df(double %x, i64 %n) {
  %r = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (double, i64)* @f to i8*), double %x, i64 %n)
  ret double %r
}

; CHECK: define internal { double } @diffef(double %x, i64 %n, double %differeturn)
