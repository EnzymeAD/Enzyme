; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s

define internal double @mygamma(double %x) {
top:
  br label %loop1

loop1:
  %x.phi = phi double [ %x.sub, %loop1 ], [ %x, %top ]
  %z.phi = phi double [ %z.mul, %loop1 ], [ 1.0, %top ]
  %x.sub = fadd double %x.phi, -1.0
  %z.mul = fmul double %z.phi, %x.sub
  %cmp2 = fcmp ult double %x.sub, 3.0
  br i1 %cmp2, label %loop2.preheader, label %loop1

loop2.preheader:
  %z.lcssa = phi double [ %z.mul, %loop1 ]
  %x.lcssa = phi double [ %x, %loop1 ]
  %cmp3 = fcmp uge double %x.lcssa, 2.0
  br label %loop2

loop2:
  %x2.phi = phi double [ 0.0, %loop2 ], [ %x.lcssa, %loop2.preheader ]
  %z2.phi = phi double [ %z.div, %loop2 ], [ %z.lcssa, %loop2.preheader ]
  %z.div = fdiv double %z2.phi, %x2.phi
  %cmp4 = fcmp uge double %x2.phi, 2.0
  br i1 %cmp4, label %exit, label %loop2

exit:
  %z.res = phi double [ %z.div, %loop2 ]
  ret double %z.res
}

define internal double @mygamma_load(double* %p) {
top:
  %x = load double, double* %p, align 8
  br label %loop2

loop2:
  %i2 = phi i64 [ %i2.next, %loop2.latch ], [ 0, %top ]
  %z2.phi = phi double [ %z.div, %loop2.latch ], [ %x, %top ]
  %z.div = fdiv double %z2.phi, 2.0
  %sq = fmul double %z.div, %z.div
  %cmp_z = fcmp uge double %z2.phi, 1.0
  br i1 %cmp_z, label %if.true, label %loop2.latch

if.true:
  store double %sq, double* %p, align 8
  br label %loop2.latch

loop2.latch:
  %i2.next = add nuw nsw i64 %i2, 1
  %cmp4 = icmp eq i64 %i2.next, 4
  br i1 %cmp4, label %exit, label %loop2

exit:
  %z.res = phi double [ %sq, %loop2.latch ]
  ret double %z.res
}

define void @dtarget(double %a, double %b) {
entry:
  %z = call i8* (...) @__enzyme_virtualreverse(double (double)* @mygamma)
  %z2 = call i8* (...) @__enzyme_virtualreverse(double (double*)* @mygamma_load)
  ret void
}

declare i8* @__enzyme_virtualreverse(...)

; CHECK: define internal { double } @diffemygamma(double %x, double %differeturn, {{i8\*|ptr}} %tapeArg)

; CHECK: invertloop2:
; CHECK:   %[[divadj:.+]] = fdiv fast double %"z.div'de.0", %{{.+}}
; CHECK-NEXT:   %[[muladj:.+]] = fmul fast double %z.mul, %[[divadj]]

; CHECK: define internal void @diffemygamma_load({{double\*|ptr}} nocapture %p, {{double\*|ptr}} nocapture %"p'", double %differeturn, {{i8\*|ptr}} %tapeArg)
