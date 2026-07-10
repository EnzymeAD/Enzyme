; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s

define internal double @mygamma(double %x) {
top:
  %cmp1 = fcmp ult double %x, 3.0
  br i1 %cmp1, label %loop2.preheader, label %loop1

loop1:
  %x.phi = phi double [ %x.sub, %loop1 ], [ %x, %top ]
  %z.phi = phi double [ %z.mul, %loop1 ], [ 1.0, %top ]
  %x.sub = fadd double %x.phi, -1.0
  %z.mul = fmul double %z.phi, %x.sub
  %cmp2 = fcmp ult double %x.sub, 3.0
  br i1 %cmp2, label %loop2.preheader, label %loop1

loop2.preheader:
  %z.lcssa = phi double [ 1.0, %top ], [ %z.mul, %loop1 ]
  %x.lcssa = phi double [ %x, %top ], [ %x.sub, %loop1 ]
  %cmp3 = fcmp uge double %x.lcssa, 2.0
  br i1 %cmp3, label %exit, label %loop2

loop2:
  %x2.phi = phi double [ %x.add, %loop2 ], [ %x.lcssa, %loop2.preheader ]
  %z2.phi = phi double [ %z.div, %loop2 ], [ %z.lcssa, %loop2.preheader ]
  %z.div = fdiv double %z2.phi, %x2.phi
  %x.add = fadd double %x2.phi, 1.0
  %cmp4 = fcmp uge double %x.add, 2.0
  br i1 %cmp4, label %exit, label %loop2

exit:
  %z.res = phi double [ %z.lcssa, %loop2.preheader ], [ %z.div, %loop2 ]
  %x.res = phi double [ %x.lcssa, %loop2.preheader ], [ %x.add, %loop2 ]
  %plus1 = fadd double %x.res, 1.0
  %res = fmul double %z.res, %plus1
  ret double %res
}

define double @target(double %a, double %b) {
entry:
  %g1 = call double @mygamma(double %a)
  %g2 = call double @mygamma(double %b)
  %sub = fsub double %g1, %g2
  %res = call double @llvm.log.f64(double %sub)
  ret double %res
}

declare double @llvm.log.f64(double) readnone

define void @dtarget(double %a, double %b) {
entry:
  call void @__enzyme_autodiff(ptr @target, double %a, double %b)
  ret void
}

declare void @__enzyme_autodiff(...)
