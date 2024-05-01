; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

@stdout = external local_unnamed_addr global ptr, align 8
@.str = private unnamed_addr constant [18 x i8] c" whatever ( %g )\0A\00", align 1

; Function Attrs: noinline nounwind optnone uwtable
define internal double @dotabs(double %ld) {
  %r = call i32 (ptr, ptr, ...) @fprintf(ptr @stdout, ptr noundef @.str, double noundef %ld)
  ret double %ld
}

; Function Attrs: nofree nounwind
declare noundef i32 @fprintf(ptr nocapture noundef, ptr nocapture noundef readonly, ...)

; Function Attrs: noinline nounwind optnone uwtable
define void @f(ptr %x, ptr %dx) {
  %r = call double (...) @__enzyme_autodiff(ptr @dotabs, double 1.0)
  ret void
}

declare double @__enzyme_autodiff(...)

; CHECK: define internal { double } @diffedotabs(double %ld, double %differeturn) 
; CHECK-NEXT: invert:
; CHECK-NEXT:   %r = call i32 (ptr, ptr, ...) @fprintf(ptr @stdout, ptr noundef @.str, double noundef %ld)
; CHECK-NEXT:   %0 = insertvalue { double } {{(undef|poison)}}, double %differeturn, 0
; CHECK-NEXT:   ret { double } %0
; CHECK-NEXT: }
