; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

declare noalias ptr @malloc(i64)
declare void @free(ptr)
declare double @__enzyme_autodiff(...)

define double @foo(ptr %fn, double %x) {
entry:
  %m = call ptr @malloc(i64 8)
  store i64 42, ptr %m, align 8
  %r = call double %fn(ptr nocapture readonly %m, double %x)
  call void @free(ptr %m)
  ret double %r
}

define double @caller(ptr %fn, ptr %dfn, double %x) {
entry:
  %r = call double (...) @__enzyme_autodiff(ptr @foo, metadata !"enzyme_dup", ptr %fn, ptr %dfn, double %x)
  ret double %r
}

; CHECK: define internal { double } @diffefoo(ptr %fn, ptr %"fn'", double %x, double %differeturn)
; CHECK:   %m = call ptr @malloc(i64 8)
; CHECK:   store i64 42, ptr %m, align 8
; CHECK:   %r_augmented = call { ptr, double } {{%[0-9]+}}(ptr {{.*}}readonly %m, ptr {{.*}}%m, double %x)
; CHECK:   %subcache = extractvalue { ptr, double } %r_augmented, 0
; CHECK:   call { double } {{%[0-9]+}}(ptr {{.*}}readonly %m, ptr {{.*}}%m, double %x, double %differeturn, ptr %subcache)
; CHECK:   call void @free(ptr %m)
; CHECK:   ret { double }
