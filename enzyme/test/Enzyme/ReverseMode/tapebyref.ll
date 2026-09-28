; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPloadEnzyme -enzyme-preopt=false -enzyme-max-tape-fields-by-value=2 -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPloadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s --check-prefix=BYVAL; fi

; A recursive call whose tape holds more scalar fields than the limit hands the
; tape to the reverse pass by pointer, instead of loading it in the caller.

declare double @llvm.sin.f64(double)

define double @f(ptr %x, i32 %n) {
entry:
  %cmp = icmp sle i32 %n, 0
  br i1 %cmp, label %base, label %rec

base:
  %b = load double, ptr %x
  ret double %b

rec:
  %p1 = getelementptr inbounds double, ptr %x, i64 1
  %p2 = getelementptr inbounds double, ptr %x, i64 2
  %a0 = load double, ptr %x
  %a1 = load double, ptr %p1
  %a2 = load double, ptr %p2
  %s0 = call double @llvm.sin.f64(double %a0)
  %s1 = call double @llvm.sin.f64(double %a1)
  %s2 = call double @llvm.sin.f64(double %a2)
  store double %s0, ptr %x
  store double %s1, ptr %p1
  store double %s2, ptr %p2
  %m = sub i32 %n, 1
  %r = call double @f(ptr %x, i32 %m)
  %t0 = fmul double %s0, %s1
  %t1 = fmul double %t0, %s2
  %res = fmul double %t1, %r
  ret double %res
}

define void @dsquare(ptr %x, ptr %dx) {
entry:
  %0 = call double (...) @__enzyme_autodiff(ptr @f, ptr %x, ptr %dx, i32 3)
  ret void
}

declare double @__enzyme_autodiff(...)

; CHECK: define internal void @diffef(ptr {{.*}}%x, ptr {{.*}}%"x'", i32 %n, double %differeturn)
; CHECK: call void @diffef.{{[0-9]+}}(ptr %x, ptr %"x'", i32 %{{.+}}, double %{{.+}}, ptr %{{.+}})

; CHECK: define internal void @diffef.{{[0-9]+}}(ptr {{.*}}%x, ptr {{.*}}%"x'", i32 %n, double %differeturn, ptr %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %truetape = load { ptr, double, double, double, double }, ptr %tapeArg
; CHECK-NEXT:   {{(tail )?}}call void @free(ptr nonnull %tapeArg)
; CHECK-NOT: %tapeld = load

; BYVAL: define internal void @diffef.{{[0-9]+}}(ptr {{.*}}%x, ptr {{.*}}%"x'", i32 %n, double %differeturn, { ptr, double, double, double, double } %tapeArg)
; BYVAL: %tapeld = load { ptr, double, double, double, double }, ptr
