; RUN: %opt < %s %newLoadEnzyme -passes=enzyme -S -opaque-pointers | FileCheck %s

%struct.res = type { double, i32 }

define void @callee(ptr sret(%struct.res) %sret_return, ptr %x) {
entry:
  %gep0 = getelementptr inbounds %struct.res, ptr %sret_return, i32 0, i32 0
  %val = load double, ptr %x, align 8
  %mul = fmul double %val, 2.0
  store double %mul, ptr %gep0, align 8
  
  %gep1 = getelementptr inbounds %struct.res, ptr %sret_return, i32 0, i32 1
  store i32 42, ptr %gep1, align 4
  ret void
}

define double @caller(ptr %x) {
entry:
  %sret_box = alloca %struct.res, align 8
  call void @callee(ptr sret(%struct.res) %sret_box, ptr %x)
  %gep0 = getelementptr inbounds %struct.res, ptr %sret_box, i32 0, i32 0
  %val = load double, ptr %gep0, align 8
  ret double %val
}

define void @test(ptr %x, ptr %xp) {
entry:
  call void (...) @__enzyme_augmentfwd(ptr @caller, metadata !"enzyme_dup", ptr %x, ptr %xp)
  ret void
}

declare void @__enzyme_augmentfwd(...)

; CHECK: define internal void @augmented_callee(ptr %sret_return, ptr %sret_return', ptr %x, ptr %x')
; CHECK-NOT: store {{.*}} %sret_return'
