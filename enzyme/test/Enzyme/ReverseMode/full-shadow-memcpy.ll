; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s

%State = type { ptr, double, ptr, double }

define void @scatter(ptr byval(%State) align 8 %arg) {
entry:
  %local = alloca %State, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %local, ptr align 8 %arg, i64 32, i1 false)
  %out.gep = getelementptr inbounds %State, ptr %local, i32 0, i32 0
  %out = load ptr, ptr %out.gep, align 8
  %scale.gep = getelementptr inbounds %State, ptr %local, i32 0, i32 1
  %scale = load double, ptr %scale.gep, align 8
  %in.gep = getelementptr inbounds %State, ptr %local, i32 0, i32 2
  %in = load ptr, ptr %in.gep, align 8
  %x = load double, ptr %in, align 8
  %res = fmul double %x, %scale
  store double %res, ptr %out, align 8
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1)

define void @caller(ptr %arg, ptr %darg) {
entry:
  call void (...) @__enzyme_autodiff(void (ptr)* @scatter, ptr %arg, ptr %darg)
  ret void
}

declare void @__enzyme_autodiff(...)

; CHECK: define internal void @diffescatter(ptr byval(%State) align 8 %arg, ptr align 8 %"arg'")
; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %{{.*}}, ptr align 8 %"arg'", i64 32, i1 false)
