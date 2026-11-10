; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -mem2reg -instsimplify -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify)" -S | FileCheck %s

@.path = private unnamed_addr constant [8 x i8] c"0-3\0A\00\00\00\00", align 1

declare noalias ptr @malloc(i64)
declare void @free(ptr)
declare i32 @sched_getaffinity(i32, i64, ptr)
declare i32 @open(ptr, i32, ...)
declare i64 @read(i32, ptr, i64)
declare i32 @close(i32)
declare i64 @__isoc23_strtol(ptr, ptr, i32)
declare i64 @__isoc23_strtoul(ptr, ptr, i32)
declare double @__enzyme_autodiff(...)

define internal void @query_topology(ptr nocapture %out) noinline {
entry:
  %mask = alloca [16 x i64], align 8
  %buf = alloca [32 x i8], align 1
  %end = alloca ptr, align 8
  %aff = call i32 @sched_getaffinity(i32 0, i64 128, ptr nonnull %mask)
  %fd = call i32 (ptr, i32, ...) @open(ptr nonnull @.path, i32 0)
  %n = call i64 @read(i32 %fd, ptr nonnull %buf, i64 31)
  %parsed0 = call i64 @__isoc23_strtol(ptr nonnull %buf, ptr nonnull %end, i32 10)
  %next = load ptr, ptr %end, align 8
  %parsed1 = call i64 @__isoc23_strtoul(ptr %next, ptr null, i32 10)
  %closed = call i32 @close(i32 %fd)
  %sum = add i64 %parsed0, %parsed1
  store i64 %sum, ptr %out, align 8
  ret void
}

; CHECK-LABEL: define double @caller(
define double @caller(double %x) {
entry:
  ; CHECK: %[[GRAD:.+]] = call { double } @diffecompute(double %x, double 1.000000e+00)
  ; CHECK: %[[RES:.+]] = extractvalue { double } %[[GRAD]], 0
  ; CHECK: ret double %[[RES]]
  %dx = call double (...) @__enzyme_autodiff(ptr @compute, double %x)
  ret double %dx
}

; CHECK-LABEL: define internal { double } @diffecompute(
define double @compute(double %x) {
entry:
  %scratch = call noalias ptr @malloc(i64 8)
  store double %x, ptr %scratch, align 8
  %topo = alloca i64, align 8
  ; CHECK: call void @nofree_query_topology(
  call void @query_topology(ptr nonnull %topo)
  %loaded = load double, ptr %scratch, align 8
  call void @free(ptr %scratch)
  ; CHECK: %[[PARTIAL:.+]] = fadd fast double %{{.+}}, %{{.+}}
  %sq = fmul double %loaded, %loaded
  ; CHECK: %[[PACK:.+]] = insertvalue { double } undef, double %{{.+}}, 0
  ; CHECK: ret { double } %[[PACK]]
  ret double %sq
}
