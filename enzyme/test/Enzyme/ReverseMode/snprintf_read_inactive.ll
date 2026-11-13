; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -mem2reg -instsimplify -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify)" -S | FileCheck %s

@.fmt = private unnamed_addr constant [6 x i8] c"%s/%s\00", align 1
@.dir = private unnamed_addr constant [5 x i8] c"/sys\00", align 1
@.name = private unnamed_addr constant [5 x i8] c"size\00", align 1

declare noalias ptr @malloc(i64)
declare void @free(ptr)
declare i32 @snprintf(ptr noalias nocapture writeonly, i64, ptr nocapture readonly, ...)
declare i32 @sprintf(ptr noalias nocapture writeonly, ptr nocapture readonly, ...)
declare i32 @open(ptr nocapture readonly, i32, ...)
declare i64 @read(i32, ptr nocapture, i64)
declare i64 @write(i32, ptr nocapture readonly, i64)
declare i32 @close(i32)
declare ptr @strchr(ptr, i32)
declare double @__enzyme_autodiff(...)

define double @compute(double %x) {
entry:
  %scratch = call noalias ptr @malloc(i64 8)
  store double %x, ptr %scratch, align 8
  %path = alloca [64 x i8], align 1
  %buf = alloca [64 x i8], align 1
  %n1 = call i32 (ptr, i64, ptr, ...) @snprintf(ptr nonnull %path, i64 64, ptr nonnull @.fmt, ptr nonnull @.dir, ptr nonnull @.name)
  %n2 = call i32 (ptr, ptr, ...) @sprintf(ptr nonnull %buf, ptr nonnull @.fmt, ptr nonnull @.dir, ptr nonnull @.name)
  %fd = call i32 (ptr, i32, ...) @open(ptr nonnull %path, i32 0)
  %r = call i64 @read(i32 %fd, ptr nonnull %buf, i64 63)
  %w = call i64 @write(i32 %fd, ptr nonnull %buf, i64 4)
  %sep = call ptr @strchr(ptr nonnull %buf, i32 10)
  %c = call i32 @close(i32 %fd)
  %loaded = load double, ptr %scratch, align 8
  call void @free(ptr %scratch)
  %sq = fmul double %loaded, %loaded
  ret double %sq
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
; CHECK: call i32 (ptr, i64, ptr, ...) @snprintf(
; CHECK: call i32 (ptr, ptr, ...) @sprintf(
; CHECK: call i32 (ptr, i32, ...) @open(
; CHECK: call i64 @read(
; CHECK: call i64 @write(
; CHECK: call ptr @strchr(
; CHECK: call i32 @close(
; CHECK: fadd fast double
; CHECK: ret { double }
