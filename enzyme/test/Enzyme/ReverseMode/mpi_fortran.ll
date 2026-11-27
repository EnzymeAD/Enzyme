; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@mpi_real_ = internal constant i32 1275069468, align 4
@mpi_comm_world_ = internal constant i32 0, align 4
@c1 = internal constant i32 1, align 4
@c0 = internal constant i32 0, align 4

define void @power(float* %x, float* %y) {
entry:
  %ierr = alloca i32, align 4
  %rank = alloca i32, align 4
  %request = alloca i32, align 4
  %status = alloca [6 x i32], align 4
  call void @mpi_comm_rank_(i32* @mpi_comm_world_, i32* %rank, i32* %ierr)
  %x_val = load float, float* %x, align 4
  %sq = fmul float %x_val, %x_val
  store float %sq, float* %y, align 4
  %rank_val = load i32, i32* %rank, align 4
  %cmp = icmp eq i32 %rank_val, 1
  %y_i8 = bitcast float* %y to i8*
  %status_i32 = getelementptr inbounds [6 x i32], [6 x i32]* %status, i64 0, i64 0
  br i1 %cmp, label %if.then, label %if.else

if.then:
  call void @mpi_isend_(i8* %y_i8, i32* @c1, i32* @mpi_real_, i32* @c0, i32* @c0, i32* @mpi_comm_world_, i32* %request, i32* %ierr)
  call void @mpi_wait_(i32* %request, i32* %status_i32, i32* %ierr)
  br label %if.end

if.else:
  call void @mpi_irecv_(i8* %y_i8, i32* @c1, i32* @mpi_real_, i32* @c1, i32* @c0, i32* @mpi_comm_world_, i32* %request, i32* %ierr)
  call void @mpi_wait_(i32* %request, i32* %status_i32, i32* %ierr)
  br label %if.end

if.end:
  ret void
}

declare void @mpi_comm_rank_(i32*, i32*, i32*)
declare void @mpi_isend_(i8*, i32*, i32*, i32*, i32*, i32*, i32*, i32*)
declare void @mpi_irecv_(i8*, i32*, i32*, i32*, i32*, i32*, i32*, i32*)
declare void @mpi_wait_(i32*, i32*, i32*)

declare void @__enzyme_autodiff(i8*, ...)

define void @caller(float* %x, float* %dx, float* %y, float* %dy) {
entry:
  call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (float*, float*)* @power to i8*), float* %x, float* %dx, float* %y, float* %dy)
  ret void
}

; CHECK: define internal void @diffepower(float* %x, float* %"x'", float* %y, float* %"y'")
; CHECK: entry:
; CHECK:   %{{[0-9]+}} = alloca [24 x i8], align 1
; CHECK:   %enzyme_mpi_ierr = alloca i32, align 4
; CHECK:   %{{[0-9]+}} = alloca [24 x i8], align 1
; CHECK:   %enzyme_mpi_ierr{{[0-9]*}} = alloca i32, align 4
; CHECK:   call void @mpi_comm_rank_(i32* @mpi_comm_world_, i32* %rank, i32* %ierr)
; CHECK: if.then:
; CHECK:   call void @mpi_type_size_(i8* bitcast (i32* @mpi_real_ to i8*), i32* %{{[0-9]+}}, i32* %{{[0-9]+}})
; CHECK:   call void @mpi_isend_(i8* %y_i8, i32* @c1, i32* @mpi_real_, i32* @c0, i32* @c0, i32* @mpi_comm_world_, i32* %request, i32* %ierr)
; CHECK:   call void @mpi_wait_(i32* %request, i32* %status_i32, i32* %ierr)
; CHECK: if.else:
; CHECK:   call void @mpi_irecv_(i8* %y_i8, i32* @c1, i32* @mpi_real_, i32* @c1, i32* @c0, i32* @mpi_comm_world_, i32* %request, i32* %ierr)
; CHECK:   call void @mpi_wait_(i32* %request, i32* %status_i32, i32* %ierr)
; CHECK: invertif.then_end:
; CHECK:   call void @mpi_type_size_(i8* bitcast (i32* @mpi_real_ to i8*), i32* %{{[0-9]+}}, i32* %{{[0-9]+}})
; CHECK:   call void @mpi_wait_(i32* %request, i32* %{{[0-9]+}}, i32* %enzyme_mpi_ierr)
; CHECK: invertif.else_end:
; CHECK:   call void @mpi_type_size_(i8* bitcast (i32* @mpi_real_ to i8*), i32* %{{[0-9]+}}, i32* %{{[0-9]+}})
; CHECK:   call void @mpi_wait_(i32* %request, i32* %{{[0-9]+}}, i32* %enzyme_mpi_ierr{{[0-9]*}})
; CHECK:   call void @llvm.memset.p0i8.i64(

; CHECK: define internal void @__enzyme_differential_mpi_wait(i8* %buf, i64 %count, i8* %datatype, i64 %source, i64 %tag, i8* %comm, i8 %fn, i32* %d_req)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[CNT_PTR:.+]] = alloca i32, align 4
; CHECK-NEXT:   %[[CNT_TR:.+]] = trunc i64 %count to i32
; CHECK-NEXT:   store i32 %[[CNT_TR]], i32* %[[CNT_PTR]], align 4
; CHECK-NEXT:   %[[DT_PTR:.+]] = bitcast i8* %datatype to i32*
; CHECK-NEXT:   %[[SRC_PTR:.+]] = alloca i32, align 4
; CHECK-NEXT:   %[[SRC_TR:.+]] = trunc i64 %source to i32
; CHECK-NEXT:   store i32 %[[SRC_TR]], i32* %[[SRC_PTR]], align 4
; CHECK-NEXT:   %[[TAG_PTR:.+]] = alloca i32, align 4
; CHECK-NEXT:   %[[TAG_TR:.+]] = trunc i64 %tag to i32
; CHECK-NEXT:   store i32 %[[TAG_TR]], i32* %[[TAG_PTR]], align 4
; CHECK-NEXT:   %[[COMM_PTR:.+]] = bitcast i8* %comm to i32*
; CHECK-NEXT:   %[[IERR_PTR:.+]] = alloca i32, align 4
; CHECK-NEXT:   %[[CMP_FN:.+]] = icmp eq i8 %fn, 1
; CHECK-NEXT:   br i1 %[[CMP_FN]], label %invertISend, label %invertIRecv
; CHECK: invertISend:
; CHECK-NEXT:   call void @mpi_irecv_(i8* %buf, i32* %[[CNT_PTR]], i32* %[[DT_PTR]], i32* %[[SRC_PTR]], i32* %[[TAG_PTR]], i32* %[[COMM_PTR]], i32* %d_req, i32* %[[IERR_PTR]])
; CHECK-NEXT:   br label %common.ret
; CHECK: invertIRecv:
; CHECK-NEXT:   call void @mpi_isend_(i8* %buf, i32* %[[CNT_PTR]], i32* %[[DT_PTR]], i32* %[[SRC_PTR]], i32* %[[TAG_PTR]], i32* %[[COMM_PTR]], i32* %d_req, i32* %[[IERR_PTR]])
; CHECK-NEXT:   br label %common.ret

