; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="preserve-nvvm"  -opaque-pointers -S | FileCheck %s

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @cblas_dtrmm(i32 noundef %Order, i32 noundef %Side, i32 noundef %Uplo, i32 noundef %TransA, i32 noundef %Diag, i32 noundef %M, i32 noundef %N, double noundef %alpha, ptr noundef %A, i32 noundef %lda, ptr noundef %B, i32 noundef %ldb) #0 {
entry:
  %Order.addr = alloca i32, align 4
  %Side.addr = alloca i32, align 4
  %Uplo.addr = alloca i32, align 4
  %TransA.addr = alloca i32, align 4
  %Diag.addr = alloca i32, align 4
  %M.addr = alloca i32, align 4
  %N.addr = alloca i32, align 4
  %alpha.addr = alloca double, align 8
  %A.addr = alloca ptr, align 8
  %lda.addr = alloca i32, align 4
  %B.addr = alloca ptr, align 8
  %ldb.addr = alloca i32, align 4
  %UL = alloca i8, align 1
  %TA = alloca i8, align 1
  %SD = alloca i8, align 1
  %DI = alloca i8, align 1
  store i32 %Order, ptr %Order.addr, align 4
  store i32 %Side, ptr %Side.addr, align 4
  store i32 %Uplo, ptr %Uplo.addr, align 4
  store i32 %TransA, ptr %TransA.addr, align 4
  store i32 %Diag, ptr %Diag.addr, align 4
  store i32 %M, ptr %M.addr, align 4
  store i32 %N, ptr %N.addr, align 4
  store double %alpha, ptr %alpha.addr, align 8
  store ptr %A, ptr %A.addr, align 8
  store i32 %lda, ptr %lda.addr, align 4
  store ptr %B, ptr %B.addr, align 8
  store i32 %ldb, ptr %ldb.addr, align 4
  %0 = load ptr, ptr %A.addr, align 8
  %1 = load ptr, ptr %B.addr, align 8
  call void @dtrmm_(ptr noundef %SD, ptr noundef %UL, ptr noundef %TA, ptr noundef %DI, ptr noundef %M.addr, ptr noundef %N.addr, ptr noundef %alpha.addr, ptr noundef %0, ptr noundef %lda.addr, ptr noundef %1, ptr noundef %ldb.addr)
  ret void
}

declare dso_local void @dtrmm_(ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef) #1

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"uwtable", i32 2}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{!"google3 clang version 9999.0.0 (7752e0a10b25da2f2eadbed10606bd5454dbca05)"}

; CHECK: declare dso_local void @dtrmm_
