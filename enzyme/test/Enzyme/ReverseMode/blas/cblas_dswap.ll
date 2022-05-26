;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

@enzyme_const = common global i32 0, align 4

define void @wrapper(i32 %n, double* %x, i32 %incx, double* %y, i32 %incy) {
entry:
  tail call void @cblas_dswap(i32 %n, double* %x, i32 %incx, double* %y, i32 %incy)
  ret void
}

declare void @cblas_dswap(i32, double*, i32, double*, i32)

define void @wrapperMod(i32 %n, double* %x, i32 %incx, double* %y, i32 %incy) {
entry:
  tail call void @cblas_dswap(i32 %n, double* %x, i32 %incx, double* %y, i32 %incy)
  store double 0.000000e+00, double* %x, align 8
  store double 1.000000e+00, double* %y, align 8
  ret void
}

define void @active(i32 %n, double* %x, double* %_x, i32 %incx, double* %y, double* %_y, i32 %incy) {
entry:
  tail call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (i32, double*, i32, double*, i32)* @wrapper to i8*), i32 %n, double* %x, double* %_x, i32 %incx, double* %y, double* %_y, i32 %incy)
  ret void
}

declare void @__enzyme_autodiff(i8*, ...)

define void @inactiveX(i32 %n, double* %x, double* nocapture readnone %_x, i32 %incx, double* %y, double* %_y, i32 %incy) {
entry:
  %0 = load i32, i32* @enzyme_const, align 4
  tail call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (i32, double*, i32, double*, i32)* @wrapper to i8*), i32 %n, i32 %0, double* %x, i32 %incx, double* %y, double* %_y, i32 %incy)
  ret void
}

define void @inactiveXY(i32 %n, double* %x, double* nocapture readnone %_x, i32 %incx, double* %y, double* nocapture readnone %_y, i32 %incy) {
entry:
  %0 = load i32, i32* @enzyme_const, align 4
  tail call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (i32, double*, i32, double*, i32)* @wrapper to i8*), i32 %n, i32 %0, double* %x, i32 %incx, i32 %0, double* %y, i32 %incy)
  ret void
}

define void @activeMod(i32 %n, double* %x, double* %_x, i32 %incx, double* %y, double* %_y, i32 %incy) {
entry:
  tail call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (i32, double*, i32, double*, i32)* @wrapperMod to i8*), i32 %n, double* %x, double* %_x, i32 %incx, double* %y, double* %_y, i32 %incy)
  ret void
}

;CHECK: define void @active
;CHECK-NEXT: entry
;CHECK-NEXT: call void @[[active:.+]](

;CHECK: define void @inactiveX
;CHECK-NEXT: entry
;CHECK-NEXT: call void @[[inactiveX:.+]](

;CHECK: define void @inactiveXY
;CHECK-NEXT: entry
;CHECK-NEXT: call void @[[inactiveXY:.+]](

;CHECK: define void @activeMod
;CHECK-NEXT: entry
;CHECK-NEXT: call void @[[activeMod:.+]](

;CHECK:define internal void @[[active]](i32 %n, double* %x, double* %"x'", i32 %incx, double* %y, double* %"y'", i32 %incy)
;CHECK-NEXT:entry:
;CHECK-NEXT:  tail call void @cblas_dswap(i32 %n, double* %x, i32 %incx, double* %y, i32 %incy)
;CHECK-NEXT:  call void @cblas_dswap(i32 %n, double* %"x'", i32 %incx, double* %"y'", i32 %incy)
;CHECK-NEXT:  ret void
;CHECK-NEXT:}

;CHECK:define internal void @[[inactiveX]](i32 %n, double* %x, i32 %incx, double* %y, double* %"y'", i32 %incy)
;CHECK-NEXT:entry:
;CHECK-NEXT:  tail call void @cblas_dswap(i32 %n, double* %x, i32 %incx, double* %y, i32 %incy)
;CHECK-NEXT:  %0 = icmp eq i32 %n, 0
;CHECK-NEXT:  br i1 %0, label %__enzyme_memset_double_32_align0stride.exit, label %for.body.i

;CHECK:for.body.i:                                       ; preds = %for.body.i, %entry
;CHECK-NEXT:  %idx.i = phi i32 [ 0, %entry ], [ %idx.next.i, %for.body.i ]
;CHECK-NEXT:  %dst.i.i = getelementptr inbounds double, double* %"y'", i32 %idx.i
;CHECK-NEXT:  store double 0.000000e+00, double* %dst.i.i
;CHECK-NEXT:  %idx.next.i = add nuw i32 %idx.i, 1
;CHECK-NEXT:  %1 = icmp eq i32 %n, %idx.next.i
;CHECK-NEXT:  br i1 %1, label %__enzyme_memset_double_32_align0stride.exit, label %for.body.i

;CHECK:__enzyme_memset_double_32_align0stride.exit:      ; preds = %entry, %for.body.i
;CHECK-NEXT:  ret void
;CHECK-NEXT:}

;CHECK:define internal void @[[inactiveXY]](i32 %n, double* %x, i32 %incx, double* %y, i32 %incy)
;CHECK-NEXT:entry:
;CHECK-NEXT:  tail call void @cblas_dswap(i32 %n, double* %x, i32 %incx, double* %y, i32 %incy)
;CHECK-NEXT:  ret void
;CHECK-NEXT:}

;CHECK:define internal void @[[activeMod]](i32 %n, double* %x, double* %"x'", i32 %incx, double* %y, double* %"y'", i32 %incy)
;CHECK-NEXT:entry:
;CHECK-NEXT:  tail call void @cblas_dswap(i32 %n, double* %x, i32 %incx, double* %y, i32 %incy)
;CHECK-NEXT:  store double 0.000000e+00, double* %x, align 8
;CHECK-NEXT:  store double 1.000000e+00, double* %y, align 8
;CHECK-NEXT:  store double 0.000000e+00, double* %"y'", align 8
;CHECK-NEXT:  store double 0.000000e+00, double* %"x'", align 8
;CHECK-NEXT:  call void @cblas_dswap(i32 %n, double* %"x'", i32 %incx, double* %"y'", i32 %incy)
;CHECK-NEXT:  ret void
;CHECK-NEXT:}
