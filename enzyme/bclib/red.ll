; ModuleID = 'red.ll'
source_filename = "text"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@0 = internal unnamed_addr addrspace(3) global [32 x double] zeroinitializer, align 32

define void @__enzyme_dreduce_double_3(double addrspace(3)* %ptr, double %0) #23 {
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %3 = add nuw nsw i32 %2, 1
  %4 = zext i32 %3 to i64
  %5 = lshr i64 %4, 5
  %6 = and i64 %4, 2016
  %7 = icmp ne i64 %6, %4
  %8 = zext i1 %7 to i64
  %9 = add nuw nsw i64 %5, %8
  %10 = and i64 %4, 31
  %11 = icmp eq i64 %10, 0
  %12 = select i1 %11, i64 32, i64 %10
  %13 = bitcast double %0 to i64
  %14 = lshr i64 %13, 32
  %15 = trunc i64 %14 to i32
  %16 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %15, i32 1, i32 31)
  %17 = zext i32 %16 to i64
  %18 = shl nuw i64 %17, 32
  %19 = trunc i64 %13 to i32
  %20 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %19, i32 1, i32 31)
  %21 = zext i32 %20 to i64
  %22 = or i64 %18, %21
  %23 = bitcast i64 %22 to double
  %24 = fadd fast double %23, %0
  %25 = bitcast double %24 to i64
  %26 = lshr i64 %25, 32
  %27 = trunc i64 %26 to i32
  %28 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %27, i32 2, i32 31)
  %29 = zext i32 %28 to i64
  %30 = shl nuw i64 %29, 32
  %31 = trunc i64 %25 to i32
  %32 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %31, i32 2, i32 31)
  %33 = zext i32 %32 to i64
  %34 = or i64 %30, %33
  %35 = bitcast i64 %34 to double
  %36 = fadd fast double %24, %35
  %37 = bitcast double %36 to i64
  %38 = lshr i64 %37, 32
  %39 = trunc i64 %38 to i32
  %40 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %39, i32 4, i32 31)
  %41 = zext i32 %40 to i64
  %42 = shl nuw i64 %41, 32
  %43 = trunc i64 %37 to i32
  %44 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %43, i32 4, i32 31)
  %45 = zext i32 %44 to i64
  %46 = or i64 %42, %45
  %47 = bitcast i64 %46 to double
  %48 = fadd fast double %36, %47
  %49 = bitcast double %48 to i64
  %50 = lshr i64 %49, 32
  %51 = trunc i64 %50 to i32
  %52 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %51, i32 8, i32 31)
  %53 = zext i32 %52 to i64
  %54 = shl nuw i64 %53, 32
  %55 = trunc i64 %49 to i32
  %56 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %55, i32 8, i32 31)
  %57 = zext i32 %56 to i64
  %58 = or i64 %54, %57
  %59 = bitcast i64 %58 to double
  %60 = fadd fast double %48, %59
  %61 = bitcast double %60 to i64
  %62 = lshr i64 %61, 32
  %63 = trunc i64 %62 to i32
  %64 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %63, i32 16, i32 31)
  %65 = trunc i64 %61 to i32
  %66 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %65, i32 16, i32 31)
  %67 = icmp eq i64 %12, 1
  br i1 %67, label %68, label %77

68:                                               ; preds = %1
  %69 = zext i32 %64 to i64
  %70 = shl nuw i64 %69, 32
  %71 = zext i32 %66 to i64
  %72 = or i64 %70, %71
  %73 = bitcast i64 %72 to double
  %74 = fadd fast double %60, %73
  %75 = add nsw i64 %9, -1
  %76 = getelementptr inbounds [32 x double], [32 x double] addrspace(3)* @0, i64 0, i64 %75
  store double %74, double addrspace(3)* %76, align 8
  br label %77

77:                                               ; preds = %68, %1
  call void @llvm.nvvm.barrier0()
  %78 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x(), !range !6
  %79 = zext i32 %78 to i64
  %80 = lshr i64 %79, 5
  %81 = and i64 %79, 992
  %82 = icmp ne i64 %81, %79
  %83 = zext i1 %82 to i64
  %84 = add nuw nsw i64 %80, %83
  %85 = icmp ult i64 %84, %4
  br i1 %85, label %90, label %86

86:                                               ; preds = %77
  %87 = add nsw i64 %12, -1
  %88 = getelementptr inbounds [32 x double], [32 x double] addrspace(3)* @0, i64 0, i64 %87
  %89 = load double, double addrspace(3)* %88, align 8
  br label %90

90:                                               ; preds = %86, %77
  %91 = phi double [ %89, %86 ], [ 0.000000e+00, %77 ]
  %92 = icmp eq i64 %9, 1
  br i1 %92, label %93, label %154

93:                                               ; preds = %90
  %94 = bitcast double %91 to i64
  %95 = lshr i64 %94, 32
  %96 = trunc i64 %95 to i32
  %97 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %96, i32 1, i32 31)
  %98 = zext i32 %97 to i64
  %99 = shl nuw i64 %98, 32
  %100 = trunc i64 %94 to i32
  %101 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %100, i32 1, i32 31)
  %102 = zext i32 %101 to i64
  %103 = or i64 %99, %102
  %104 = bitcast i64 %103 to double
  %105 = fadd fast double %91, %104
  %106 = bitcast double %105 to i64
  %107 = lshr i64 %106, 32
  %108 = trunc i64 %107 to i32
  %109 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %108, i32 2, i32 31)
  %110 = zext i32 %109 to i64
  %111 = shl nuw i64 %110, 32
  %112 = trunc i64 %106 to i32
  %113 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %112, i32 2, i32 31)
  %114 = zext i32 %113 to i64
  %115 = or i64 %111, %114
  %116 = bitcast i64 %115 to double
  %117 = fadd fast double %105, %116
  %118 = bitcast double %117 to i64
  %119 = lshr i64 %118, 32
  %120 = trunc i64 %119 to i32
  %121 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %120, i32 4, i32 31)
  %122 = zext i32 %121 to i64
  %123 = shl nuw i64 %122, 32
  %124 = trunc i64 %118 to i32
  %125 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %124, i32 4, i32 31)
  %126 = zext i32 %125 to i64
  %127 = or i64 %123, %126
  %128 = bitcast i64 %127 to double
  %129 = fadd fast double %117, %128
  %130 = bitcast double %129 to i64
  %131 = lshr i64 %130, 32
  %132 = trunc i64 %131 to i32
  %133 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %132, i32 8, i32 31)
  %134 = zext i32 %133 to i64
  %135 = shl nuw i64 %134, 32
  %136 = trunc i64 %130 to i32
  %137 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %136, i32 8, i32 31)
  %138 = zext i32 %137 to i64
  %139 = or i64 %135, %138
  %140 = bitcast i64 %139 to double
  %141 = fadd fast double %129, %140
  %142 = bitcast double %141 to i64
  %143 = lshr i64 %142, 32
  %144 = trunc i64 %143 to i32
  %145 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %144, i32 16, i32 31)
  %146 = zext i32 %145 to i64
  %147 = shl nuw i64 %146, 32
  %148 = trunc i64 %142 to i32
  %149 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %148, i32 16, i32 31)
  %150 = zext i32 %149 to i64
  %151 = or i64 %147, %150
  %152 = bitcast i64 %151 to double
  %153 = fadd fast double %141, %152
  br label %154

154:                                              ; preds = %93, %90
  %155 = phi double [ %91, %90 ], [ %153, %93 ]
  %cmp = icmp eq i32 0, %2
  br i1 %cmp, label %adder, label %end

adder:
  %prev = load double, double addrspace(3)* %ptr
  %next = fadd fast double %prev, %155
  store double %next, double addrspace(3)* %ptr
  br label %end

end:
  call void @llvm.nvvm.barrier0()
  ret void
}

; Function Attrs: convergent inaccessiblememonly nounwind
declare i32 @llvm.nvvm.shfl.sync.down.i32(i32, i32, i32, i32) #1

; Function Attrs: convergent nounwind
declare void @llvm.nvvm.barrier0() #2

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0

attributes #0 = { nounwind readnone }
attributes #1 = { convergent inaccessiblememonly nounwind }
attributes #2 = { convergent nounwind }
attributes #3 = { nofree nosync nounwind willreturn }
attributes #23 = { alwaysinline }

!2 = !{i32 0, i32 1023}
!6 = !{i32 1, i32 1024}
