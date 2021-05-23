; ModuleID = 'red.ll'
source_filename = "text"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@0 = internal unnamed_addr addrspace(3) global [32 x double] zeroinitializer, align 32

define void @__enzyme_dreduce_double_3(double addrspace(3)* %0, double %1) {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %4 = add nuw nsw i32 %3, 1
  %5 = zext i32 %4 to i64
  %6 = lshr i64 %5, 5
  %7 = and i64 %5, 2016
  %8 = icmp ne i64 %7, %5
  %9 = zext i1 %8 to i64
  %10 = add nuw nsw i64 %6, %9
  %11 = and i64 %5, 31
  %12 = icmp eq i64 %11, 0
  %13 = select i1 %12, i64 32, i64 %11
  %14 = bitcast double %1 to i64
  %15 = lshr i64 %14, 32
  %16 = trunc i64 %15 to i32
  %17 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %16, i32 1, i32 31)
  %18 = zext i32 %17 to i64
  %19 = shl nuw i64 %18, 32
  %20 = trunc i64 %14 to i32
  %21 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %20, i32 1, i32 31)
  %22 = zext i32 %21 to i64
  %23 = or i64 %19, %22
  %24 = bitcast i64 %23 to double
  %25 = fadd fast double %24, %1
  %26 = bitcast double %25 to i64
  %27 = lshr i64 %26, 32
  %28 = trunc i64 %27 to i32
  %29 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %28, i32 2, i32 31)
  %30 = zext i32 %29 to i64
  %31 = shl nuw i64 %30, 32
  %32 = trunc i64 %26 to i32
  %33 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %32, i32 2, i32 31)
  %34 = zext i32 %33 to i64
  %35 = or i64 %31, %34
  %36 = bitcast i64 %35 to double
  %37 = fadd fast double %25, %36
  %38 = bitcast double %37 to i64
  %39 = lshr i64 %38, 32
  %40 = trunc i64 %39 to i32
  %41 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %40, i32 4, i32 31)
  %42 = zext i32 %41 to i64
  %43 = shl nuw i64 %42, 32
  %44 = trunc i64 %38 to i32
  %45 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %44, i32 4, i32 31)
  %46 = zext i32 %45 to i64
  %47 = or i64 %43, %46
  %48 = bitcast i64 %47 to double
  %49 = fadd fast double %37, %48
  %50 = bitcast double %49 to i64
  %51 = lshr i64 %50, 32
  %52 = trunc i64 %51 to i32
  %53 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %52, i32 8, i32 31)
  %54 = zext i32 %53 to i64
  %55 = shl nuw i64 %54, 32
  %56 = trunc i64 %50 to i32
  %57 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %56, i32 8, i32 31)
  %58 = zext i32 %57 to i64
  %59 = or i64 %55, %58
  %60 = bitcast i64 %59 to double
  %61 = fadd fast double %49, %60
  %62 = bitcast double %61 to i64
  %63 = lshr i64 %62, 32
  %64 = trunc i64 %63 to i32
  %65 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %64, i32 16, i32 31)
  %66 = trunc i64 %62 to i32
  %67 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %66, i32 16, i32 31)
  %68 = icmp eq i64 %13, 1
  br i1 %68, label %69, label %78

69:                                               ; preds = %2
  %70 = zext i32 %65 to i64
  %71 = shl nuw i64 %70, 32
  %72 = zext i32 %67 to i64
  %73 = or i64 %71, %72
  %74 = bitcast i64 %73 to double
  %75 = fadd fast double %61, %74
  %76 = add nsw i64 %10, -1
  %77 = getelementptr inbounds [32 x double], [32 x double] addrspace(3)* @0, i64 0, i64 %76
  store double %75, double addrspace(3)* %77, align 8, !tbaa !3
  br label %78

78:                                               ; preds = %69, %2
  call void @llvm.nvvm.barrier0()
  %79 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x(), !range !6
  %80 = zext i32 %79 to i64
  %81 = lshr i64 %80, 5
  %82 = and i64 %80, 992
  %83 = icmp ne i64 %82, %80
  %84 = zext i1 %83 to i64
  %85 = add nuw nsw i64 %81, %84
  %86 = icmp ult i64 %85, %5
  br i1 %86, label %91, label %87

87:                                               ; preds = %78
  %88 = add nsw i64 %13, -1
  %89 = getelementptr inbounds [32 x double], [32 x double] addrspace(3)* @0, i64 0, i64 %88
  %90 = load double, double addrspace(3)* %89, align 8, !tbaa !3
  br label %91

91:                                               ; preds = %87, %78
  %92 = phi double [ %90, %87 ], [ 0.000000e+00, %78 ]
  %93 = icmp eq i64 %10, 1
  br i1 %93, label %94, label %155

94:                                               ; preds = %91
  %95 = bitcast double %92 to i64
  %96 = lshr i64 %95, 32
  %97 = trunc i64 %96 to i32
  %98 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %97, i32 1, i32 31)
  %99 = zext i32 %98 to i64
  %100 = shl nuw i64 %99, 32
  %101 = trunc i64 %95 to i32
  %102 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %101, i32 1, i32 31)
  %103 = zext i32 %102 to i64
  %104 = or i64 %100, %103
  %105 = bitcast i64 %104 to double
  %106 = fadd fast double %92, %105
  %107 = bitcast double %106 to i64
  %108 = lshr i64 %107, 32
  %109 = trunc i64 %108 to i32
  %110 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %109, i32 2, i32 31)
  %111 = zext i32 %110 to i64
  %112 = shl nuw i64 %111, 32
  %113 = trunc i64 %107 to i32
  %114 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %113, i32 2, i32 31)
  %115 = zext i32 %114 to i64
  %116 = or i64 %112, %115
  %117 = bitcast i64 %116 to double
  %118 = fadd fast double %106, %117
  %119 = bitcast double %118 to i64
  %120 = lshr i64 %119, 32
  %121 = trunc i64 %120 to i32
  %122 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %121, i32 4, i32 31)
  %123 = zext i32 %122 to i64
  %124 = shl nuw i64 %123, 32
  %125 = trunc i64 %119 to i32
  %126 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %125, i32 4, i32 31)
  %127 = zext i32 %126 to i64
  %128 = or i64 %124, %127
  %129 = bitcast i64 %128 to double
  %130 = fadd fast double %118, %129
  %131 = bitcast double %130 to i64
  %132 = lshr i64 %131, 32
  %133 = trunc i64 %132 to i32
  %134 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %133, i32 8, i32 31)
  %135 = zext i32 %134 to i64
  %136 = shl nuw i64 %135, 32
  %137 = trunc i64 %131 to i32
  %138 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %137, i32 8, i32 31)
  %139 = zext i32 %138 to i64
  %140 = or i64 %136, %139
  %141 = bitcast i64 %140 to double
  %142 = fadd fast double %130, %141
  %143 = bitcast double %142 to i64
  %144 = lshr i64 %143, 32
  %145 = trunc i64 %144 to i32
  %146 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %145, i32 16, i32 31)
  %147 = zext i32 %146 to i64
  %148 = shl nuw i64 %147, 32
  %149 = trunc i64 %143 to i32
  %150 = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %149, i32 16, i32 31)
  %151 = zext i32 %150 to i64
  %152 = or i64 %148, %151
  %153 = bitcast i64 %152 to double
  %154 = fadd fast double %142, %153
  br label %155

155:                                              ; preds = %94, %91
  %156 = phi double [ %92, %91 ], [ %154, %94 ]
  %157 = icmp eq i32 %3, 0
  br i1 %157, label %158, label %161

158:                                              ; preds = %155
  %159 = load double, double addrspace(3)* %0, align 8, !tbaa !3
  %160 = fadd fast double %159, %156
  store double %160, double addrspace(3)* %0, align 8, !tbaa !3
  br label %161

161:                                              ; preds = %158, %155
  call void @llvm.nvvm.barrier0()
  ret void
}

; Function Attrs: convergent inaccessiblememonly nounwind
declare i32 @llvm.nvvm.shfl.sync.down.i32(i32, i32, i32, i32) #0

; Function Attrs: convergent nounwind
declare void @llvm.nvvm.barrier0() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #2

attributes #0 = { convergent inaccessiblememonly nounwind }
attributes #1 = { convergent nounwind }
attributes #2 = { nounwind readnone }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 1, !"Debug Info Version", i32 3}
!2 = !{i32 0, i32 1023}
!3 = !{!4, !4, i64 0, i64 0}
!4 = !{!"custom_tbaa_addrspace(3)", !5, i64 0}
!5 = !{!"custom_tbaa"}
!6 = !{i32 1, i32 1024}
