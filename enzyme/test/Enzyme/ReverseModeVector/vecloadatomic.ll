; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -mem2reg -simplifycfg -adce -instsimplify -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,%simplifycfg,adce,instsimplify)" -enzyme-preopt=false -enzyme-detect-readthrow=0 -S | FileCheck %s

source_filename = "text"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define void @tester(i64 addrspace(12)* %i2, i64 addrspace(13)* %i7) {
entry:
  %i3 = load i64, i64 addrspace(12)* %i2, align 8, !dbg !5, !tbaa !15
  store i64 %i3, i64 addrspace(13)* %i7, align 8, !dbg !35, !tbaa !40
  ret void
}

declare void @__enzyme_reverse(...)

define void @test_derivative(i64 addrspace(12)* %x, i64 addrspace(12)* %dx1, i64 addrspace(12)* %dx2, {} addrspace(13)* %y, {} addrspace(13)* %dy1, {} addrspace(13)* %dy2,  i8* %tape) {
entry:
  call void (...) @__enzyme_reverse(void (i64 addrspace(12)*, i64 addrspace(13)*)* nonnull @tester, metadata !"enzyme_width", i64 2, metadata !"enzyme_dup", i64 addrspace(12)* %x, i64 addrspace(12)* %dx1, i64 addrspace(12)* %dx2, metadata !"enzyme_dup", {} addrspace(13)* %y, {} addrspace(13)* %dy1, {} addrspace(13)* %dy2, i8* %tape)
  ret void
}

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !3, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !4, nameTableKind: None)
!3 = !DIFile(filename: "/mnt/Data/git/Enzyme.jl/revjac.jl", directory: ".")
!4 = !{}
!5 = !DILocation(line: 33, scope: !6, inlinedAt: !9)
!6 = distinct !DISubprogram(name: "getproperty;", linkageName: "getproperty", scope: !7, file: !7, type: !8, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!7 = !DIFile(filename: "Base.jl", directory: ".")
!8 = !DISubroutineType(types: !4)
!9 = distinct !DILocation(line: 56, scope: !10, inlinedAt: !12)
!10 = distinct !DISubprogram(name: "getindex;", linkageName: "getindex", scope: !11, file: !11, type: !8, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!11 = !DIFile(filename: "refvalue.jl", directory: ".")
!12 = distinct !DILocation(line: 6, scope: !13, inlinedAt: !14)
!13 = distinct !DISubprogram(name: "batchbwd", linkageName: "julia_batchbwd_1599", scope: null, file: !3, line: 5, type: !8, scopeLine: 5, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!14 = distinct !DILocation(line: 0, scope: !13)
!15 = !{!16, !16, i64 0}
!16 = !{!"double", !17, i64 0}
!17 = !{!"jtbaa_value", !18, i64 0}
!18 = !{!"jtbaa_data", !19, i64 0}
!19 = !{!"jtbaa", !20, i64 0}
!20 = !{!"jtbaa"}
!21 = !DILocation(line: 448, scope: !22, inlinedAt: !24)
!22 = distinct !DISubprogram(name: "Array;", linkageName: "Array", scope: !23, file: !23, type: !8, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!23 = !DIFile(filename: "boot.jl", directory: ".")
!24 = distinct !DILocation(line: 457, scope: !22, inlinedAt: !25)
!25 = distinct !DILocation(line: 785, scope: !26, inlinedAt: !28)
!26 = distinct !DISubprogram(name: "similar;", linkageName: "similar", scope: !27, file: !27, type: !8, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!27 = !DIFile(filename: "abstractarray.jl", directory: ".")
!28 = distinct !DILocation(line: 784, scope: !26, inlinedAt: !29)
!29 = distinct !DILocation(line: 672, scope: !30, inlinedAt: !32)
!30 = distinct !DISubprogram(name: "_array_for;", linkageName: "_array_for", scope: !31, file: !31, type: !8, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!31 = !DIFile(filename: "array.jl", directory: ".")
!32 = distinct !DILocation(line: 670, scope: !30, inlinedAt: !33)
!33 = distinct !DILocation(line: 108, scope: !34, inlinedAt: !12)
!34 = distinct !DISubprogram(name: "vect;", linkageName: "vect", scope: !31, file: !31, type: !8, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!35 = !DILocation(line: 843, scope: !36, inlinedAt: !33)
!36 = distinct !DISubprogram(name: "setindex!;", linkageName: "setindex!", scope: !31, file: !31, type: !8, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!37 = !{!38, !38, i64 0}
!38 = !{!"jtbaa_arrayptr", !39, i64 0}
!39 = !{!"jtbaa_array", !19, i64 0}
!40 = !{!41, !41, i64 0}
!41 = !{!"jtbaa_arraybuf", !18, i64 0}

; CHECK: define internal void @diffe2tester(i64 addrspace(12)* %i2, [2 x i64 addrspace(12)*] %"i2'", i64 addrspace(13)* %i7, [2 x i64 addrspace(13)*] %"i7'", i8* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   tail call void @free(i8* nonnull %tapeArg)
; CHECK-NEXT:   %"i3'de" = alloca [2 x i64]
; CHECK-NEXT:   store [2 x i64] zeroinitializer, [2 x i64]* %"i3'de"
; CHECK-NEXT:   %[[i0:.+]] = extractvalue [2 x i64 addrspace(13)*] %"i7'", 0
; CHECK-NEXT:   %[[i1:.+]] = load i64, i64 addrspace(13)* %[[i0]]
; CHECK-NEXT:   %[[i2:.+]] = extractvalue [2 x i64 addrspace(13)*] %"i7'", 1
; CHECK-NEXT:   %[[i3:.+]] = load i64, i64 addrspace(13)* %[[i2]]
; CHECK-NEXT:   store i64 0, i64 addrspace(13)* %[[i0]]
; CHECK-NEXT:   store i64 0, i64 addrspace(13)* %[[i2]]
; CHECK-NEXT:   %[[i6:.+]] = getelementptr inbounds [2 x i64], [2 x i64]* %"i3'de", i32 0, i32 0
; CHECK-NEXT:   %[[i7:.+]] = load i64, i64* %[[i6]]
; CHECK-NEXT:   %[[i8:.+]] = bitcast i64 %[[i7]] to double
; CHECK-NEXT:   %[[i9:.+]] = bitcast i64 %[[i1]] to double
; CHECK-NEXT:   %[[i10:.+]] = fadd fast double %[[i8]], %[[i9]]
; CHECK-NEXT:   %[[i11:.+]] = bitcast double %[[i10]] to i64
; CHECK-NEXT:   store i64 %[[i11]], i64* %[[i6]]
; CHECK-NEXT:   %[[i12:.+]] = getelementptr inbounds [2 x i64], [2 x i64]* %"i3'de", i32 0, i32 1
; CHECK-NEXT:   %[[i13:.+]] = load i64, i64* %[[i12]]
; CHECK-NEXT:   %[[i14:.+]] = bitcast i64 %[[i13]] to double
; CHECK-NEXT:   %[[i15:.+]] = bitcast i64 %[[i3]] to double
; CHECK-NEXT:   %[[i16:.+]] = fadd fast double %[[i14]], %[[i15]]
; CHECK-NEXT:   %[[i17:.+]] = bitcast double %[[i16]] to i64
; CHECK-NEXT:   store i64 %[[i17]], i64* %[[i12]]
; CHECK-NEXT:   %[[i18:.+]] = load [2 x i64], [2 x i64]* %"i3'de"
; CHECK-NEXT:   store [2 x i64] zeroinitializer, [2 x i64]* %"i3'de"
; CHECK-NEXT:   %[[i19:.+]] = extractvalue [2 x i64 addrspace(12)*] %"i2'", 0
; CHECK-NEXT:   %[[i20:.+]] = bitcast i64 addrspace(12)* %[[i19]] to double addrspace(12)*
; CHECK-NEXT:   %[[i21:.+]] = extractvalue [2 x i64 addrspace(12)*] %"i2'", 1
; CHECK-NEXT:   %[[i22:.+]] = bitcast i64 addrspace(12)* %[[i21]] to double addrspace(12)*
; CHECK-NEXT:   %[[i23:.+]] = extractvalue [2 x i64] %[[i18]], 0
; CHECK-NEXT:   %[[i24:.+]] = bitcast i64 %[[i23]] to double
; CHECK-NEXT:   %[[i25:.+]] = extractvalue [2 x i64] %[[i18]], 1
; CHECK-NEXT:   %[[i26:.+]] = bitcast i64 %[[i25]] to double
; CHECK-NEXT:   %[[i27:.+]] = atomicrmw fadd double addrspace(12)* %[[i20]], double %[[i24]] monotonic
; CHECK-NEXT:   %[[i28:.+]] = atomicrmw fadd double addrspace(12)* %[[i22]], double %[[i26]] monotonic
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
