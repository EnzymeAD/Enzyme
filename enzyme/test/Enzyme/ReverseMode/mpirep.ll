; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s
; ModuleID = 'text'
source_filename = "text"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Function Attrs: cold noreturn nounwind
declare void @llvm.trap() #2

declare dso_local i32 @MPI_Comm_rank(i32, i64)

define void @julia_mul_3262({} addrspace(10)* %0, {} addrspace(10)* %1) {
entry:
  %2 = alloca i32, align 8
  %3 = bitcast i32* %2 to i8*
  %4 = bitcast {} addrspace(10)* %1 to i32 addrspace(10)*
  %5 = addrspacecast i32 addrspace(10)* %4 to i32 addrspace(11)*
  %6 = ptrtoint i32* %2 to i64
  %7 = bitcast {} addrspace(10)* %0 to double addrspace(13)* addrspace(10)*
  %8 = addrspacecast double addrspace(13)* addrspace(10)* %7 to double addrspace(13)* addrspace(11)*
  %9 = load i32, i32 addrspace(11)* %5, align 4, !dbg !38, !tbaa !49
  %10 = call i32 @MPI_Comm_rank(i32 %9, i64 noundef %6), !dbg !55
  %.not = icmp eq i32 %10, 0, !dbg !56
  br i1 %.not, label %L13.i, label %L16.i, !dbg !55

L13.i:                                            ; preds = %entry
   %11 = load i32, i32* %2, align 8, !dbg !59, !tbaa !49
   %12 = load double addrspace(13)*, double addrspace(13)* addrspace(11)* %8, align 16, !dbg !64, !tbaa !68, !nonnull !4
   %13 = load double, double addrspace(13)* %12, align 8, !dbg !64, !tbaa !71
   %14 = sitofp i32 %11 to double, !dbg !73
   %15 = fmul double %13, %14, !dbg !88
   %16 = fmul double %13, %15, !dbg !90
   store double %16, double addrspace(13)* %12, align 8, !dbg !91, !tbaa !71
   %17 = load i32, i32 addrspace(11)* %5, align 4, !dbg !38, !tbaa !49
   %18 = call i32 @MPI_Comm_rank(i32 %17, i64 noundef %6), !dbg !55
   %.not.1 = icmp eq i32 %18, 0, !dbg !56
   br i1 %.not.1, label %exit, label %L16.i, !dbg !55

L16.i:                                            ; preds = %L13.i.1, %L13.i, %entry
   call void @llvm.trap() #7, !dbg !55
   unreachable, !dbg !55

exit:
   ret void
}

attributes #0 = { inaccessiblememonly allocsize(1) }
attributes #1 = { nounwind readnone }
attributes #2 = { cold noreturn nounwind }
attributes #3 = { alwaysinline "probe-stack"="inline-asm" }
attributes #4 = { argmemonly nofree nosync nounwind willreturn }
attributes #5 = { inaccessiblemem_or_argmemonly }
attributes #6 = { allocsize(1) }
attributes #7 = { noreturn }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2, !5, !7, !8, !9, !11, !12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !3, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !4, nameTableKind: None)
!3 = !DIFile(filename: "/home/vchuravy/src/Enzyme/nc.jl", directory: ".")
!4 = !{}
!5 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !6, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !4, nameTableKind: None)
!6 = !DIFile(filename: "/home/vchuravy/.julia/packages/GPUCompiler/YNSWF/src/runtime.jl", directory: ".")
!7 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !6, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !4, nameTableKind: None)
!8 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !6, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !4, nameTableKind: None)
!9 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !10, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !4, nameTableKind: None)
!10 = !DIFile(filename: "/home/vchuravy/src/Enzyme/src/compiler.jl", directory: ".")
!11 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !6, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !4, nameTableKind: None)
!12 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !6, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !4, nameTableKind: None)
!13 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !10, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !4, nameTableKind: None)
!14 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !6, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !4, nameTableKind: None)
!15 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !6, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !4, nameTableKind: None)
!16 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !10, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !4, nameTableKind: None)
!17 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !6, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !4, nameTableKind: None)
!18 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !10, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !4, nameTableKind: None)
!19 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !6, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !4, nameTableKind: None)
!20 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !6, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !4, nameTableKind: None)
!21 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !6, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !4, nameTableKind: None)
!22 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !6, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !4, nameTableKind: None)
!23 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !6, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !4, nameTableKind: None)
!24 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !6, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !4, nameTableKind: None)
!25 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !6, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !4, nameTableKind: None)
!26 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !6, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !4, nameTableKind: None)
!27 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !6, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !4, nameTableKind: None)
!28 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !6, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !4, nameTableKind: None)
!29 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !6, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !4, nameTableKind: None)
!30 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !6, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !4, nameTableKind: None)
!31 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !6, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !4, nameTableKind: None)
!32 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !6, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !4, nameTableKind: None)
!33 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !6, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !4, nameTableKind: None)
!34 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !10, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !4, nameTableKind: None)
!35 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !10, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !4, nameTableKind: None)
!36 = distinct !DISubprogram(name: "mul", linkageName: "julia_mul_3262", scope: null, file: !3, line: 6, type: !37, scopeLine: 6, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!37 = !DISubroutineType(types: !4)
!38 = !DILocation(line: 42, scope: !39, inlinedAt: !41)
!39 = distinct !DISubprogram(name: "getproperty;", linkageName: "getproperty", scope: !40, file: !40, type: !37, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!40 = !DIFile(filename: "Base.jl", directory: ".")
!41 = distinct !DILocation(line: 38, scope: !42, inlinedAt: !44)
!42 = distinct !DISubprogram(name: "unsafe_convert;", linkageName: "unsafe_convert", scope: !43, file: !43, type: !37, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!43 = !DIFile(filename: "/home/vchuravy/.julia/packages/MPI/08SPr/src/handle.jl", directory: ".")
!44 = distinct !DILocation(line: 50, scope: !45, inlinedAt: !47)
!45 = distinct !DISubprogram(name: "Comm_rank;", linkageName: "Comm_rank", scope: !46, file: !46, type: !37, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!46 = !DIFile(filename: "/home/vchuravy/.julia/packages/MPI/08SPr/src/comm.jl", directory: ".")
!47 = distinct !DILocation(line: 9, scope: !36, inlinedAt: !48)
!48 = distinct !DILocation(line: 0, scope: !36)
!49 = !{!50, !50, i64 0}
!50 = !{!"jtbaa_mutab", !51, i64 0}
!51 = !{!"jtbaa_value", !52, i64 0}
!52 = !{!"jtbaa_data", !53, i64 0}
!53 = !{!"jtbaa", !54, i64 0}
!54 = !{!"jtbaa"}
!55 = !DILocation(line: 50, scope: !45, inlinedAt: !47)
!56 = !DILocation(line: 468, scope: !57, inlinedAt: !44)
!57 = distinct !DISubprogram(name: "==;", linkageName: "==", scope: !58, file: !58, type: !37, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!58 = !DIFile(filename: "promotion.jl", directory: ".")
!59 = !DILocation(line: 42, scope: !39, inlinedAt: !60)
!60 = distinct !DILocation(line: 56, scope: !61, inlinedAt: !63)
!61 = distinct !DISubprogram(name: "getindex;", linkageName: "getindex", scope: !62, file: !62, type: !37, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!62 = !DIFile(filename: "refvalue.jl", directory: ".")
!63 = distinct !DILocation(line: 51, scope: !45, inlinedAt: !47)
!64 = !DILocation(line: 861, scope: !65, inlinedAt: !67)
!65 = distinct !DISubprogram(name: "getindex;", linkageName: "getindex", scope: !66, file: !66, type: !37, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!66 = !DIFile(filename: "array.jl", directory: ".")
!67 = distinct !DILocation(line: 10, scope: !36, inlinedAt: !48)
!68 = !{!69, !69, i64 0}
!69 = !{!"jtbaa_arrayptr", !70, i64 0}
!70 = !{!"jtbaa_array", !53, i64 0}
!71 = !{!72, !72, i64 0}
!72 = !{!"jtbaa_arraybuf", !52, i64 0}
!73 = !DILocation(line: 146, scope: !74, inlinedAt: !76)
!74 = distinct !DISubprogram(name: "Float64;", linkageName: "Float64", scope: !75, file: !75, type: !37, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!75 = !DIFile(filename: "float.jl", directory: ".")
!76 = distinct !DILocation(line: 7, scope: !77, inlinedAt: !79)
!77 = distinct !DISubprogram(name: "convert;", linkageName: "convert", scope: !78, file: !78, type: !37, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!78 = !DIFile(filename: "number.jl", directory: ".")
!79 = distinct !DILocation(line: 327, scope: !80, inlinedAt: !81)
!80 = distinct !DISubprogram(name: "_promote;", linkageName: "_promote", scope: !58, file: !58, type: !37, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!81 = distinct !DILocation(line: 350, scope: !82, inlinedAt: !83)
!82 = distinct !DISubprogram(name: "promote;", linkageName: "promote", scope: !58, file: !58, type: !37, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!83 = distinct !DILocation(line: 380, scope: !84, inlinedAt: !85)
!84 = distinct !DISubprogram(name: "*;", linkageName: "*", scope: !58, file: !58, type: !37, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!85 = distinct !DILocation(line: 655, scope: !86, inlinedAt: !67)
!86 = distinct !DISubprogram(name: "*;", linkageName: "*", scope: !87, file: !87, type: !37, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!87 = !DIFile(filename: "operators.jl", directory: ".")
!88 = !DILocation(line: 405, scope: !89, inlinedAt: !83)
!89 = distinct !DISubprogram(name: "*;", linkageName: "*", scope: !75, file: !75, type: !37, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!90 = !DILocation(line: 405, scope: !89, inlinedAt: !85)
!91 = !DILocation(line: 903, scope: !92, inlinedAt: !67)
!92 = distinct !DISubprogram(name: "setindex!;", linkageName: "setindex!", scope: !66, file: !66, type: !37, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
