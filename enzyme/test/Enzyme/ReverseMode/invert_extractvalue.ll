; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -simplifycfg -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg,adce)" -S | FileCheck %s

source_filename = "start"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
target triple = "x86_64-linux-gnu"

declare void @__enzyme_autodiff(...)

define void @test({} addrspace(10)* %a0, {} addrspace(10)* %a1, { i32, {} addrspace(10)* } %a2) {
entry:
  call void (...) @__enzyme_autodiff({ {} addrspace(10)*, { i32, {} addrspace(10)* } } ({} addrspace(10)*, { i32, {} addrspace(10)* })* @f, metadata !"enzyme_dup", {} addrspace(10)* %a0, {} addrspace(10)* %a1, metadata !"enzyme_const", { i32, {} addrspace(10)* } %a2)
  ret void
}

define "enzyme_type"="{[0]:Pointer, [8]:Integer, [9]:Integer, [10]:Integer, [11]:Integer, [12]:Anything, [13]:Anything, [14]:Anything, [15]:Anything, [16]:Pointer}" { {} addrspace(10)*, { i32, {} addrspace(10)* } } @f({} addrspace(10)* nofree noundef nonnull align 8 dereferenceable(24) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@double, [-1,8]:Pointer, [-1,8,0]:Integer, [-1,8,1]:Integer, [-1,8,2]:Integer, [-1,8,3]:Integer, [-1,8,4]:Integer, [-1,8,5]:Integer, [-1,8,6]:Integer, [-1,8,7]:Integer, [-1,8,8]:Pointer, [-1,8,8,-1]:Float@double, [-1,16]:Integer, [-1,17]:Integer, [-1,18]:Integer, [-1,19]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer}" "enzymejl_parmtype"="4750038272" "enzymejl_parmtype_ref"="2" "enzymejl_parmtype_str"="Vector{Float64}" %a0, { i32, {} addrspace(10)* } "enzyme_type"="{[0]:Integer, [1]:Integer, [2]:Integer, [3]:Integer, [4]:Anything, [5]:Anything, [6]:Anything, [7]:Anything, [8]:Pointer}" "enzymejl_parmtype"="4566615952" "enzymejl_parmtype_ref"="0" %a1) {
entry:
  %a4 = extractvalue { i32, {} addrspace(10)* } %a1, 1, !dbg !35, !enzyme_type !23, !enzymejl_source_type_Any !7, !enzymejl_byref_MUT_REF !7
  %.fca.1.0.extract = extractvalue { i32, {} addrspace(10)* } %a1, 0, !dbg !35, !enzyme_type !25, !enzyme_inactive !7, !enzymejl_source_type_Char !7, !enzymejl_byref_BITS_VALUE !7
  %.fca.0.insert3 = insertvalue { {} addrspace(10)*, { i32, {} addrspace(10)* } } poison, {} addrspace(10)* %a0, 0, !dbg !37
  %.fca.1.0.insert = insertvalue { {} addrspace(10)*, { i32, {} addrspace(10)* } } %.fca.0.insert3, i32 %.fca.1.0.extract, 1, 0, !dbg !37
  %.fca.1.1.insert = insertvalue { {} addrspace(10)*, { i32, {} addrspace(10)* } } %.fca.1.0.insert, {} addrspace(10)* %a4, 1, 1, !dbg !37
  ret { {} addrspace(10)*, { i32, {} addrspace(10)* } } %.fca.1.1.insert, !dbg !37
}

!llvm.module.flags = !{!1, !2}
!llvm.dbg.cu = !{!3}

!0 = !{}
!1 = !{i32 2, !"Dwarf Version", i32 4}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !4, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, nameTableKind: None)
!4 = !DIFile(filename: "julia", directory: ".")
!5 = distinct !DISubprogram(name: "augmented_julia_my_view_154_inner_1wrap", linkageName: "augmented_julia_my_view_154_inner_1wrap", scope: null, file: !6, type: !7, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !3, retainedNodes: !0)
!6 = !DIFile(filename: "test_ka.jl", directory: ".")
!7 = !DISubroutineType(types: !0)
!12 = !{!13, !13, i64 0}
!13 = !{!"jtbaa_gcframe", !14, i64 0}
!14 = !{!"jtbaa", !15, i64 0}
!15 = !{!"jtbaa"}
!16 = !{!17, !17, i64 0}
!17 = !{!"jtbaa_safepoint", !14, i64 0}
!23 = !{!"Unknown", i32 -1, !24}
!24 = !{!"Pointer"}
!25 = !{!"Unknown", i32 -1, !26}
!26 = !{!"Integer"}
!28 = distinct !DISubprogram(name: "my_view", linkageName: "julia_my_view_154", scope: null, file: !6, line: 17, type: !7, scopeLine: 17, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !3, retainedNodes: !0)
!29 = !{!30}
!30 = distinct !{!30, !31, !"shadow_0"}
!31 = distinct !{!31, !"julia_my_view_154_inner"}
!32 = distinct !DISubprogram(name: "MyLTA;", linkageName: "MyLTA", scope: !6, file: !6, type: !7, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !3, retainedNodes: !0)
!33 = !DILocation(line: 17, scope: !28)
!35 = !DILocation(line: 13, scope: !32, inlinedAt: !33)
!37 = !DILocation(line: 17, scope: !28)
