// RUN: %eopt --print-activity-analysis='relative verbose' --split-input-file %s | FileCheck %s

#alias_scope_domain = #llvm.alias_scope_domain<id = distinct[0]<>, description = "wrap">
#loop_unroll = #llvm.loop_unroll<disable = true>
#tbaa_root = #llvm.tbaa_root<id = "Simple C/C++ TBAA">
#alias_scope = #llvm.alias_scope<id = distinct[1]<>, domain = #alias_scope_domain, description = "wrap: %protein">
#alias_scope1 = #llvm.alias_scope<id = distinct[2]<>, domain = #alias_scope_domain, description = "wrap: %ligand">
#alias_scope2 = #llvm.alias_scope<id = distinct[3]<>, domain = #alias_scope_domain, description = "wrap: %forcefield">
#alias_scope3 = #llvm.alias_scope<id = distinct[4]<>, domain = #alias_scope_domain, description = "wrap: %etot">
#loop_annotation = #llvm.loop_annotation<unroll = #loop_unroll, mustProgress = true>
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "float", members = {<#tbaa_type_desc, 0>}>
#tbaa_type_desc2 = #llvm.tbaa_type_desc<id = "any pointer", members = {<#tbaa_type_desc, 0>}>
#tbaa_type_desc3 = #llvm.tbaa_type_desc<id = "long", members = {<#tbaa_type_desc, 0>}>
#tbaa_type_desc4 = #llvm.tbaa_type_desc<id = "int", members = {<#tbaa_type_desc, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc1, access_type = #tbaa_type_desc1, offset = 0>
#tbaa_tag1 = #llvm.tbaa_tag<base_type = #tbaa_type_desc2, access_type = #tbaa_type_desc2, offset = 0>
#tbaa_tag2 = #llvm.tbaa_tag<base_type = #tbaa_type_desc3, access_type = #tbaa_type_desc3, offset = 0>
#tbaa_tag3 = #llvm.tbaa_tag<base_type = #tbaa_type_desc4, access_type = #tbaa_type_desc4, offset = 0>
#tbaa_type_desc5 = #llvm.tbaa_type_desc<id = "", members = {<#tbaa_type_desc4, 0>, <#tbaa_type_desc4, 4>, <#tbaa_type_desc4, 8>, <#tbaa_type_desc4, 12>, <#tbaa_type_desc2, 16>, <#tbaa_type_desc2, 24>, <#tbaa_type_desc2, 32>, <#tbaa_type_desc, 40>, <#tbaa_type_desc2, 88>, <#tbaa_type_desc4, 96>}>
#tbaa_type_desc6 = #llvm.tbaa_type_desc<id = "timeval", members = {<#tbaa_type_desc3, 0>, <#tbaa_type_desc4, 8>}>
#tbaa_type_desc7 = #llvm.tbaa_type_desc<id = "", members = {<#tbaa_type_desc1, 0>, <#tbaa_type_desc1, 4>, <#tbaa_type_desc1, 8>, <#tbaa_type_desc4, 12>}>
#tbaa_tag4 = #llvm.tbaa_tag<base_type = #tbaa_type_desc5, access_type = #tbaa_type_desc4, offset = 12>
#tbaa_tag5 = #llvm.tbaa_tag<base_type = #tbaa_type_desc5, access_type = #tbaa_type_desc4, offset = 96>
#tbaa_tag6 = #llvm.tbaa_tag<base_type = #tbaa_type_desc5, access_type = #tbaa_type_desc4, offset = 0>
#tbaa_tag7 = #llvm.tbaa_tag<base_type = #tbaa_type_desc5, access_type = #tbaa_type_desc4, offset = 4>
#tbaa_tag8 = #llvm.tbaa_tag<base_type = #tbaa_type_desc5, access_type = #tbaa_type_desc2, offset = 88>
#tbaa_tag9 = #llvm.tbaa_tag<base_type = #tbaa_type_desc5, access_type = #tbaa_type_desc2, offset = 24>
#tbaa_tag10 = #llvm.tbaa_tag<base_type = #tbaa_type_desc5, access_type = #tbaa_type_desc2, offset = 16>
#tbaa_tag11 = #llvm.tbaa_tag<base_type = #tbaa_type_desc5, access_type = #tbaa_type_desc2, offset = 32>
#tbaa_tag12 = #llvm.tbaa_tag<base_type = #tbaa_type_desc5, access_type = #tbaa_type_desc4, offset = 8>
#tbaa_tag13 = #llvm.tbaa_tag<base_type = #tbaa_type_desc6, access_type = #tbaa_type_desc4, offset = 8>
#tbaa_tag14 = #llvm.tbaa_tag<base_type = #tbaa_type_desc6, access_type = #tbaa_type_desc3, offset = 0>
#tbaa_tag15 = #llvm.tbaa_tag<base_type = #tbaa_type_desc7, access_type = #tbaa_type_desc1, offset = 0>
#tbaa_tag16 = #llvm.tbaa_tag<base_type = #tbaa_type_desc7, access_type = #tbaa_type_desc1, offset = 4>
#tbaa_tag17 = #llvm.tbaa_tag<base_type = #tbaa_type_desc7, access_type = #tbaa_type_desc1, offset = 8>

// CHECK-LABEL: processing function @fasten_main
// CHECK: p2p summary:
// CHECK-NEXT:     distinct[0]<"fresh-etot"> -> []
// CHECK-NEXT:     distinct[0]<"fresh-lpos_x"> -> []
// CHECK-NEXT:     distinct[0]<"fresh-lpos_y"> -> []
// CHECK-NEXT:     distinct[0]<"fresh-lpos_z"> -> []
// CHECK-NEXT:     distinct[0]<"fresh-transform"> -> []
// CHECK: forward value origins:
// CHECK:          distinct[0]<#enzyme.pseudoclass<@fasten_main(10, 0)>> originates from [#enzyme.argorigin<@fasten_main(10)>, #enzyme.argorigin<@fasten_main(11)>, #enzyme.argorigin<@fasten_main(2)>, #enzyme.argorigin<@fasten_main(3)>, #enzyme.argorigin<@fasten_main(4)>, #enzyme.argorigin<@fasten_main(5)>, #enzyme.argorigin<@fasten_main(6)>, #enzyme.argorigin<@fasten_main(7)>, #enzyme.argorigin<@fasten_main(8)>, #enzyme.argorigin<@fasten_main(9)>]
// CHECK: backward value origins:
// CHECK:          distinct[0]<#enzyme.pseudoclass<@fasten_main(11, 0)>> goes to [#enzyme.argorigin<@fasten_main(10)>, #enzyme.argorigin<@fasten_main(11)>]
// CHECK:          distinct[0]<#enzyme.pseudoclass<@fasten_main(2, 0)>> goes to [#enzyme.argorigin<@fasten_main(10)>, #enzyme.argorigin<@fasten_main(2)>]
// CHECK:          distinct[0]<#enzyme.pseudoclass<@fasten_main(3, 0)>> goes to [#enzyme.argorigin<@fasten_main(10)>, #enzyme.argorigin<@fasten_main(3)>]
// CHECK:          distinct[0]<#enzyme.pseudoclass<@fasten_main(4, 0)>> goes to [#enzyme.argorigin<@fasten_main(10)>, #enzyme.argorigin<@fasten_main(4)>]
// CHECK:          distinct[0]<#enzyme.pseudoclass<@fasten_main(5, 0)>> goes to [#enzyme.argorigin<@fasten_main(10)>, #enzyme.argorigin<@fasten_main(5)>]
// CHECK:          distinct[0]<#enzyme.pseudoclass<@fasten_main(6, 0)>> goes to [#enzyme.argorigin<@fasten_main(10)>, #enzyme.argorigin<@fasten_main(6)>]
// CHECK:          distinct[0]<#enzyme.pseudoclass<@fasten_main(7, 0)>> goes to [#enzyme.argorigin<@fasten_main(10)>, #enzyme.argorigin<@fasten_main(7)>]
// CHECK:          distinct[0]<#enzyme.pseudoclass<@fasten_main(8, 0)>> goes to [#enzyme.argorigin<@fasten_main(10)>, #enzyme.argorigin<@fasten_main(8)>]
// CHECK:          distinct[0]<#enzyme.pseudoclass<@fasten_main(9, 0)>> goes to [#enzyme.argorigin<@fasten_main(10)>, #enzyme.argorigin<@fasten_main(9)>]
llvm.func local_unnamed_addr @fasten_main(%arg0: i32 {llvm.noundef}, %arg1: i32 {llvm.noundef}, %arg2: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg3: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg4: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg5: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg6: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg7: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg8: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg9: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg10: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.writeonly}, %arg11: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg12: i32 {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<"non-leaf">, memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = readwrite>, passthrough = ["nofree", "norecurse", "nosync", "nounwind", "ssp", ["uwtable", "1"], ["approx-func-fp-math", "true"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["unsafe-fp-math", "true"]], target_cpu = "apple-m1", target_features = #llvm.target_features<["+aes", "+complxnum", "+crc", "+dotprod", "+fp-armv8", "+fp16fml", "+fullfp16", "+jsconv", "+lse", "+neon", "+ras", "+rcpc", "+rdm", "+sha2", "+sha3", "+v8.1a", "+v8.2a", "+v8.3a", "+v8.4a", "+v8.5a", "+v8a", "+zcm", "+zcz"]>} {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.constant(8 : i32) : i32
  %2 = llvm.mlir.constant(0 : i64) : i64
  %3 = llvm.mlir.constant(1 : i64) : i64
  %4 = llvm.mlir.constant(2 : i64) : i64
  %5 = llvm.mlir.constant(0 : i8) : i8
  %6 = llvm.mlir.constant(1024 : i64) : i64
  %7 = llvm.mlir.constant(3 : i64) : i64
  %8 = llvm.mlir.constant(256 : i64) : i64
  %9 = llvm.mlir.constant(4 : i64) : i64
  %10 = llvm.mlir.constant(8 : i64) : i64
  %11 = llvm.mlir.constant(12 : i64) : i64
  %12 = llvm.mlir.constant(0.000000e+00 : f32) : f32
  %13 = llvm.mlir.constant(70 : i32) : i32
  %14 = llvm.mlir.constant(69 : i32) : i32
  %15 = llvm.mlir.constant(5.500000e+00 : f32) : f32
  %16 = llvm.mlir.constant(1.000000e+00 : f32) : f32
  %17 = llvm.mlir.constant(-3.40282347E+38 : f32) : f32
  %18 = llvm.mlir.constant(false) : i1
  %19 = llvm.mlir.constant(4.000000e+00 : f32) : f32
  %20 = llvm.mlir.constant(2.000000e+00 : f32) : f32
  %21 = llvm.mlir.constant(2.500000e-01 : f32) : f32
  %22 = llvm.mlir.constant(5.000000e-01 : f32) : f32
  %23 = llvm.mlir.constant(true) : i1
  %24 = llvm.mlir.constant(7.600000e+01 : f32) : f32
  %25 = llvm.mlir.constant(4.500000e+01 : f32) : f32
  %26 = llvm.alloca %0 x !llvm.array<256 x f32> {tag = "lpos_x", alignment = 4 : i64} : (i32) -> !llvm.ptr
  %27 = llvm.alloca %0 x !llvm.array<256 x f32> {tag = "lpos_y", alignment = 4 : i64} : (i32) -> !llvm.ptr
  %28 = llvm.alloca %0 x !llvm.array<256 x f32> {tag = "lpos_z", alignment = 4 : i64} : (i32) -> !llvm.ptr
  %29 = llvm.alloca %0 x !llvm.array<3 x array<4 x array<256 x f32>>> {tag = "transform", alignment = 4 : i64} : (i32) -> !llvm.ptr
  %30 = llvm.alloca %0 x !llvm.array<256 x f32> {tag = "etot", alignment = 4 : i64} : (i32) -> !llvm.ptr
  llvm.intr.lifetime.start 12288, %29 : !llvm.ptr
  llvm.intr.lifetime.start 1024, %30 : !llvm.ptr
  %31 = llvm.shl %arg12, %1 overflow<nsw>  : i32
  %32 = llvm.getelementptr inbounds %29[%2, %3] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<3 x array<4 x array<256 x f32>>>
  %33 = llvm.getelementptr inbounds %29[%2, %4] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<3 x array<4 x array<256 x f32>>>
  "llvm.intr.memset"(%30, %5, %6) <{isVolatile = false, tbaa = [#tbaa_tag]}> : (!llvm.ptr, i8, i64) -> ()
  %34 = llvm.sext %31 : i32 to i64
  llvm.br ^bb2(%2 : i64)
^bb1:  // pred: ^bb2
  %35 = llvm.intr.smax(%arg1, %0)  : (i32, i32) -> i32
  %36 = llvm.intr.smax(%arg0, %0)  : (i32, i32) -> i32
  %37 = llvm.zext %36 : i32 to i64
  %38 = llvm.zext %35 : i32 to i64
  llvm.br ^bb3(%2 : i64)
^bb2(%39: i64):  // 2 preds: ^bb0, ^bb2
  %40 = llvm.add %39, %34 overflow<nsw, nuw>  : i64
  %41 = llvm.getelementptr inbounds %arg4[%40] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  %42 = llvm.load %41 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
  %43 = llvm.intr.sin(%42)  {fastmathFlags = #llvm.fastmath<fast>} : (f32) -> f32
  %44 = llvm.intr.cos(%42)  {fastmathFlags = #llvm.fastmath<fast>} : (f32) -> f32
  %45 = llvm.getelementptr inbounds %arg5[%40] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  %46 = llvm.load %45 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
  %47 = llvm.intr.sin(%46)  {fastmathFlags = #llvm.fastmath<fast>} : (f32) -> f32
  %48 = llvm.intr.cos(%46)  {fastmathFlags = #llvm.fastmath<fast>} : (f32) -> f32
  %49 = llvm.getelementptr inbounds %arg6[%40] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  %50 = llvm.load %49 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
  %51 = llvm.intr.sin(%50)  {fastmathFlags = #llvm.fastmath<fast>} : (f32) -> f32
  %52 = llvm.intr.cos(%50)  {fastmathFlags = #llvm.fastmath<fast>} : (f32) -> f32
  %53 = llvm.fmul %52, %48  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %54 = llvm.getelementptr inbounds %29[%2, %39] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<256 x f32>
  llvm.store %53, %54 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : f32, !llvm.ptr
  %55 = llvm.fmul %47, %43  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %56 = llvm.fmul %55, %52  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %57 = llvm.fmul %51, %44  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %58 = llvm.fsub %56, %57  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %59 = llvm.getelementptr inbounds %29[%2, %3, %39] : (!llvm.ptr, i64, i64, i64) -> !llvm.ptr, !llvm.array<4 x array<256 x f32>>
  llvm.store %58, %59 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : f32, !llvm.ptr
  %60 = llvm.fmul %47, %44  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %61 = llvm.fmul %60, %52  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %62 = llvm.fmul %51, %43  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %63 = llvm.fadd %61, %62  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %64 = llvm.getelementptr inbounds %29[%2, %4, %39] : (!llvm.ptr, i64, i64, i64) -> !llvm.ptr, !llvm.array<4 x array<256 x f32>>
  llvm.store %63, %64 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : f32, !llvm.ptr
  %65 = llvm.getelementptr inbounds %arg7[%40] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  %66 = llvm.load %65 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
  %67 = llvm.getelementptr inbounds %29[%2, %7, %39] : (!llvm.ptr, i64, i64, i64) -> !llvm.ptr, !llvm.array<4 x array<256 x f32>>
  llvm.store %66, %67 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : f32, !llvm.ptr
  %68 = llvm.fmul %51, %48  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %69 = llvm.getelementptr inbounds %32[%2, %39] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<256 x f32>
  llvm.store %68, %69 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : f32, !llvm.ptr
  %70 = llvm.fmul %55, %51  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %71 = llvm.fmul %52, %44  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %72 = llvm.fadd %70, %71  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %73 = llvm.getelementptr inbounds %29[%2, %3, %3, %39] : (!llvm.ptr, i64, i64, i64, i64) -> !llvm.ptr, !llvm.array<3 x array<4 x array<256 x f32>>>
  llvm.store %72, %73 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : f32, !llvm.ptr
  %74 = llvm.fmul %60, %51  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %75 = llvm.fmul %52, %43  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %76 = llvm.fsub %74, %75  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %77 = llvm.getelementptr inbounds %29[%2, %3, %4, %39] : (!llvm.ptr, i64, i64, i64, i64) -> !llvm.ptr, !llvm.array<3 x array<4 x array<256 x f32>>>
  llvm.store %76, %77 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : f32, !llvm.ptr
  %78 = llvm.getelementptr inbounds %arg8[%40] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  %79 = llvm.load %78 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
  %80 = llvm.getelementptr inbounds %29[%2, %3, %7, %39] : (!llvm.ptr, i64, i64, i64, i64) -> !llvm.ptr, !llvm.array<3 x array<4 x array<256 x f32>>>
  llvm.store %79, %80 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : f32, !llvm.ptr
  %81 = llvm.fneg %47  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %82 = llvm.getelementptr inbounds %33[%2, %39] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<256 x f32>
  llvm.store %81, %82 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : f32, !llvm.ptr
  %83 = llvm.fmul %48, %43  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %84 = llvm.getelementptr inbounds %29[%2, %4, %3, %39] : (!llvm.ptr, i64, i64, i64, i64) -> !llvm.ptr, !llvm.array<3 x array<4 x array<256 x f32>>>
  llvm.store %83, %84 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : f32, !llvm.ptr
  %85 = llvm.fmul %48, %44  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %86 = llvm.getelementptr inbounds %29[%2, %4, %4, %39] : (!llvm.ptr, i64, i64, i64, i64) -> !llvm.ptr, !llvm.array<3 x array<4 x array<256 x f32>>>
  llvm.store %85, %86 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : f32, !llvm.ptr
  %87 = llvm.getelementptr inbounds %arg9[%40] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  %88 = llvm.load %87 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
  %89 = llvm.getelementptr inbounds %29[%2, %4, %7, %39] : (!llvm.ptr, i64, i64, i64, i64) -> !llvm.ptr, !llvm.array<3 x array<4 x array<256 x f32>>>
  llvm.store %88, %89 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : f32, !llvm.ptr
  %90 = llvm.add %39, %3 overflow<nsw, nuw>  : i64
  %91 = llvm.icmp "eq" %90, %8 : i64
  llvm.cond_br %91, ^bb1, ^bb2(%90 : i64) {loop_annotation = #loop_annotation}
^bb3(%92: i64):  // 2 preds: ^bb1, ^bb9
  llvm.intr.experimental.noalias.scope.decl #alias_scope
  llvm.intr.experimental.noalias.scope.decl #alias_scope1
  llvm.intr.experimental.noalias.scope.decl #alias_scope2
  llvm.intr.experimental.noalias.scope.decl #alias_scope3
  %93 = llvm.getelementptr inbounds %arg3[%92] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Atom", (f32, f32, f32, i32)>
  %94 = llvm.load %93 {alias_scopes = [#alias_scope1], alignment = 4 : i64, noalias_scopes = [#alias_scope, #alias_scope2, #alias_scope3]} : !llvm.ptr -> f32
  %95 = llvm.getelementptr inbounds %93[%9] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  %96 = llvm.load %95 {alias_scopes = [#alias_scope1], alignment = 4 : i64, noalias_scopes = [#alias_scope, #alias_scope2, #alias_scope3]} : !llvm.ptr -> f32
  %97 = llvm.getelementptr inbounds %93[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  %98 = llvm.load %97 {alias_scopes = [#alias_scope1], alignment = 4 : i64, noalias_scopes = [#alias_scope, #alias_scope2, #alias_scope3]} : !llvm.ptr -> f32
  %99 = llvm.getelementptr inbounds %93[%11] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  %100 = llvm.load %99 {alias_scopes = [#alias_scope1], alignment = 4 : i64, noalias_scopes = [#alias_scope, #alias_scope2, #alias_scope3]} : !llvm.ptr -> i32
  %101 = llvm.sext %100 : i32 to i64
  %102 = llvm.getelementptr inbounds %arg11[%101] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.FFParams", (i32, f32, f32, f32)>
  %103 = llvm.load %102 {alias_scopes = [#alias_scope2], alignment = 4 : i64, noalias_scopes = [#alias_scope, #alias_scope1, #alias_scope3]} : !llvm.ptr -> i32
  %104 = llvm.getelementptr inbounds %102[%9] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  %105 = llvm.load %104 {alias_scopes = [#alias_scope2], alignment = 4 : i64, noalias_scopes = [#alias_scope, #alias_scope1, #alias_scope3]} : !llvm.ptr -> f32
  %106 = llvm.getelementptr inbounds %102[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  %107 = llvm.load %106 {alias_scopes = [#alias_scope2], alignment = 4 : i64, noalias_scopes = [#alias_scope, #alias_scope1, #alias_scope3]} : !llvm.ptr -> f32
  %108 = llvm.getelementptr inbounds %102[%11] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  %109 = llvm.load %108 {alias_scopes = [#alias_scope2], alignment = 4 : i64, noalias_scopes = [#alias_scope, #alias_scope1, #alias_scope3]} : !llvm.ptr -> f32
  %110 = llvm.fcmp "olt" %107, %12 {fastmathFlags = #llvm.fastmath<fast>} : f32
  llvm.intr.lifetime.start 1024, %26 : !llvm.ptr
  llvm.intr.lifetime.start 1024, %27 : !llvm.ptr
  llvm.intr.lifetime.start 1024, %28 : !llvm.ptr
  llvm.br ^bb5(%2 : i64)
^bb4:  // pred: ^bb5
  %111 = llvm.fcmp "ogt" %107, %12 {fastmathFlags = #llvm.fastmath<fast>} : f32
  %112 = llvm.icmp "eq" %103, %13 : i32
  %113 = llvm.icmp "eq" %103, %14 : i32
  %114 = llvm.fneg %107  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %115 = llvm.select %110, %15, %16 {fastmathFlags = #llvm.fastmath<fast>} : i1, f32
  %116 = llvm.select %110, %16, %17 {fastmathFlags = #llvm.fastmath<fast>} : i1, f32
  llvm.br ^bb6(%2 : i64)
^bb5(%117: i64):  // 2 preds: ^bb3, ^bb5
  %118 = llvm.getelementptr inbounds %29[%2, %7, %117] : (!llvm.ptr, i64, i64, i64) -> !llvm.ptr, !llvm.array<4 x array<256 x f32>>
  %119 = llvm.load %118 {alignment = 4 : i64, noalias_scopes = [#alias_scope, #alias_scope1, #alias_scope2, #alias_scope3], tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
  %120 = llvm.getelementptr inbounds %29[%2, %117] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<256 x f32>
  %121 = llvm.load %120 {alignment = 4 : i64, noalias_scopes = [#alias_scope, #alias_scope1, #alias_scope2, #alias_scope3], tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
  %122 = llvm.fmul %121, %94  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %123 = llvm.fadd %122, %119  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %124 = llvm.getelementptr inbounds %29[%2, %3, %117] : (!llvm.ptr, i64, i64, i64) -> !llvm.ptr, !llvm.array<4 x array<256 x f32>>
  %125 = llvm.load %124 {alignment = 4 : i64, noalias_scopes = [#alias_scope, #alias_scope1, #alias_scope2, #alias_scope3], tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
  %126 = llvm.fmul %125, %96  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %127 = llvm.fadd %123, %126  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %128 = llvm.getelementptr inbounds %29[%2, %4, %117] : (!llvm.ptr, i64, i64, i64) -> !llvm.ptr, !llvm.array<4 x array<256 x f32>>
  %129 = llvm.load %128 {alignment = 4 : i64, noalias_scopes = [#alias_scope, #alias_scope1, #alias_scope2, #alias_scope3], tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
  %130 = llvm.fmul %129, %98  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %131 = llvm.fadd %127, %130  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %132 = llvm.getelementptr inbounds %26[%2, %117] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<256 x f32>
  llvm.store %131, %132 {alignment = 4 : i64, noalias_scopes = [#alias_scope, #alias_scope1, #alias_scope2, #alias_scope3], tbaa = [#tbaa_tag]} : f32, !llvm.ptr
  %133 = llvm.getelementptr inbounds %29[%3, %7, %117] : (!llvm.ptr, i64, i64, i64) -> !llvm.ptr, !llvm.array<4 x array<256 x f32>>
  %134 = llvm.load %133 {alignment = 4 : i64, noalias_scopes = [#alias_scope, #alias_scope1, #alias_scope2, #alias_scope3], tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
  %135 = llvm.getelementptr inbounds %32[%2, %117] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<256 x f32>
  %136 = llvm.load %135 {alignment = 4 : i64, noalias_scopes = [#alias_scope, #alias_scope1, #alias_scope2, #alias_scope3], tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
  %137 = llvm.fmul %136, %94  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %138 = llvm.fadd %137, %134  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %139 = llvm.getelementptr inbounds %29[%3, %3, %117] : (!llvm.ptr, i64, i64, i64) -> !llvm.ptr, !llvm.array<4 x array<256 x f32>>
  %140 = llvm.load %139 {alignment = 4 : i64, noalias_scopes = [#alias_scope, #alias_scope1, #alias_scope2, #alias_scope3], tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
  %141 = llvm.fmul %140, %96  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %142 = llvm.fadd %138, %141  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %143 = llvm.getelementptr inbounds %29[%3, %4, %117] : (!llvm.ptr, i64, i64, i64) -> !llvm.ptr, !llvm.array<4 x array<256 x f32>>
  %144 = llvm.load %143 {alignment = 4 : i64, noalias_scopes = [#alias_scope, #alias_scope1, #alias_scope2, #alias_scope3], tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
  %145 = llvm.fmul %144, %98  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %146 = llvm.fadd %142, %145  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %147 = llvm.getelementptr inbounds %27[%2, %117] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<256 x f32>
  llvm.store %146, %147 {alignment = 4 : i64, noalias_scopes = [#alias_scope, #alias_scope1, #alias_scope2, #alias_scope3], tbaa = [#tbaa_tag]} : f32, !llvm.ptr
  %148 = llvm.getelementptr inbounds %29[%4, %7, %117] : (!llvm.ptr, i64, i64, i64) -> !llvm.ptr, !llvm.array<4 x array<256 x f32>>
  %149 = llvm.load %148 {alignment = 4 : i64, noalias_scopes = [#alias_scope, #alias_scope1, #alias_scope2, #alias_scope3], tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
  %150 = llvm.getelementptr inbounds %33[%2, %117] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<256 x f32>
  %151 = llvm.load %150 {alignment = 4 : i64, noalias_scopes = [#alias_scope, #alias_scope1, #alias_scope2, #alias_scope3], tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
  %152 = llvm.fmul %151, %94  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %153 = llvm.fadd %152, %149  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %154 = llvm.getelementptr inbounds %29[%4, %3, %117] : (!llvm.ptr, i64, i64, i64) -> !llvm.ptr, !llvm.array<4 x array<256 x f32>>
  %155 = llvm.load %154 {alignment = 4 : i64, noalias_scopes = [#alias_scope, #alias_scope1, #alias_scope2, #alias_scope3], tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
  %156 = llvm.fmul %155, %96  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %157 = llvm.fadd %153, %156  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %158 = llvm.getelementptr inbounds %29[%4, %4, %117] : (!llvm.ptr, i64, i64, i64) -> !llvm.ptr, !llvm.array<4 x array<256 x f32>>
  %159 = llvm.load %158 {alignment = 4 : i64, noalias_scopes = [#alias_scope, #alias_scope1, #alias_scope2, #alias_scope3], tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
  %160 = llvm.fmul %159, %98  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %161 = llvm.fadd %157, %160  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %162 = llvm.getelementptr inbounds %28[%2, %117] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<256 x f32>
  llvm.store %161, %162 {alignment = 4 : i64, noalias_scopes = [#alias_scope, #alias_scope1, #alias_scope2, #alias_scope3], tbaa = [#tbaa_tag]} : f32, !llvm.ptr
  %163 = llvm.add %117, %3 overflow<nsw, nuw>  : i64
  %164 = llvm.icmp "eq" %163, %8 : i64
  llvm.cond_br %164, ^bb4, ^bb5(%163 : i64) {loop_annotation = #loop_annotation}
^bb6(%165: i64):  // 2 preds: ^bb4, ^bb7
  %166 = llvm.getelementptr inbounds %arg2[%165] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Atom", (f32, f32, f32, i32)>
  %167 = llvm.load %166 {alias_scopes = [#alias_scope], alignment = 4 : i64, noalias_scopes = [#alias_scope1, #alias_scope2, #alias_scope3]} : !llvm.ptr -> f32
  %168 = llvm.getelementptr inbounds %166[%9] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  %169 = llvm.load %168 {alias_scopes = [#alias_scope], alignment = 4 : i64, noalias_scopes = [#alias_scope1, #alias_scope2, #alias_scope3]} : !llvm.ptr -> f32
  %170 = llvm.getelementptr inbounds %166[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  %171 = llvm.load %170 {alias_scopes = [#alias_scope], alignment = 4 : i64, noalias_scopes = [#alias_scope1, #alias_scope2, #alias_scope3]} : !llvm.ptr -> f32
  %172 = llvm.getelementptr inbounds %166[%11] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  %173 = llvm.load %172 {alias_scopes = [#alias_scope], alignment = 4 : i64, noalias_scopes = [#alias_scope1, #alias_scope2, #alias_scope3]} : !llvm.ptr -> i32
  %174 = llvm.sext %173 : i32 to i64
  %175 = llvm.getelementptr inbounds %arg11[%174] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.FFParams", (i32, f32, f32, f32)>
  %176 = llvm.load %175 {alias_scopes = [#alias_scope2], alignment = 4 : i64, noalias_scopes = [#alias_scope, #alias_scope1, #alias_scope3]} : !llvm.ptr -> i32
  %177 = llvm.getelementptr inbounds %175[%9] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  %178 = llvm.load %177 {alias_scopes = [#alias_scope2], alignment = 4 : i64, noalias_scopes = [#alias_scope, #alias_scope1, #alias_scope3]} : !llvm.ptr -> f32
  %179 = llvm.getelementptr inbounds %175[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  %180 = llvm.load %179 {alias_scopes = [#alias_scope2], alignment = 4 : i64, noalias_scopes = [#alias_scope, #alias_scope1, #alias_scope3]} : !llvm.ptr -> f32
  %181 = llvm.getelementptr inbounds %175[%11] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  %182 = llvm.load %181 {alias_scopes = [#alias_scope2], alignment = 4 : i64, noalias_scopes = [#alias_scope, #alias_scope1, #alias_scope3]} : !llvm.ptr -> f32
  %183 = llvm.fadd %178, %105  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %184 = llvm.icmp "eq" %176, %13 : i32
  %185 = llvm.select %184, %112, %18 : i1, i1
  %186 = llvm.select %185, %19, %20 {fastmathFlags = #llvm.fastmath<fast>} : i1, f32
  %187 = llvm.select %185, %21, %22 {fastmathFlags = #llvm.fastmath<fast>} : i1, f32
  %188 = llvm.icmp "eq" %176, %14 : i32
  %189 = llvm.select %188, %23, %113 : i1, i1
  %190 = llvm.fcmp "olt" %180, %12 {fastmathFlags = #llvm.fastmath<fast>} : f32
  %191 = llvm.fcmp "ogt" %180, %12 {fastmathFlags = #llvm.fastmath<fast>} : f32
  %192 = llvm.fcmp "une" %180, %12 {fastmathFlags = #llvm.fastmath<fast>} : f32
  %193 = llvm.select %190, %111, %18 : i1, i1
  %194 = llvm.fneg %180  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %195 = llvm.select %193, %194, %180 {fastmathFlags = #llvm.fastmath<fast>} : i1, f32
  %196 = llvm.select %191, %110, %18 : i1, i1
  %197 = llvm.select %196, %114, %107 {fastmathFlags = #llvm.fastmath<fast>} : i1, f32
  %198 = llvm.select %190, %115, %116 {fastmathFlags = #llvm.fastmath<fast>} : i1, f32
  %199 = llvm.fadd %195, %197  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %200 = llvm.fmul %182, %109  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %201 = llvm.fdiv %16, %183  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %202 = llvm.fdiv %16, %198  {fastmathFlags = #llvm.fastmath<fast>} : f32
  llvm.br ^bb8(%2 : i64)
^bb7:  // pred: ^bb8
  %203 = llvm.add %165, %3 overflow<nsw, nuw>  : i64
  %204 = llvm.icmp "eq" %203, %38 : i64
  llvm.cond_br %204, ^bb9, ^bb6(%203 : i64) {loop_annotation = #loop_annotation}
^bb8(%205: i64):  // 2 preds: ^bb6, ^bb8
  %206 = llvm.getelementptr inbounds %26[%2, %205] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<256 x f32>
  %207 = llvm.load %206 {alignment = 4 : i64, noalias_scopes = [#alias_scope, #alias_scope1, #alias_scope2, #alias_scope3], tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
  %208 = llvm.fsub %207, %167  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %209 = llvm.getelementptr inbounds %27[%2, %205] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<256 x f32>
  %210 = llvm.load %209 {alignment = 4 : i64, noalias_scopes = [#alias_scope, #alias_scope1, #alias_scope2, #alias_scope3], tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
  %211 = llvm.fsub %210, %169  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %212 = llvm.getelementptr inbounds %28[%2, %205] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<256 x f32>
  %213 = llvm.load %212 {alignment = 4 : i64, noalias_scopes = [#alias_scope, #alias_scope1, #alias_scope2, #alias_scope3], tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
  %214 = llvm.fsub %213, %171  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %215 = llvm.fmul %208, %208  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %216 = llvm.fmul %211, %211  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %217 = llvm.fadd %216, %215  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %218 = llvm.fmul %214, %214  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %219 = llvm.fadd %217, %218  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %220 = llvm.intr.sqrt(%219)  {fastmathFlags = #llvm.fastmath<fast>} : (f32) -> f32
  %221 = llvm.fsub %220, %183  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %222 = llvm.fcmp "olt" %221, %12 {fastmathFlags = #llvm.fastmath<fast>} : f32
  %223 = llvm.fmul %220, %201  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %224 = llvm.fsub %16, %223  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %225 = llvm.select %222, %24, %12 {fastmathFlags = #llvm.fastmath<fast>} : i1, f32
  %226 = llvm.fmul %225, %224  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %227 = llvm.getelementptr inbounds %30[%205] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  %228 = llvm.load %227 {alias_scopes = [#alias_scope3], alignment = 4 : i64, noalias_scopes = [#alias_scope, #alias_scope1, #alias_scope2], tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
  %229 = llvm.fadd %226, %228  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %230 = llvm.fmul %221, %187  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %231 = llvm.fsub %16, %230  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %232 = llvm.select %222, %16, %231 {fastmathFlags = #llvm.fastmath<fast>} : i1, f32
  %233 = llvm.fcmp "olt" %221, %186 {fastmathFlags = #llvm.fastmath<fast>} : f32
  %234 = llvm.select %233, %200, %12 {fastmathFlags = #llvm.fastmath<fast>} : i1, f32
  %235 = llvm.fmul %234, %232  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %236 = llvm.intr.fabs(%235)  {fastmathFlags = #llvm.fastmath<fast>} : (f32) -> f32
  %237 = llvm.fneg %236  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %238 = llvm.select %189, %237, %235 {fastmathFlags = #llvm.fastmath<fast>} : i1, f32
  %239 = llvm.fmul %238, %25  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %240 = llvm.fmul %221, %202  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %241 = llvm.fsub %16, %240  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %242 = llvm.fcmp "olt" %221, %198 {fastmathFlags = #llvm.fastmath<fast>} : f32
  %243 = llvm.select %242, %192, %18 : i1, i1
  %244 = llvm.select %243, %199, %12 {fastmathFlags = #llvm.fastmath<fast>} : i1, f32
  %245 = llvm.select %222, %16, %241 {fastmathFlags = #llvm.fastmath<fast>} : i1, f32
  %246 = llvm.fmul %244, %245  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %247 = llvm.fadd %229, %246  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %248 = llvm.fadd %247, %239  {fastmathFlags = #llvm.fastmath<fast>} : f32
  llvm.store %248, %227 {alias_scopes = [#alias_scope3], alignment = 4 : i64, noalias_scopes = [#alias_scope, #alias_scope1, #alias_scope2], tbaa = [#tbaa_tag]} : f32, !llvm.ptr
  %249 = llvm.add %205, %3 overflow<nsw, nuw>  : i64
  %250 = llvm.icmp "eq" %249, %8 : i64
  llvm.cond_br %250, ^bb7, ^bb8(%249 : i64) {loop_annotation = #loop_annotation}
^bb9:  // pred: ^bb7
  llvm.intr.lifetime.end 1024, %28 : !llvm.ptr
  llvm.intr.lifetime.end 1024, %27 : !llvm.ptr
  llvm.intr.lifetime.end 1024, %26 : !llvm.ptr
  %251 = llvm.add %92, %3 overflow<nsw, nuw>  : i64
  %252 = llvm.icmp "eq" %251, %37 : i64
  llvm.cond_br %252, ^bb10, ^bb3(%251 : i64) {loop_annotation = #loop_annotation}
^bb10:  // pred: ^bb9
  %253 = llvm.getelementptr %arg10[%34] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  llvm.br ^bb12(%2 : i64)
^bb11:  // pred: ^bb12
  llvm.intr.lifetime.end 1024, %30 : !llvm.ptr
  llvm.intr.lifetime.end 12288, %29 : !llvm.ptr
  llvm.return
^bb12(%254: i64):  // 2 preds: ^bb10, ^bb12
  %255 = llvm.getelementptr inbounds %30[%2, %254] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<256 x f32>
  %256 = llvm.load %255 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f32
  %257 = llvm.fmul %256, %22  {fastmathFlags = #llvm.fastmath<fast>} : f32
  %258 = llvm.getelementptr %253[%254] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  llvm.store %257, %258 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : f32, !llvm.ptr
  %259 = llvm.add %254, %3 overflow<nsw, nuw>  : i64
  %260 = llvm.icmp "eq" %259, %8 : i64
  llvm.cond_br %260, ^bb11, ^bb12(%259 : i64) {loop_annotation = #loop_annotation}
}

llvm.mlir.global internal unnamed_addr @params() {addr_space = 0 : i32, alignment = 8 : i64, dso_local, sym_visibility = "private"} : !llvm.struct<"struct.anon", (i32, i32, i32, i32, ptr, ptr, ptr, array<6 x ptr>, ptr, i32)> {
  %0 = llvm.mlir.constant(0 : i32) : i32
  %1 = llvm.mlir.zero : !llvm.ptr
  %2 = llvm.mlir.undef : !llvm.array<6 x ptr>
  %3 = llvm.insertvalue %1, %2[0] : !llvm.array<6 x ptr> 
  %4 = llvm.insertvalue %1, %3[1] : !llvm.array<6 x ptr> 
  %5 = llvm.insertvalue %1, %4[2] : !llvm.array<6 x ptr> 
  %6 = llvm.insertvalue %1, %5[3] : !llvm.array<6 x ptr> 
  %7 = llvm.insertvalue %1, %6[4] : !llvm.array<6 x ptr> 
  %8 = llvm.insertvalue %1, %7[5] : !llvm.array<6 x ptr> 
  %9 = llvm.mlir.undef : !llvm.struct<"struct.anon", (i32, i32, i32, i32, ptr, ptr, ptr, array<6 x ptr>, ptr, i32)>
  %10 = llvm.insertvalue %0, %9[0] : !llvm.struct<"struct.anon", (i32, i32, i32, i32, ptr, ptr, ptr, array<6 x ptr>, ptr, i32)> 
  %11 = llvm.insertvalue %0, %10[1] : !llvm.struct<"struct.anon", (i32, i32, i32, i32, ptr, ptr, ptr, array<6 x ptr>, ptr, i32)> 
  %12 = llvm.insertvalue %0, %11[2] : !llvm.struct<"struct.anon", (i32, i32, i32, i32, ptr, ptr, ptr, array<6 x ptr>, ptr, i32)> 
  %13 = llvm.insertvalue %0, %12[3] : !llvm.struct<"struct.anon", (i32, i32, i32, i32, ptr, ptr, ptr, array<6 x ptr>, ptr, i32)> 
  %14 = llvm.insertvalue %1, %13[4] : !llvm.struct<"struct.anon", (i32, i32, i32, i32, ptr, ptr, ptr, array<6 x ptr>, ptr, i32)> 
  %15 = llvm.insertvalue %1, %14[5] : !llvm.struct<"struct.anon", (i32, i32, i32, i32, ptr, ptr, ptr, array<6 x ptr>, ptr, i32)> 
  %16 = llvm.insertvalue %1, %15[6] : !llvm.struct<"struct.anon", (i32, i32, i32, i32, ptr, ptr, ptr, array<6 x ptr>, ptr, i32)> 
  %17 = llvm.insertvalue %8, %16[7] : !llvm.struct<"struct.anon", (i32, i32, i32, i32, ptr, ptr, ptr, array<6 x ptr>, ptr, i32)> 
  %18 = llvm.insertvalue %1, %17[8] : !llvm.struct<"struct.anon", (i32, i32, i32, i32, ptr, ptr, ptr, array<6 x ptr>, ptr, i32)> 
  %19 = llvm.insertvalue %0, %18[9] : !llvm.struct<"struct.anon", (i32, i32, i32, i32, ptr, ptr, ptr, array<6 x ptr>, ptr, i32)> 
  llvm.return %19 : !llvm.struct<"struct.anon", (i32, i32, i32, i32, ptr, ptr, ptr, array<6 x ptr>, ptr, i32)>
}

// CHECK-LABEL: processing function @onecompute
// CHECK: p2p summary:
// CHECK-NEXT:    <empty>
// CHECK: forward value origins:
// CHECK:         distinct[0]<#enzyme.pseudoclass<@onecompute(8, 0)>> originates from [#enzyme.argorigin<@onecompute(0)>, #enzyme.argorigin<@onecompute(1)>, #enzyme.argorigin<@onecompute(2)>, #enzyme.argorigin<@onecompute(3)>, #enzyme.argorigin<@onecompute(4)>, #enzyme.argorigin<@onecompute(5)>, #enzyme.argorigin<@onecompute(6)>, #enzyme.argorigin<@onecompute(7)>, #enzyme.argorigin<@onecompute(8)>, #enzyme.argorigin<@onecompute(9)>]
// CHECK: backward value origins:
// CHECK:         distinct[0]<#enzyme.pseudoclass<@onecompute(0, 0)>> goes to [#enzyme.argorigin<@onecompute(0)>, #enzyme.argorigin<@onecompute(8)>]
// CHECK:         distinct[0]<#enzyme.pseudoclass<@onecompute(1, 0)>> goes to [#enzyme.argorigin<@onecompute(1)>, #enzyme.argorigin<@onecompute(8)>]
// CHECK:         distinct[0]<#enzyme.pseudoclass<@onecompute(2, 0)>> goes to [#enzyme.argorigin<@onecompute(2)>, #enzyme.argorigin<@onecompute(8)>]
// CHECK:         distinct[0]<#enzyme.pseudoclass<@onecompute(3, 0)>> goes to [#enzyme.argorigin<@onecompute(3)>, #enzyme.argorigin<@onecompute(8)>]
// CHECK:         distinct[0]<#enzyme.pseudoclass<@onecompute(4, 0)>> goes to [#enzyme.argorigin<@onecompute(4)>, #enzyme.argorigin<@onecompute(8)>]
// CHECK:         distinct[0]<#enzyme.pseudoclass<@onecompute(5, 0)>> goes to [#enzyme.argorigin<@onecompute(5)>, #enzyme.argorigin<@onecompute(8)>]
// CHECK:         distinct[0]<#enzyme.pseudoclass<@onecompute(6, 0)>> goes to [#enzyme.argorigin<@onecompute(6)>, #enzyme.argorigin<@onecompute(8)>]
// CHECK:         distinct[0]<#enzyme.pseudoclass<@onecompute(7, 0)>> goes to [#enzyme.argorigin<@onecompute(7)>, #enzyme.argorigin<@onecompute(8)>]
// CHECK:         distinct[0]<#enzyme.pseudoclass<@onecompute(9, 0)>> goes to [#enzyme.argorigin<@onecompute(8)>, #enzyme.argorigin<@onecompute(9)>]
llvm.func @onecompute(%arg0: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg2: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg3: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg4: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg5: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg6: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg7: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg8: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.writeonly}, %arg9: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}) attributes {frame_pointer = #llvm.framePointerKind<"non-leaf">, memory_effects = #llvm.memory_effects<other = read, argMem = readwrite, inaccessibleMem = readwrite>, passthrough = ["nofree", "norecurse", "nosync", "nounwind", "ssp", ["uwtable", "1"], ["approx-func-fp-math", "true"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["unsafe-fp-math", "true"]], target_cpu = "apple-m1", target_features = #llvm.target_features<["+aes", "+complxnum", "+crc", "+dotprod", "+fp-armv8", "+fp16fml", "+fullfp16", "+jsconv", "+lse", "+neon", "+ras", "+rcpc", "+rdm", "+sha2", "+sha3", "+v8.1a", "+v8.2a", "+v8.3a", "+v8.4a", "+v8.5a", "+v8a", "+zcm", "+zcz"]>} {
  %0 = llvm.mlir.constant(3 : i32) : i32
  %1 = llvm.mlir.constant(0 : i64) : i64
  %2 = llvm.mlir.constant(0 : i32) : i32
  %3 = llvm.mlir.zero : !llvm.ptr
  %4 = llvm.mlir.undef : !llvm.array<6 x ptr>
  %5 = llvm.insertvalue %3, %4[0] : !llvm.array<6 x ptr> 
  %6 = llvm.insertvalue %3, %5[1] : !llvm.array<6 x ptr> 
  %7 = llvm.insertvalue %3, %6[2] : !llvm.array<6 x ptr> 
  %8 = llvm.insertvalue %3, %7[3] : !llvm.array<6 x ptr> 
  %9 = llvm.insertvalue %3, %8[4] : !llvm.array<6 x ptr> 
  %10 = llvm.insertvalue %3, %9[5] : !llvm.array<6 x ptr> 
  %11 = llvm.mlir.undef : !llvm.struct<"struct.anon", (i32, i32, i32, i32, ptr, ptr, ptr, array<6 x ptr>, ptr, i32)>
  %12 = llvm.insertvalue %2, %11[0] : !llvm.struct<"struct.anon", (i32, i32, i32, i32, ptr, ptr, ptr, array<6 x ptr>, ptr, i32)> 
  %13 = llvm.insertvalue %2, %12[1] : !llvm.struct<"struct.anon", (i32, i32, i32, i32, ptr, ptr, ptr, array<6 x ptr>, ptr, i32)> 
  %14 = llvm.insertvalue %2, %13[2] : !llvm.struct<"struct.anon", (i32, i32, i32, i32, ptr, ptr, ptr, array<6 x ptr>, ptr, i32)> 
  %15 = llvm.insertvalue %2, %14[3] : !llvm.struct<"struct.anon", (i32, i32, i32, i32, ptr, ptr, ptr, array<6 x ptr>, ptr, i32)> 
  %16 = llvm.insertvalue %3, %15[4] : !llvm.struct<"struct.anon", (i32, i32, i32, i32, ptr, ptr, ptr, array<6 x ptr>, ptr, i32)> 
  %17 = llvm.insertvalue %3, %16[5] : !llvm.struct<"struct.anon", (i32, i32, i32, i32, ptr, ptr, ptr, array<6 x ptr>, ptr, i32)> 
  %18 = llvm.insertvalue %3, %17[6] : !llvm.struct<"struct.anon", (i32, i32, i32, i32, ptr, ptr, ptr, array<6 x ptr>, ptr, i32)> 
  %19 = llvm.insertvalue %10, %18[7] : !llvm.struct<"struct.anon", (i32, i32, i32, i32, ptr, ptr, ptr, array<6 x ptr>, ptr, i32)> 
  %20 = llvm.insertvalue %3, %19[8] : !llvm.struct<"struct.anon", (i32, i32, i32, i32, ptr, ptr, ptr, array<6 x ptr>, ptr, i32)> 
  %21 = llvm.insertvalue %2, %20[9] : !llvm.struct<"struct.anon", (i32, i32, i32, i32, ptr, ptr, ptr, array<6 x ptr>, ptr, i32)> 
  %22 = llvm.mlir.addressof @params : !llvm.ptr
  %23 = llvm.getelementptr inbounds %22[%1, 3] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.anon", (i32, i32, i32, i32, ptr, ptr, ptr, array<6 x ptr>, ptr, i32)>
  %24 = llvm.mlir.constant(255 : i32) : i32
  %25 = llvm.mlir.constant(511 : i32) : i32
  %26 = llvm.mlir.constant(256 : i32) : i32
  %27 = llvm.mlir.constant(1 : i32) : i32
  %28 = llvm.getelementptr inbounds %22[%1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.anon", (i32, i32, i32, i32, ptr, ptr, ptr, array<6 x ptr>, ptr, i32)>
  %29 = llvm.load %23 {alignment = 4 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
  %30 = llvm.add %29, %24  : i32
  %31 = llvm.icmp "ult" %30, %25 : i32
  llvm.cond_br %31, ^bb2, ^bb1
^bb1:  // pred: ^bb0
  %32 = llvm.sdiv %29, %26  : i32
  %33 = llvm.load %22 {alignment = 8 : i64, tbaa = [#tbaa_tag6]} : !llvm.ptr -> i32
  %34 = llvm.load %28 {alignment = 4 : i64, tbaa = [#tbaa_tag7]} : !llvm.ptr -> i32
  %35 = llvm.intr.umax(%32, %27)  : (i32, i32) -> i32
  llvm.br ^bb3(%2 : i32)
^bb2:  // 2 preds: ^bb0, ^bb3
  llvm.return
^bb3(%36: i32):  // 2 preds: ^bb1, ^bb3
  llvm.call @fasten_main(%33, %34, %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %36) : (i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> ()
  %37 = llvm.add %36, %27 overflow<nuw>  : i32
  %38 = llvm.icmp "eq" %37, %35 : i32
  llvm.cond_br %38, ^bb2, ^bb3(%37 : i32) {loop_annotation = #loop_annotation}
}
