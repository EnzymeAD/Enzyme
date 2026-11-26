; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

; Each kernel's derivative zeroes the shadow of its own shared buffer only, so
; the other kernel's shadow does not count towards its static shared memory.

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@tile_a = internal unnamed_addr addrspace(3) global [64 x double] undef, align 8
@tile_b = internal unnamed_addr addrspace(3) global [512 x double] undef, align 8

define void @ka(ptr %x, ptr %y) {
entry:
  %t = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %i = zext i32 %t to i64
  %px = getelementptr inbounds double, ptr %x, i64 %i
  %v = load double, ptr %px, align 8
  %sa = getelementptr inbounds [64 x double], ptr addrspace(3) @tile_a, i64 0, i64 %i
  store double %v, ptr addrspace(3) %sa, align 8
  tail call void @llvm.nvvm.barrier0()
  %w = load double, ptr addrspace(3) %sa, align 8
  %m = fmul double %w, %w
  %py = getelementptr inbounds double, ptr %y, i64 %i
  store double %m, ptr %py, align 8
  ret void
}

define void @kb(ptr %x, ptr %y) {
entry:
  %t = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %i = zext i32 %t to i64
  %px = getelementptr inbounds double, ptr %x, i64 %i
  %v = load double, ptr %px, align 8
  %sb = getelementptr inbounds [512 x double], ptr addrspace(3) @tile_b, i64 0, i64 %i
  store double %v, ptr addrspace(3) %sb, align 8
  tail call void @llvm.nvvm.barrier0()
  %w = load double, ptr addrspace(3) %sb, align 8
  %m = fmul double %w, %v
  %py = getelementptr inbounds double, ptr %y, i64 %i
  store double %m, ptr %py, align 8
  ret void
}

define void @grads(ptr %x, ptr %dx, ptr %y, ptr %dy) {
entry:
  tail call void (...) @__enzyme_autodiff(ptr @ka, ptr %x, ptr %dx, ptr %y, ptr %dy)
  tail call void (...) @__enzyme_autodiff(ptr @kb, ptr %x, ptr %dx, ptr %y, ptr %dy)
  ret void
}

declare void @__enzyme_autodiff(...)

declare void @llvm.nvvm.barrier0()

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

; CHECK: @tile_a_shadow = internal unnamed_addr addrspace(3) global [64 x double] undef, align 8
; CHECK: @tile_b_shadow = internal unnamed_addr addrspace(3) global [512 x double] undef, align 8

; CHECK-LABEL: define internal void @diffeka(ptr {{.*}}%x, ptr {{.*}}%"x'", ptr {{.*}}%y, ptr {{.*}}%"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[TX:.+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
; CHECK-NEXT:   %[[TY:.+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
; CHECK-NEXT:   %[[TZ:.+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.z()
; CHECK-NEXT:   %[[TXY:.+]] = or i32 %[[TX]], %[[TY]]
; CHECK-NEXT:   %[[TXYZ:.+]] = or i32 %[[TXY]], %[[TZ]]
; CHECK-NEXT:   %[[LEADER:.+]] = icmp eq i32 %[[TXYZ]], 0
; CHECK-NEXT:   br i1 %[[LEADER]], label %shblock, label %invertentry
; CHECK-EMPTY:
; CHECK-NEXT: shblock:
; CHECK-NEXT:   store [64 x double] zeroinitializer, ptr addrspace(3) @tile_a_shadow, align 8
; CHECK-NEXT:   br label %invertentry
; CHECK-EMPTY:
; CHECK-NEXT: invertentry:
; CHECK-NEXT:   call void @llvm.nvvm.barrier{{0\(\)|\.cta\.sync\.aligned\.all\(i32 0\)}}
; CHECK-NEXT:   %i = zext i32 %[[TX]] to i64
; CHECK-NEXT:   %"px'ipg" = getelementptr inbounds {{double|\[8 x i8\]}}, ptr %"x'", i64 %i
; CHECK-NEXT:   %px = getelementptr inbounds double, ptr %x, i64 %i
; CHECK-NEXT:   %v = load double, ptr %px, align 8
; CHECK-NEXT:   %"sa'ipg" = getelementptr inbounds {{\[64 x double\], ptr addrspace\(3\) @tile_a_shadow, i64 0|\[8 x i8\], ptr addrspace\(3\) @tile_a_shadow}}, i64 %i
; CHECK-NEXT:   %sa = getelementptr inbounds [64 x double], ptr addrspace(3) @tile_a, i64 0, i64 %i
; CHECK-NEXT:   store double %v, ptr addrspace(3) %sa, align 8
; CHECK-NEXT:   call void @llvm.nvvm.barrier{{0\(\)|\.cta\.sync\.aligned\.all\(i32 0\)}}
; CHECK-NEXT:   %w = load double, ptr addrspace(3) %sa, align 8
; CHECK-NEXT:   %m = fmul double %w, %w
; CHECK-NEXT:   %"py'ipg" = getelementptr inbounds {{double|\[8 x i8\]}}, ptr %"y'", i64 %i
; CHECK-NEXT:   %py = getelementptr inbounds double, ptr %y, i64 %i
; CHECK-NEXT:   store double %m, ptr %py, align 8
; CHECK-NEXT:   %[[DM:.+]] = load double, ptr %"py'ipg", align 8
; CHECK-NEXT:   store double 0.000000e+00, ptr %"py'ipg", align 8
; CHECK-NEXT:   %[[DWHALF:.+]] = fmul fast double %[[DM]], %w
; CHECK-NEXT:   %[[DW:.+]] = fadd fast double %[[DWHALF]], %[[DWHALF]]
; CHECK-NEXT:   %{{.+}} = atomicrmw fadd ptr addrspace(3) %"sa'ipg", double %[[DW]] monotonic, align 8
; CHECK-NEXT:   call void @llvm.nvvm.barrier{{0\(\)|\.cta\.sync\.aligned\.all\(i32 0\)}}
; CHECK-NEXT:   %[[DSA:.+]] = load double, ptr addrspace(3) %"sa'ipg", align 8
; CHECK-NEXT:   store double 0.000000e+00, ptr addrspace(3) %"sa'ipg", align 8
; CHECK-NEXT:   %{{.+}} = atomicrmw fadd ptr %"px'ipg", double %[[DSA]] monotonic, align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK-LABEL: define internal void @diffekb(ptr {{.*}}%x, ptr {{.*}}%"x'", ptr {{.*}}%y, ptr {{.*}}%"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[TX:.+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
; CHECK-NEXT:   %[[TY:.+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
; CHECK-NEXT:   %[[TZ:.+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.z()
; CHECK-NEXT:   %[[TXY:.+]] = or i32 %[[TX]], %[[TY]]
; CHECK-NEXT:   %[[TXYZ:.+]] = or i32 %[[TXY]], %[[TZ]]
; CHECK-NEXT:   %[[LEADER:.+]] = icmp eq i32 %[[TXYZ]], 0
; CHECK-NEXT:   br i1 %[[LEADER]], label %shblock, label %invertentry
; CHECK-EMPTY:
; CHECK-NEXT: shblock:
; CHECK-NEXT:   store [512 x double] zeroinitializer, ptr addrspace(3) @tile_b_shadow, align 8
; CHECK-NEXT:   br label %invertentry
; CHECK-EMPTY:
; CHECK-NEXT: invertentry:
; CHECK-NEXT:   call void @llvm.nvvm.barrier{{0\(\)|\.cta\.sync\.aligned\.all\(i32 0\)}}
; CHECK-NEXT:   %i = zext i32 %[[TX]] to i64
; CHECK-NEXT:   %"px'ipg" = getelementptr inbounds {{double|\[8 x i8\]}}, ptr %"x'", i64 %i
; CHECK-NEXT:   %px = getelementptr inbounds double, ptr %x, i64 %i
; CHECK-NEXT:   %v = load double, ptr %px, align 8
; CHECK-NEXT:   %"sb'ipg" = getelementptr inbounds {{\[512 x double\], ptr addrspace\(3\) @tile_b_shadow, i64 0|\[8 x i8\], ptr addrspace\(3\) @tile_b_shadow}}, i64 %i
; CHECK-NEXT:   %sb = getelementptr inbounds [512 x double], ptr addrspace(3) @tile_b, i64 0, i64 %i
; CHECK-NEXT:   store double %v, ptr addrspace(3) %sb, align 8
; CHECK-NEXT:   call void @llvm.nvvm.barrier{{0\(\)|\.cta\.sync\.aligned\.all\(i32 0\)}}
; CHECK-NEXT:   %w = load double, ptr addrspace(3) %sb, align 8
; CHECK-NEXT:   %m = fmul double %w, %v
; CHECK-NEXT:   %"py'ipg" = getelementptr inbounds {{double|\[8 x i8\]}}, ptr %"y'", i64 %i
; CHECK-NEXT:   %py = getelementptr inbounds double, ptr %y, i64 %i
; CHECK-NEXT:   store double %m, ptr %py, align 8
; CHECK-NEXT:   %[[DM:.+]] = load double, ptr %"py'ipg", align 8
; CHECK-NEXT:   store double 0.000000e+00, ptr %"py'ipg", align 8
; CHECK-NEXT:   %[[DW:.+]] = fmul fast double %[[DM]], %v
; CHECK-NEXT:   %[[DVM:.+]] = fmul fast double %[[DM]], %w
; CHECK-NEXT:   %{{.+}} = atomicrmw fadd ptr addrspace(3) %"sb'ipg", double %[[DW]] monotonic, align 8
; CHECK-NEXT:   call void @llvm.nvvm.barrier{{0\(\)|\.cta\.sync\.aligned\.all\(i32 0\)}}
; CHECK-NEXT:   %[[DSB:.+]] = load double, ptr addrspace(3) %"sb'ipg", align 8
; CHECK-NEXT:   store double 0.000000e+00, ptr addrspace(3) %"sb'ipg", align 8
; CHECK-NEXT:   %[[DV:.+]] = fadd fast double %[[DVM]], %[[DSB]]
; CHECK-NEXT:   %{{.+}} = atomicrmw fadd ptr %"px'ipg", double %[[DV]] monotonic, align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
