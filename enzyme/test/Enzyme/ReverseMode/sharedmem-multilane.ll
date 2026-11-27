; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,verify" -S | FileCheck %s; fi

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@sh_scalar = internal addrspace(3) global float undef, align 4
@sh_f32 = internal addrspace(3) global [32 x float] undef, align 4
@sh_f64 = internal addrspace(3) global [64 x double] undef, align 8
@sh_odd = internal addrspace(3) global [17 x i8] undef, align 1

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare void @llvm.nvvm.barrier0()
declare void @__enzyme_autodiff(ptr, ...)

define void @kernel(ptr %in, ptr %out) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %idx = zext i32 %tid to i64
  %p_in = getelementptr inbounds double, ptr %in, i64 %idx
  %val = load double, ptr %p_in, align 8
  %valf = fptrunc double %val to float
  store float %valf, ptr addrspace(3) @sh_scalar, align 4
  %s32 = getelementptr inbounds [32 x float], ptr addrspace(3) @sh_f32, i64 0, i64 %idx
  store float %valf, ptr addrspace(3) %s32, align 4
  %s64 = getelementptr inbounds [64 x double], ptr addrspace(3) @sh_f64, i64 0, i64 %idx
  store double %val, ptr addrspace(3) %s64, align 8
  %sodd = getelementptr inbounds [17 x i8], ptr addrspace(3) @sh_odd, i64 0, i64 %idx
  %valf_p = getelementptr inbounds float, ptr addrspace(3) %sodd, i64 0
  store float %valf, ptr addrspace(3) %valf_p, align 1
  call void @llvm.nvvm.barrier0()
  %r_scalar = load float, ptr addrspace(3) @sh_scalar, align 4
  %r_f32 = load float, ptr addrspace(3) %s32, align 4
  %r_f64 = load double, ptr addrspace(3) %s64, align 8
  %r_odd = load float, ptr addrspace(3) %valf_p, align 1
  %sumf0 = fadd float %r_scalar, %r_f32
  %sumf1 = fadd float %sumf0, %r_odd
  %sumd0 = fpext float %sumf1 to double
  %sumd = fadd double %sumd0, %r_f64
  %p_out = getelementptr inbounds double, ptr %out, i64 %idx
  store double %sumd, ptr %p_out, align 8
  ret void
}

define void @caller(ptr %in, ptr %din, ptr %out, ptr %dout) {
entry:
  call void (ptr, ...) @__enzyme_autodiff(ptr @kernel, ptr %in, ptr %din, ptr %out, ptr %dout)
  ret void
}

; CHECK-LABEL: define internal void @diffekernel(
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[TX:.+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
; CHECK-NEXT:   %[[TY:.+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
; CHECK-NEXT:   %[[TZ:.+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.z()
; CHECK-NEXT:   %[[BX:.+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
; CHECK-NEXT:   %[[BY:.+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
; CHECK-NEXT:   %[[BZ:.+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
; CHECK-NEXT:   %[[BYTZ:.+]] = mul i32 %[[BY]], %[[TZ]]
; CHECK-NEXT:   %[[TYZ:.+]] = add i32 %[[TY]], %[[BYTZ]]
; CHECK-NEXT:   %[[BXTYZ:.+]] = mul i32 %[[BX]], %[[TYZ]]
; CHECK-NEXT:   %[[TID:.+]] = add i32 %[[TX]], %[[BXTYZ]]
; CHECK-NEXT:   %[[BXBY:.+]] = mul i32 %[[BX]], %[[BY]]
; CHECK-NEXT:   %[[NTHREADS:.+]] = mul i32 %[[BXBY]], %[[BZ]]
; CHECK-NEXT:   %[[IS0:.+]] = icmp eq i32 %[[TID]], 0
; CHECK-NEXT:   br i1 %[[IS0]], label %shblock, label %shblock.cont

; CHECK: shblock:
; CHECK-NEXT:   store float 0.000000e+00, ptr addrspace(3) @sh_scalar_shadow, align 4
; CHECK-NEXT:   br label %shblock.cont

; CHECK: shblock.cont:
; CHECK-NEXT:   %[[C0:.+]] = icmp ult i32 %[[TID]], 32
; CHECK-NEXT:   br i1 %[[C0]], label %[[L0:.+]], label %[[N0:.+]]

; CHECK: [[L0]]:
; CHECK-NEXT:   %[[I0:.+]] = phi i32 [ %[[TID]], %shblock.cont ], [ %[[I0N:.+]], %[[L0]] ]
; CHECK-NEXT:   %[[P0:.+]] = getelementptr inbounds i32, ptr addrspace(3) @sh_f32_shadow, i32 %[[I0]]
; CHECK-NEXT:   store i32 0, ptr addrspace(3) %[[P0]], align 4
; CHECK-NEXT:   %[[I0N]] = add i32 %[[I0]], %[[NTHREADS]]
; CHECK-NEXT:   %[[CL0:.+]] = icmp ult i32 %[[I0N]], 32
; CHECK-NEXT:   br i1 %[[CL0]], label %[[L0]], label %[[N0]]

; CHECK: [[N0]]:
; CHECK-NEXT:   %[[C1:.+]] = icmp ult i32 %[[TID]], 64
; CHECK-NEXT:   br i1 %[[C1]], label %[[L1:.+]], label %[[N1:.+]]

; CHECK: [[L1]]:
; CHECK-NEXT:   %[[I1:.+]] = phi i32 [ %[[TID]], %[[N0]] ], [ %[[I1N:.+]], %[[L1]] ]
; CHECK-NEXT:   %[[P1:.+]] = getelementptr inbounds i64, ptr addrspace(3) @sh_f64_shadow, i32 %[[I1]]
; CHECK-NEXT:   store i64 0, ptr addrspace(3) %[[P1]], align 8
; CHECK-NEXT:   %[[I1N]] = add i32 %[[I1]], %[[NTHREADS]]
; CHECK-NEXT:   %[[CL1:.+]] = icmp ult i32 %[[I1N]], 64
; CHECK-NEXT:   br i1 %[[CL1]], label %[[L1]], label %[[N1]]

; CHECK: [[N1]]:
; CHECK-NEXT:   %[[C2:.+]] = icmp ult i32 %[[TID]], 17
; CHECK-NEXT:   br i1 %[[C2]], label %[[L2:.+]], label %[[AFTER:.+]]

; CHECK: [[L2]]:
; CHECK-NEXT:   %[[I2:.+]] = phi i32 [ %[[TID]], %[[N1]] ], [ %[[I2N:.+]], %[[L2]] ]
; CHECK-NEXT:   %[[P2:.+]] = getelementptr inbounds i8, ptr addrspace(3) @sh_odd_shadow, i32 %[[I2]]
; CHECK-NEXT:   store i8 0, ptr addrspace(3) %[[P2]], align 1
; CHECK-NEXT:   %[[I2N]] = add i32 %[[I2]], %[[NTHREADS]]
; CHECK-NEXT:   %[[CL2:.+]] = icmp ult i32 %[[I2N]], 17
; CHECK-NEXT:   br i1 %[[CL2]], label %[[L2]], label %[[AFTER]]

; CHECK: [[AFTER]]:
; CHECK-NEXT:   call void @llvm.nvvm.barrier{{0\(\)|\.cta\.sync\.aligned\.all\(i32 0\)}}
