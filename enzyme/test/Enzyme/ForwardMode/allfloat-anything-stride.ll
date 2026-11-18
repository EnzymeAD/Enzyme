; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -enzyme-preopt=false -S | %lli - | FileCheck %s --check-prefix=EXEC; fi

; TypeTree::allFloat(..., anythingIsFloat=true) must advance bytewise across
; Anything entries rather than by the size of an earlier float field.
; Otherwise, an Anything byte after a float skips trailing Integer/Pointer
; fields, causing GradientUtils::invertPointerM to zero out the entire shadow.
; In addition, the byte-clearing fallback in invertPointerM must align float
; zero stores to the alloca/offset alignment instead of using default ABI
; alignment on an unaligned offset.

declare void @__enzyme_fwddiff(...)
declare i32 @printf(ptr, ...)

@enzyme_const = external global i32
@.fmt = private unnamed_addr constant [37 x i8] c"struct=%g,%d arr=%d i128=%d allf=%g\0A\00", align 1

; 1. { double, i8 (Anything), i32 (Integer) }: offset 8 is Anything, offset 12 is Integer.
define void @copy_struct(
    ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Anything, [-1,9]:Anything, [-1,10]:Anything, [-1,11]:Anything, [-1,12]:Integer, [-1,13]:Integer, [-1,14]:Integer, [-1,15]:Integer}" %src,
    ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Anything, [-1,9]:Anything, [-1,10]:Anything, [-1,11]:Anything, [-1,12]:Integer, [-1,13]:Integer, [-1,14]:Integer, [-1,15]:Integer}" %dst,
    ptr %fdst,
    double %x) {
entry:
  %v = load { double, i8, i32 }, ptr %src, align 8
  store { double, i8, i32 } %v, ptr %dst, align 8
  store double %x, ptr %fdst, align 8
  ret void
}

; 2. [8 x i8] with Anything at 0..1, Float at 2..5 (unaligned!), Anything at 6, Integer at 7.
define void @copy_offset_anything(
    ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Anything, [-1,1]:Anything, [-1,2]:Float@float, [-1,6]:Anything, [-1,7]:Integer}" %src,
    ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Anything, [-1,1]:Anything, [-1,2]:Float@float, [-1,6]:Anything, [-1,7]:Integer}" %dst,
    ptr %fdst,
    double %x) {
entry:
  %v = load [8 x i8], ptr %src, align 1
  store [8 x i8] %v, ptr %dst, align 1
  store double %x, ptr %fdst, align 8
  ret void
}

; 3. ConstantInt i128: lower 64 bits = double pi, bytes 8..11 = Anything, bytes 12..15 = Integer (42).
; (42 << 96) | 0x400921fb54442d18 = 3327582825603716435585397959960
define void @store_const_i128(
    ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Anything, [-1,9]:Anything, [-1,10]:Anything, [-1,11]:Anything, [-1,12]:Integer, [-1,13]:Integer, [-1,14]:Integer, [-1,15]:Integer}" %dst,
    ptr %fdst,
    double %x) {
entry:
  store i128 3327582825603716435585397959960, ptr %dst, align 16
  store double %x, ptr %fdst, align 8
  ret void
}

; 4. Positive case: { double, i8 } with Double at 0..7 and Anything at 8..15 still folds to zeroinitializer.
define void @copy_all_float_anything(
    ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Anything, [-1,9]:Anything, [-1,10]:Anything, [-1,11]:Anything, [-1,12]:Anything, [-1,13]:Anything, [-1,14]:Anything, [-1,15]:Anything}" %src,
    ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Anything, [-1,9]:Anything, [-1,10]:Anything, [-1,11]:Anything, [-1,12]:Anything, [-1,13]:Anything, [-1,14]:Anything, [-1,15]:Anything}" %dst,
    ptr %fdst,
    double %x) {
entry:
  %v = load { double, i8 }, ptr %src, align 8
  store { double, i8 } %v, ptr %dst, align 8
  store double %x, ptr %fdst, align 8
  ret void
}

define i32 @main() {
entry:
  %fdst = alloca double, align 8
  %dfdst = alloca double, align 8

  ; 1. copy_struct
  %s_src = alloca { double, i8, i32 }, align 8
  %s_dst = alloca { double, i8, i32 }, align 8
  %s_ddst = alloca { double, i8, i32 }, align 8
  store { double, i8, i32 } { double 3.500000e+00, i8 7, i32 42 }, ptr %s_src, align 8
  call void (...) @__enzyme_fwddiff(ptr @copy_struct, ptr @enzyme_const, ptr %s_src, ptr %s_dst, ptr %s_ddst, ptr %fdst, ptr %dfdst, double 1.0, double 1.0)
  %s_val = load { double, i8, i32 }, ptr %s_ddst, align 8
  %s_d = extractvalue { double, i8, i32 } %s_val, 0
  %s_i = extractvalue { double, i8, i32 } %s_val, 2

  ; 2. copy_offset_anything
  %a_src = alloca [8 x i8], align 1
  %a_dst = alloca [8 x i8], align 1
  %a_ddst = alloca [8 x i8], align 1
  store [8 x i8] [i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 99], ptr %a_src, align 1
  call void (...) @__enzyme_fwddiff(ptr @copy_offset_anything, ptr @enzyme_const, ptr %a_src, ptr %a_dst, ptr %a_ddst, ptr %fdst, ptr %dfdst, double 1.0, double 1.0)
  %a_val = load [8 x i8], ptr %a_ddst, align 1
  %a_b7 = extractvalue [8 x i8] %a_val, 7
  %a_i = zext i8 %a_b7 to i32

  ; 3. store_const_i128
  %c_dst = alloca i128, align 16
  %c_ddst = alloca i128, align 16
  call void (...) @__enzyme_fwddiff(ptr @store_const_i128, ptr %c_dst, ptr %c_ddst, ptr %fdst, ptr %dfdst, double 1.0, double 1.0)
  %c_val = load i128, ptr %c_ddst, align 16
  %c_hi = lshr i128 %c_val, 96
  %c_i = trunc i128 %c_hi to i32

  ; 4. copy_all_float_anything
  %f_src = alloca { double, i8 }, align 8
  %f_dst = alloca { double, i8 }, align 8
  %f_ddst = alloca { double, i8 }, align 8
  store { double, i8 } { double 9.000000e+00, i8 5 }, ptr %f_src, align 8
  call void (...) @__enzyme_fwddiff(ptr @copy_all_float_anything, ptr @enzyme_const, ptr %f_src, ptr %f_dst, ptr %f_ddst, ptr %fdst, ptr %dfdst, double 1.0, double 1.0)
  %f_val = load { double, i8 }, ptr %f_ddst, align 8
  %f_d = extractvalue { double, i8 } %f_val, 0

  call i32 (ptr, ...) @printf(ptr @.fmt, double %s_d, i32 %s_i, i32 %a_i, i32 %c_i, double %f_d)
  ret i32 0
}

; EXEC: struct=0,42 arr=99 i128=42 allf=0

; CHECK-LABEL: define internal void @fwddiffecopy_struct(
; CHECK-NOT: store { double, i8, i32 } zeroinitializer
; CHECK: store double 0.000000e+00, ptr %{{.+}}, align 8
; CHECK: store i8 0, ptr %{{.+}}, align 1
; CHECK: store { double, i8, i32 } %{{.+}}, ptr %"dst'"

; CHECK-LABEL: define internal void @fwddiffecopy_offset_anything(
; CHECK-NOT: store [8 x i8] zeroinitializer
; CHECK: store float 0.000000e+00, ptr %{{.+}}, align 1
; CHECK: store [8 x i8] %{{.+}}, ptr %"dst'"

; CHECK-LABEL: define internal void @fwddiffestore_const_i128(
; CHECK: store i128 3327582825599102178928845914112, ptr %"dst'", align 16

; CHECK-LABEL: define internal void @fwddiffecopy_all_float_anything(
; CHECK: store { double, i8 } zeroinitializer, ptr %"dst'", align 8
