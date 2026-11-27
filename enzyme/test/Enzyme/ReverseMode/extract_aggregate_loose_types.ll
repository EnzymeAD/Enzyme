; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme-loose-types=1 -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -enzyme-loose-types=1 -passes="enzyme" -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -enzyme-loose-types=1 -passes="enzyme,default<O3>" | %lli -

%struct.IndexedTensor = type { i32, [2 x [3 x double]] }
%struct.TensorPair = type { %struct.IndexedTensor, [2 x double] }

define %struct.TensorPair @produce_tensor(double %x, i64 %index) noinline {
entry:
  %buffer = alloca %struct.TensorPair, align 8
  %cond = icmp eq i64 %index, 0
  br i1 %cond, label %active, label %done

active:
  %v0 = fmul double %x, 5.000000e+00
  %v1 = fmul double %x, 6.000000e+00
  %p0 = getelementptr inbounds %struct.TensorPair, ptr %buffer, i32 0, i32 0, i32 1, i32 1, i32 2
  store double %v0, ptr %p0, align 8
  %p1 = getelementptr inbounds %struct.TensorPair, ptr %buffer, i32 0, i32 1, i32 0
  store double %v1, ptr %p1, align 8
  br label %done

done:
  %result = load %struct.TensorPair, ptr %buffer, align 8
  ret %struct.TensorPair %result
}

define double @tensor_roundtrip(double %x, i64 %index) {
entry:
  %pair = call %struct.TensorPair @produce_tensor(double %x, i64 %index)
  %tensor = extractvalue %struct.TensorPair %pair, 0
  %matrix = extractvalue %struct.IndexedTensor %tensor, 1
  %tail = extractvalue %struct.TensorPair %pair, 1
  %rebuilt_tensor = insertvalue %struct.IndexedTensor zeroinitializer, [2 x [3 x double]] %matrix, 1
  %rebuilt_pair0 = insertvalue %struct.TensorPair zeroinitializer, %struct.IndexedTensor %rebuilt_tensor, 0
  %rebuilt_pair1 = insertvalue %struct.TensorPair %rebuilt_pair0, [2 x double] %tail, 1
  %elem_a = extractvalue %struct.TensorPair %rebuilt_pair1, 0, 1, 1, 2
  %elem_b = extractvalue %struct.TensorPair %rebuilt_pair1, 1, 0
  %sum = fadd double %elem_a, %elem_b
  ret double %sum
}

declare double @__enzyme_autodiff(...)

define i32 @main() {
entry:
  %grad = call double (...) @__enzyme_autodiff(ptr @tensor_roundtrip, double 2.000000e+00, i64 0)
  %err = fsub double %grad, 1.100000e+01
  %neg = fsub double 0.000000e+00, %err
  %isneg = fcmp olt double %err, 0.000000e+00
  %abs = select i1 %isneg, double %neg, double %err
  %ok = fcmp olt double %abs, 1.000000e-09
  %ret = select i1 %ok, i32 0, i32 1
  ret i32 %ret
}

; CHECK-LABEL: define internal { double } @diffetensor_roundtrip(
; CHECK: call { double } @diffeproduce_tensor(
