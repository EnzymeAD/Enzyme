; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify),verify" -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,verify" -S | %lli - | FileCheck %s --check-prefix=EVAL; fi
; EVAL: 6.500000

; Verify that unoptimized (-O0) read-only accessor methods that spill their
; `this` pointer into a local non-escaping stack slot before returning a pointer
; derived from an `enzyme_const` span/view argument are recognized as returning
; inactive memory, without duplicating the accessor or mutating the constant
; buffer.

%struct.Span = type { ptr, i32 }

@.fmt = private unnamed_addr constant [4 x i8] c"%f\0A\00", align 1

declare i32 @printf(ptr, ...)
declare double @__enzyme_autodiff(...)

define linkonce_odr noundef nonnull align 8 dereferenceable(8) ptr @span_at(ptr noundef nonnull align 8 dereferenceable(16) %this, i32 noundef %idx) noinline readonly {
entry:
  %this.addr = alloca ptr, align 8
  %idx.addr = alloca i32, align 4
  store ptr %this, ptr %this.addr, align 8
  store i32 %idx, ptr %idx.addr, align 4
  %self = load ptr, ptr %this.addr, align 8
  %data_slot = getelementptr inbounds %struct.Span, ptr %self, i32 0, i32 0
  %data = load ptr, ptr %data_slot, align 8
  %i = load i32, ptr %idx.addr, align 4
  %i64 = sext i32 %i to i64
  %elem = getelementptr inbounds double, ptr %data, i64 %i64
  ret ptr %elem
}

define void @scale_from_const_span(ptr noundef nonnull align 8 dereferenceable(16) %weights, ptr noundef nonnull align 8 dereferenceable(16) %out, double noundef %scale) {
entry:
  %scale.addr = alloca double, align 8
  %i = alloca i32, align 4
  store double %scale, ptr %scale.addr, align 8
  store i32 0, ptr %i, align 4
  br label %loop.cond

loop.cond:
  %cur = load i32, ptr %i, align 4
  %n_slot = getelementptr inbounds %struct.Span, ptr %out, i32 0, i32 1
  %n = load i32, ptr %n_slot, align 8
  %cmp = icmp slt i32 %cur, %n
  br i1 %cmp, label %loop.body, label %loop.end

loop.body:
  %idx0 = load i32, ptr %i, align 4
  %w_ptr = call noundef nonnull align 8 dereferenceable(8) ptr @span_at(ptr noundef nonnull align 8 dereferenceable(16) %weights, i32 noundef %idx0)
  %w = load double, ptr %w_ptr, align 8
  %s = load double, ptr %scale.addr, align 8
  %prod = fmul double %s, %w
  %idx1 = load i32, ptr %i, align 4
  %o_ptr = call noundef nonnull align 8 dereferenceable(8) ptr @span_at(ptr noundef nonnull align 8 dereferenceable(16) %out, i32 noundef %idx1)
  store double %prod, ptr %o_ptr, align 8
  %next = add nsw i32 %idx1, 1
  store i32 %next, ptr %i, align 4
  br label %loop.cond

loop.end:
  ret void
}

define double @grad_scale(ptr %w_span, ptr %out_span, ptr %dout_span, double %scale) {
entry:
  %ds = call double (...) @__enzyme_autodiff(ptr @scale_from_const_span, metadata !"enzyme_const", ptr %w_span, metadata !"enzyme_dupnoneed", ptr %out_span, ptr %dout_span, double %scale)
  ret double %ds
}

define i32 @main() {
entry:
  %w_buf = alloca [4 x double], align 16
  %out_buf = alloca [4 x double], align 16
  %dout_buf = alloca [4 x double], align 16
  %w_span = alloca %struct.Span, align 8
  %out_span = alloca %struct.Span, align 8
  %dout_span = alloca %struct.Span, align 8

  %w0 = getelementptr inbounds [4 x double], ptr %w_buf, i64 0, i64 0
  %w1 = getelementptr inbounds [4 x double], ptr %w_buf, i64 0, i64 1
  %w2 = getelementptr inbounds [4 x double], ptr %w_buf, i64 0, i64 2
  %w3 = getelementptr inbounds [4 x double], ptr %w_buf, i64 0, i64 3
  store double 2.000000e+00, ptr %w0, align 8
  store double 5.000000e-01, ptr %w1, align 8
  store double 1.000000e+00, ptr %w2, align 8
  store double 3.000000e+00, ptr %w3, align 8

  %d0 = getelementptr inbounds [4 x double], ptr %dout_buf, i64 0, i64 0
  %d1 = getelementptr inbounds [4 x double], ptr %dout_buf, i64 0, i64 1
  %d2 = getelementptr inbounds [4 x double], ptr %dout_buf, i64 0, i64 2
  %d3 = getelementptr inbounds [4 x double], ptr %dout_buf, i64 0, i64 3
  store double 1.000000e+00, ptr %d0, align 8
  store double 1.000000e+00, ptr %d1, align 8
  store double 1.000000e+00, ptr %d2, align 8
  store double 1.000000e+00, ptr %d3, align 8

  store ptr %w0, ptr %w_span, align 8
  %wn = getelementptr inbounds %struct.Span, ptr %w_span, i32 0, i32 1
  store i32 4, ptr %wn, align 8

  store ptr %out_buf, ptr %out_span, align 8
  %on = getelementptr inbounds %struct.Span, ptr %out_span, i32 0, i32 1
  store i32 4, ptr %on, align 8

  store ptr %dout_buf, ptr %dout_span, align 8
  %dn = getelementptr inbounds %struct.Span, ptr %dout_span, i32 0, i32 1
  store i32 4, ptr %dn, align 8

  %grad = call double @grad_scale(ptr %w_span, ptr %out_span, ptr %dout_span, double 2.000000e+00)
  call i32 (ptr, ...) @printf(ptr @.fmt, double %grad)
  ret i32 0
}

; CHECK-LABEL: define internal { double } @diffescale_from_const_span(
; CHECK-NOT: @augmented_span_at(ptr %weights
; CHECK: %w_ptr = call {{.*}}ptr @{{(nofree_)?}}span_at(ptr {{.*}}%weights
; CHECK: call {{.*}}@augmented_span_at(ptr {{.*}}%out, ptr {{.*}}%"out'"
; CHECK: ret { double }
