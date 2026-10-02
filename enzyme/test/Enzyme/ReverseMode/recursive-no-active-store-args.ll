; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -enzyme-detect-recursive-no-active-store -enzyme-print-recursive-no-active-store -passes="enzyme" -S 2>&1 | FileCheck %s; fi

; A pointer argument has the recursive no-active-store property if nothing is
; captured through it, nor through any pointer loaded (transitively) from the
; memory it points to, and no active value is ever stored through these
; pointers: stores of constants, stores marked enzyme_inactive, stores of
; integers (by TBAA) and memsets are fine, and so are calls that are known
; inactive and do not capture the pointer, or that receive it as a parameter
; with the same property.

; Reads a double (typed as such by TBAA) through the data pointer of a
; vector-like header.
define double @read_nested(ptr %hdr, i64 %i) {
  %data = load ptr, ptr %hdr, align 8
  %p = getelementptr inbounds double, ptr %data, i64 %i
  %v = load double, ptr %p, align 8, !tbaa !5
  ret double %v
}

; The same without TBAA: the loaded bits might be a pointer into the memory,
; so returning them counts as a capture.
define double @read_untyped_nested(ptr %hdr, i64 %i) {
  %data = load ptr, ptr %hdr, align 8
  %p = getelementptr inbounds double, ptr %data, i64 %i
  %v = load double, ptr %p, align 8
  ret double %v
}

; Loads untyped bits, turns them back into a pointer after some arithmetic,
; and stores through it.
define void @load_bits_store_through_nested(ptr %hdr, double %v) {
  %data = load ptr, ptr %hdr, align 8
  %bits = load i64, ptr %data, align 8
  %off = add i64 %bits, 8
  %q = inttoptr i64 %off to ptr
  store double %v, ptr %q, align 8
  ret void
}

; Writes through the loaded data pointer, like det_of_minor's r[n] = r[R0].
define void @write_nested(ptr %hdr, i64 %i) {
  %data = load ptr, ptr %hdr, align 8
  %p = getelementptr inbounds i64, ptr %data, i64 %i
  store i64 0, ptr %p, align 8
  ret void
}

; Stores the loaded data pointer somewhere else.
define void @capture_nested(ptr %hdr, ptr %out) {
  %data = load ptr, ptr %hdr, align 8
  store ptr %data, ptr %out, align 8
  ret void
}

; Recursion that only reads through the header.
define double @recurse_read(ptr %hdr, i64 %n) {
entry:
  %c = icmp eq i64 %n, 0
  br i1 %c, label %base, label %rec

base:
  %v = call double @read_nested(ptr %hdr, i64 0)
  ret double %v

rec:
  %m = add i64 %n, -1
  %r = call double @recurse_read(ptr %hdr, i64 %m)
  ret double %r
}

; Mutual recursion where one side writes through the nested pointer.
define void @even(ptr %hdr, i64 %n) {
  %c = icmp eq i64 %n, 0
  br i1 %c, label %done, label %rec

rec:
  %m = add i64 %n, -1
  call void @odd(ptr %hdr, i64 %m)
  br label %done

done:
  ret void
}

define void @odd(ptr %hdr, i64 %n) {
  call void @write_nested(ptr %hdr, i64 %n)
  call void @even(ptr %hdr, i64 %n)
  ret void
}

; Stores an integer derived from a pointer through the loaded data pointer:
; the pointer's bits are followed through the arithmetic, so the store of the
; integer is a capture of %q.
define void @store_ptrtoint_nested(ptr %hdr, ptr %q) {
  %data = load ptr, ptr %hdr, align 8
  %bits = ptrtoint ptr %q to i64
  %sum = add i64 %bits, 8
  store i64 %sum, ptr %data, align 8
  ret void
}

; Stores a non-zero double through the loaded data pointer.
define void @store_double_nested(ptr %hdr, double %v) {
  %data = load ptr, ptr %hdr, align 8
  store double %v, ptr %data, align 8
  ret void
}

; Stores a zero double through the loaded data pointer.
define void @store_zero_nested(ptr %hdr) {
  %data = load ptr, ptr %hdr, align 8
  store double 0.000000e+00, ptr %data, align 8
  ret void
}

; Stores the bits of a double as an i64, but with an integer TBAA tag. TBAA
; is trusted, as type analysis does, so this counts as an integer store.
define void @store_double_bits_nested(ptr %hdr, double %v) {
  %data = load ptr, ptr %hdr, align 8
  %bits = bitcast double %v to i64
  store i64 %bits, ptr %data, align 8, !tbaa !0
  ret void
}

; Stores an index loaded from the same memory, with an integer TBAA type.
define void @store_loaded_int_nested(ptr %hdr, i64 %i) {
  %data = load ptr, ptr %hdr, align 8
  %p = getelementptr inbounds i64, ptr %data, i64 %i
  %v = load i64, ptr %p, align 8, !tbaa !0
  store i64 %v, ptr %data, align 8, !tbaa !0
  ret void
}

; The same store without TBAA: the loaded i64 might be a pointer's bits (or a
; moved double), so the store is rejected, as a possibly active store through
; %data or as a capture of the loaded value, whichever use is seen first.
define void @store_loaded_int_notbaa_nested(ptr %hdr, i64 %i) {
  %data = load ptr, ptr %hdr, align 8
  %p = getelementptr inbounds i64, ptr %data, i64 %i
  %v = load i64, ptr %p, align 8
  store i64 %v, ptr %data, align 8
  ret void
}

; Stores a non-zero constant double: a constant carries no derivative.
define void @store_const_double_nested(ptr %hdr) {
  %data = load ptr, ptr %hdr, align 8
  store double 2.500000e+00, ptr %data, align 8
  ret void
}

; Stores an active double, but the store is marked inactive.
define void @store_marked_inactive_nested(ptr %hdr, double %v) {
  %data = load ptr, ptr %hdr, align 8
  store double %v, ptr %data, align 8, !enzyme_inactive !4
  ret void
}

; Passes the loaded data pointer to a known-inactive function that does not
; capture it, and to one that might.
declare void @inactive_sink(ptr nocapture) "enzyme_inactive"
declare void @inactive_keeper(ptr) "enzyme_inactive"
declare void @active_sink(ptr nocapture)

define void @call_inactive_nested(ptr %hdr) {
  %data = load ptr, ptr %hdr, align 8
  call void @inactive_sink(ptr %data)
  ret void
}

define void @call_inactive_capturing_nested(ptr %hdr) {
  %data = load ptr, ptr %hdr, align 8
  call void @inactive_keeper(ptr %data)
  ret void
}

define void @call_active_nested(ptr %hdr) {
  %data = load ptr, ptr %hdr, align 8
  call void @active_sink(ptr %data)
  ret void
}

; Fills the memory with a runtime byte.
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)
define void @memset_nested(ptr %hdr, i8 %b, i64 %n) {
  %data = load ptr, ptr %hdr, align 8
  call void @llvm.memset.p0.i64(ptr %data, i8 %b, i64 %n, i1 false)
  ret void
}

; Loads untyped bits that an enzyme_type annotation marks as integers, and
; stores them back under an all-integer enzyme_truetype annotation.
define void @annotated_int_nested(ptr %hdr) {
  %data = load ptr, ptr %hdr, align 8
  %v = load i64, ptr %data, align 8, !enzyme_type !10
  store i64 %v, ptr %data, align 8, !enzyme_truetype !11
  ret void
}

; Stores a call result whose enzyme_type return attribute says integer.
declare "enzyme_type"="{[-1]:Integer}" i64 @make_int()
define void @store_annotated_call_nested(ptr %hdr) {
  %data = load ptr, ptr %hdr, align 8
  %v = call i64 @make_int()
  store i64 %v, ptr %data, align 8
  ret void
}

; A store whose enzyme_truetype annotation says the bytes are a double.
define void @store_truetype_float_nested(ptr %hdr, i64 %bits) {
  %data = load ptr, ptr %hdr, align 8
  store i64 %bits, ptr %data, align 8, !enzyme_truetype !12
  ret void
}

; A memcpy of integers (by annotation) into the memory.
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
define void @memcpy_int_nested(ptr %hdr, ptr %src) {
  %data = load ptr, ptr %hdr, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr %data, ptr %src, i64 16, i1 false), !enzyme_truetype !11
  ret void
}

define double @f(double %x) {
  ret double %x
}

define double @df(double %x) {
  %r = call double (...) @__enzyme_autodiff(ptr @f, double %x)
  ret double %r
}

declare double @__enzyme_autodiff(...)

!0 = !{!1, !1, i64 0}
!1 = !{!"long", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C++ TBAA"}
!5 = !{!6, !6, i64 0}
!6 = !{!"double", !2, i64 0}
!10 = !{!"Unknown", i32 -1, !13}
!13 = !{!"Integer"}
!11 = !{!"Integer", i64 0, !"Integer", i64 8}
!12 = !{!"Float@double", i64 0}
!4 = !{}

; CHECK-DAG: recursive no active store: read_nested arg 0 (hdr): yes
; CHECK-DAG: recursive no active store: write_nested arg 0 (hdr): yes
; CHECK-DAG: recursive no active store: capture_nested arg 0 (hdr): no, captures a pointer loaded from the argument's memory:   store ptr %data, ptr %out, align 8
; CHECK-DAG: recursive no active store: capture_nested arg 1 (out): no, may store an active value into the memory the argument points to:   store ptr %data, ptr %out, align 8
; CHECK-DAG: recursive no active store: recurse_read arg 0 (hdr): yes
; CHECK-DAG: recursive no active store: even arg 0 (hdr): yes
; CHECK-DAG: recursive no active store: odd arg 0 (hdr): yes
; CHECK-DAG: recursive no active store: store_ptrtoint_nested arg 0 (hdr): no, may store an active value through a pointer loaded from the argument's memory:   store i64 %sum, ptr %data, align 8
; CHECK-DAG: recursive no active store: store_ptrtoint_nested arg 1 (q): no, captures the argument:   store i64 %sum, ptr %data, align 8
; CHECK-DAG: recursive no active store: store_double_nested arg 0 (hdr): no, may store an active value through a pointer loaded from the argument's memory:   store double %v, ptr %data, align 8
; CHECK-DAG: recursive no active store: store_zero_nested arg 0 (hdr): yes
; CHECK-DAG: recursive no active store: store_double_bits_nested arg 0 (hdr): yes
; CHECK-DAG: recursive no active store: store_loaded_int_nested arg 0 (hdr): yes
; CHECK-DAG: recursive no active store: store_loaded_int_notbaa_nested arg 0 (hdr): no, {{(may store an active value through|captures) a pointer loaded from the argument's memory}}:   store i64 %v, ptr %data, align 8

; CHECK-DAG: define void @write_nested(ptr {{.*}}"enzyme_RecursiveNoActiveStore" %hdr, i64 %i)
; CHECK-DAG: define void @store_double_nested(ptr {{[^"]*}}%hdr, double %v)
; CHECK-DAG: recursive no active store: store_const_double_nested arg 0 (hdr): yes
; CHECK-DAG: recursive no active store: store_marked_inactive_nested arg 0 (hdr): yes
; CHECK-DAG: recursive no active store: call_inactive_nested arg 0 (hdr): yes
; CHECK-DAG: recursive no active store: call_inactive_capturing_nested arg 0 (hdr): no, passes it to a call that is neither inactive nor proven to make no active store through the parameter:   call void @inactive_keeper(ptr %data)
; CHECK-DAG: recursive no active store: call_active_nested arg 0 (hdr): no, passes it to a call that is neither inactive nor proven to make no active store through the parameter:   call void @active_sink(ptr %data)
; CHECK-DAG: recursive no active store: memset_nested arg 0 (hdr): yes
; CHECK-DAG: recursive no active store: read_untyped_nested arg 0 (hdr): no, captures a pointer loaded from the argument's memory:   ret double %v
; CHECK-DAG: recursive no active store: load_bits_store_through_nested arg 0 (hdr): no, may store an active value through a pointer loaded from the argument's memory:   store double %v, ptr %q, align 8
; CHECK-DAG: recursive no active store: annotated_int_nested arg 0 (hdr): yes
; CHECK-DAG: recursive no active store: store_annotated_call_nested arg 0 (hdr): yes
; CHECK-DAG: recursive no active store: store_truetype_float_nested arg 0 (hdr): no, may store an active value through a pointer loaded from the argument's memory:   store i64 %bits, ptr %data, align 8, !enzyme_truetype !{{[0-9]+}}
; CHECK-DAG: recursive no active store: memcpy_int_nested arg 0 (hdr): yes
; CHECK-DAG: recursive no active store: memcpy_int_nested arg 1 (src): yes
