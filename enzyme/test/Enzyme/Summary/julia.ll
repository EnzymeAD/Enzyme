; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="print<enzyme-function-summary>" -disable-output | FileCheck %s; fi

; Function summaries of Julia IR: pointers in addrspace(10)/(11)/(13),
; julia.gc_alloc_obj as an allocation, julia.gc_loaded as a pointer into its
; parent, julia.write_barrier and julia.get_pgcstack as markers, the
; swiftself task pointer as no user memory, objects addressed by constant
; inttoptr as an unnamed global ("*"), and dynamic dispatch (julia.call2).

declare ptr @julia.get_pgcstack()
declare noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr, i64, ptr addrspace(10))
declare void @julia.write_barrier(ptr addrspace(10), ...)
declare ptr addrspace(13) @julia.gc_loaded(ptr addrspace(10), ptr)
declare ptr @julia.pointer_from_objref(ptr addrspace(11))
declare void @julia_g_1(ptr addrspace(10))

; Ref(x[1] * 2): data from x's array flows into a fresh object, which is
; returned
define nonnull ptr addrspace(10) @julia_box(ptr noundef nonnull swiftself %pgcstack, ptr addrspace(10) noundef nonnull align 8 dereferenceable(24) %x) {
top:
  %pgc = call ptr @julia.get_pgcstack()
  %ptls = getelementptr inbounds ptr, ptr %pgc, i64 2
  %xd = addrspacecast ptr addrspace(10) %x to ptr addrspace(11)
  %data = load ptr, ptr addrspace(11) %xd, align 8
  %v = load double, ptr %data, align 8
  %m = fmul double %v, 2.0
  %obj = call noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr %ptls, i64 8, ptr addrspace(10) addrspacecast (ptr inttoptr (i64 140000000 to ptr) to ptr addrspace(10)))
  %od = addrspacecast ptr addrspace(10) %obj to ptr addrspace(11)
  store double %m, ptr addrspace(11) %od, align 8
  ret ptr addrspace(10) %obj
}
; CHECK-LABEL: enzyme-function-summary julia_box:
; CHECK-SAME: "args":[{"escape":false,"read":false,"write":false},{"escape":false,"read":true,"write":false}]
; CHECK-SAME: "args_write_any":[false,false]
; CHECK-SAME: "edges":[]
; CHECK-SAME: "flow":{"a1":["ret"]}
; CHECK-SAME: "frees":false
; CHECK-SAME: "globals_read":[]
; CHECK-SAME: "globals_write":[]
; CHECK-SAME: "globals_write_any":[]
; CHECK-SAME: "pts":{}
; CHECK-SAME: "unknown":false

; setindex!(r::Base.RefValue{Any}, v): stores a pointer into r with a write
; barrier
define void @julia_setref(ptr noundef nonnull swiftself %pgcstack, ptr addrspace(10) noundef nonnull %r, ptr addrspace(10) noundef nonnull %v) {
top:
  %rd = addrspacecast ptr addrspace(10) %r to ptr addrspace(11)
  store atomic ptr addrspace(10) %v, ptr addrspace(11) %rd release, align 8
  call void (ptr addrspace(10), ...) @julia.write_barrier(ptr addrspace(10) %r, ptr addrspace(10) %v)
  ret void
}
; CHECK-LABEL: enzyme-function-summary julia_setref:
; CHECK-SAME: "args":[{"escape":false,"read":false,"write":false},{"escape":false,"read":false,"write":false},{"escape":true,"read":false,"write":false}]
; CHECK-SAME: "args_write_any":[false,true,false]
; CHECK-SAME: "edges":[]
; CHECK-SAME: "flow":{"a2":["a1"]}
; CHECK-SAME: "frees":false
; CHECK-SAME: "globals_read":[]
; CHECK-SAME: "globals_write":[]
; CHECK-SAME: "globals_write_any":[]
; CHECK-SAME: "pts":{"a2":["a1"]}
; CHECK-SAME: "unknown":false

; x[1] = y[1] through julia.gc_loaded, then a call
define void @julia_copy(ptr noundef nonnull swiftself %pgcstack, ptr addrspace(10) noundef nonnull %x, ptr addrspace(10) noundef nonnull %y) {
top:
  %yd = addrspacecast ptr addrspace(10) %y to ptr addrspace(11)
  %ydata = load ptr, ptr addrspace(11) %yd, align 8
  %yl = call ptr addrspace(13) @julia.gc_loaded(ptr addrspace(10) %y, ptr %ydata)
  %v = load double, ptr addrspace(13) %yl, align 8
  %xd = addrspacecast ptr addrspace(10) %x to ptr addrspace(11)
  %xdata = load ptr, ptr addrspace(11) %xd, align 8
  %xl = call ptr addrspace(13) @julia.gc_loaded(ptr addrspace(10) %x, ptr %xdata)
  store double %v, ptr addrspace(13) %xl, align 8
  call void @julia_g_1(ptr addrspace(10) %x)
  ret void
}
; CHECK-LABEL: enzyme-function-summary julia_copy:
; CHECK-SAME: "args":[{"escape":false,"read":false,"write":false},{"escape":false,"read":false,"write":true},{"escape":false,"read":true,"write":false}]
; CHECK-SAME: "args_write_any":[false,true,false]
; CHECK-SAME: "edges":{{\[\[}}"a1","julia_g_1",0{{\]\]}}
; CHECK-SAME: "flow":{"a2":["a1"]}
; CHECK-SAME: "frees":false
; CHECK-SAME: "globals_read":[]
; CHECK-SAME: "globals_write":[]
; CHECK-SAME: "globals_write_any":[]
; CHECK-SAME: "pts":{}
; CHECK-SAME: "unknown":false

; a store to a global Ref Julia addresses by a constant
define void @julia_setglobal(ptr noundef nonnull swiftself %pgcstack, double %x) {
top:
  store double %x, ptr inttoptr (i64 140000008 to ptr), align 8
  ret void
}
; CHECK-LABEL: enzyme-function-summary julia_setglobal:
; CHECK-SAME: "args":[{"escape":false,"read":false,"write":false},{"escape":false,"read":false,"write":false}]
; CHECK-SAME: "args_write_any":[false,false]
; CHECK-SAME: "edges":[]
; CHECK-SAME: "flow":{"a1":["g"]}
; CHECK-SAME: "frees":false
; CHECK-SAME: "globals_read":[]
; CHECK-SAME: "globals_write":["*"]
; CHECK-SAME: "globals_write_any":["*"]
; CHECK-SAME: "pts":{}
; CHECK-SAME: "unknown":false

declare nonnull ptr addrspace(10) @julia.call2(ptr, ptr addrspace(10), ptr addrspace(10), ...)
declare nonnull ptr addrspace(10) @ijl_invoke(...)

; a pointer round-tripped through ptrtoint/inttoptr keeps its roots
define void @julia_roundtrip(ptr addrspace(10) %x, double %v) {
top:
  %xd = addrspacecast ptr addrspace(10) %x to ptr addrspace(11)
  %data = load ptr, ptr addrspace(11) %xd, align 8
  %i = ptrtoint ptr %data to i64
  %j = add i64 %i, 8
  %p = inttoptr i64 %j to ptr
  store double %v, ptr %p, align 8
  ret void
}
; CHECK-LABEL: enzyme-function-summary julia_roundtrip:
; CHECK-SAME: "args":[{"escape":false,"read":false,"write":true},{"escape":false,"read":false,"write":false}]
; CHECK-SAME: "args_write_any":[true,false]
; CHECK-SAME: "edges":[]
; CHECK-SAME: "flow":{"a1":["a0"]}
; CHECK-SAME: "frees":false
; CHECK-SAME: "globals_read":[]
; CHECK-SAME: "globals_write":[]
; CHECK-SAME: "globals_write_any":[]
; CHECK-SAME: "pts":{}
; CHECK-SAME: "unknown":false

; an address loaded as an integer is unknown
define void @julia_intaddr(ptr %x, double %v) {
top:
  %i = load i64, ptr %x, align 8
  %p = inttoptr i64 %i to ptr
  store double %v, ptr %p, align 8
  ret void
}
; CHECK-LABEL: enzyme-function-summary julia_intaddr:
; CHECK-SAME: "args":[{"escape":false,"read":false,"write":false},{"escape":false,"read":false,"write":false}]
; CHECK-SAME: "args_write_any":[false,false]
; CHECK-SAME: "edges":[]
; CHECK-SAME: "flow":{}
; CHECK-SAME: "frees":false
; CHECK-SAME: "globals_read":[]
; CHECK-SAME: "globals_write":[]
; CHECK-SAME: "globals_write_any":[]
; CHECK-SAME: "pts":{}
; CHECK-SAME: "unknown":true

; dynamic dispatch may read and write everything reachable from what it is
; given and any global
define nonnull ptr addrspace(10) @julia_dynamic(ptr addrspace(10) %f, ptr addrspace(10) %x) {
top:
  %r = call nonnull ptr addrspace(10) (ptr, ptr addrspace(10), ptr addrspace(10), ...) @julia.call2(ptr @ijl_invoke, ptr addrspace(10) %f, ptr addrspace(10) %x)
  ret ptr addrspace(10) %r
}
; CHECK-LABEL: enzyme-function-summary julia_dynamic:
; CHECK-SAME: "args":[{"escape":true,"read":true,"write":true},{"escape":true,"read":true,"write":true}]
; CHECK-SAME: "args_write_any":[true,true]
; CHECK-SAME: "edges":[]
; CHECK-SAME: "flow":{"a0":["a0","a1","ret","g"],"a1":["a0","a1","ret","g"],"g":["a0","a1","ret","g"]}
; CHECK-SAME: "frees":true
; CHECK-SAME: "globals_read":["*"]
; CHECK-SAME: "globals_write":["*"]
; CHECK-SAME: "globals_write_any":["*"]
; CHECK-SAME: "pts":{"a0":["a0","a1","ret","g"],"a1":["a0","a1","ret","g"],"g":["a0","a1","ret","g"]}
; CHECK-SAME: "unknown":false

