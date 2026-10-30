; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -enzyme-julia-addr-load -passes="enzyme" -S | FileCheck %s; fi

; Reduced from Enzyme.jl, logjac(x) = last((vcat([x[1]], [x[2]]), log(x[2]))).
; An active value is stored into a fresh Memory %mem, which is stored into a
; fresh array %arr passed to @consume. @consume is enzyme_ReadOnlyOrThrow, so it
; does not write the active value anywhere. While checking the activity of
; %mem under the hypothesis that %mem is active, @consume is found to be a
; constant instruction, which re-evaluates %mem inside the hypothesis and proves
; it constant. The memory search must then not treat %mem as active.

declare ptr @julia.get_pgcstack() #1
declare ptr addrspace(13) @julia.gc_loaded(ptr addrspace(10) nocapture readnone, ptr readnone) #3
declare noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr, i64, ptr addrspace(10)) #4
declare nonnull ptr @julia.pointer_from_objref(ptr addrspace(11)) #5
declare void @consume(ptr addrspace(10)) #24
declare double @__enzyme_fwddiff(...)

define double @f(ptr addrspace(10) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@double, [-1,8]:Pointer, [-1,8,0]:Integer, [-1,8,1]:Integer, [-1,8,2]:Integer, [-1,8,3]:Integer, [-1,8,4]:Integer, [-1,8,5]:Integer, [-1,8,6]:Integer, [-1,8,7]:Integer, [-1,8,8]:Pointer, [-1,8,8,-1]:Float@double, [-1,16]:Integer, [-1,17]:Integer, [-1,18]:Integer, [-1,19]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer}" %x) {
top:
  %pgcstack = call ptr @julia.get_pgcstack()
  %x11 = addrspacecast ptr addrspace(10) %x to ptr addrspace(11)
  %xdata = load ptr, ptr addrspace(11) %x11, align 8
  %xmemp = getelementptr i8, ptr addrspace(11) %x11, i64 8
  %xmem = load ptr addrspace(10), ptr addrspace(11) %xmemp, align 8
  %xp = call ptr addrspace(13) @julia.gc_loaded(ptr addrspace(10) %xmem, ptr %xdata)
  %x2p = getelementptr i8, ptr addrspace(13) %xp, i64 8
  %x2 = load double, ptr addrspace(13) %x2p, align 8
  %task = getelementptr i8, ptr %pgcstack, i64 -152
  %mem = call noalias nonnull "enzyme_type"="{[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer, [-1,8]:Pointer, [-1,8,-1]:Float@double}" ptr addrspace(10) @julia.gc_alloc_obj(ptr %task, i64 24, ptr addrspace(10) null)
  %mem11 = addrspacecast ptr addrspace(10) %mem to ptr addrspace(11)
  %memptrp = getelementptr i8, ptr addrspace(11) %mem11, i64 8
  %memobj = call nonnull "enzyme_type"="{[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer, [-1,8]:Pointer, [-1,8,-1]:Float@double}" ptr @julia.pointer_from_objref(ptr addrspace(11) %mem11)
  %memdata = getelementptr i8, ptr %memobj, i64 16
  store ptr %memdata, ptr addrspace(11) %memptrp, align 8
  %arr = call noalias nonnull "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@double, [-1,8]:Pointer, [-1,8,0]:Integer, [-1,8,1]:Integer, [-1,8,2]:Integer, [-1,8,3]:Integer, [-1,8,4]:Integer, [-1,8,5]:Integer, [-1,8,6]:Integer, [-1,8,7]:Integer, [-1,8,8]:Pointer, [-1,8,8,-1]:Float@double, [-1,16]:Integer, [-1,17]:Integer, [-1,18]:Integer, [-1,19]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer}" ptr addrspace(10) @julia.gc_alloc_obj(ptr %task, i64 24, ptr addrspace(10) null)
  %arr11 = addrspacecast ptr addrspace(10) %arr to ptr addrspace(11)
  %arrmemp = getelementptr i8, ptr addrspace(11) %arr11, i64 8
  store ptr addrspace(10) %mem, ptr addrspace(11) %arrmemp, align 8
  %dst = call ptr addrspace(13) @julia.gc_loaded(ptr addrspace(10) %mem, ptr %memdata)
  store double %x2, ptr addrspace(13) %dst, align 8
  call void @consume(ptr addrspace(10) %arr)
  ret double %x2
}

define double @df(ptr addrspace(10) %x, ptr addrspace(10) %dx) {
  %r = call double (...) @__enzyme_fwddiff(ptr @f, metadata !"enzyme_dup", ptr addrspace(10) %x, ptr addrspace(10) %dx)
  ret double %r
}

attributes #1 = { nofree nosync nounwind willreturn memory(none) "enzyme_inactive" "enzyme_no_escaping_allocation" "enzyme_shouldrecompute" }
attributes #3 = { nofree norecurse nosync nounwind speculatable willreturn memory(argmem: read) "enzyme_nocache" "enzyme_shouldrecompute" }
attributes #4 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(1) memory(inaccessiblemem: readwrite) "enzyme_ReadOnlyOrThrow" "enzyme_no_escaping_allocation" }
attributes #5 = { nofree norecurse nosync nounwind speculatable willreturn memory(none) "enzyme_no_escaping_allocation" }
attributes #24 = { memory(read, inaccessiblemem: readwrite) "enzyme_ReadOnlyOrThrow" }

; CHECK: define internal double @fwddiffef(ptr addrspace(10) {{.*}}%x, ptr addrspace(10) {{.*}}%"x'")
; CHECK:   %"x2'ipl" = load double, ptr addrspace(13) %"x2p'ipg"
; CHECK:   store double %x2, ptr addrspace(13) %dst
; CHECK-NOT: "dst'
; CHECK:   call void @consume(ptr addrspace(10) %arr)
; CHECK-NEXT:   ret double %"x2'ipl"
