// REQUIRES: enzyme, cuda-runtime
// RUN: rm -rf %t && mkdir -p %t
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O3 -poseidon-profile-generate -poseidon-cache=%t/c -I%S/../Inputs/two_tu -c %S/../Inputs/two_tu/a.cu -o %t/a.o
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O3 -poseidon-profile-generate -poseidon-cache=%t/c -I%S/../Inputs/two_tu -c %S/../Inputs/two_tu/b.cu -o %t/b.o
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O3 -poseidon-profile-generate -poseidon-cache=%t/c -I%S/../Inputs/two_tu -c %S/../Inputs/two_tu/main.cu -o %t/main.o
// RUN: %poseidon_clangxx -poseidon-profile-generate %t/a.o %t/b.o %t/main.o -L%cuda_lib_dir -lcudart -o %t/two
// RUN: llvm-ar rcs %t/libk.a %t/a.o %t/b.o
// RUN: %poseidon_clangxx -poseidon-profile-generate %t/main.o %t/libk.a -L%cuda_lib_dir -lcudart -o %t/lib
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O3 -poseidon-profile-generate -poseidon-cache=%t/c1 -I%S/../Inputs/two_tu %S/../Inputs/two_tu/one.cu -o %t/one
// RUN: env POSEIDON_PROFILE_DIR=%t/p.two %t/two | FileCheck --check-prefix=OUT %s
// RUN: env POSEIDON_PROFILE_DIR=%t/p.lib %t/lib | FileCheck --check-prefix=OUT %s
// RUN: env POSEIDON_PROFILE_DIR=%t/p.one %t/one | FileCheck --check-prefix=OUT %s
// RUN: %python %S/../Inputs/two_tu/cmpprof.py --ignore SiteId --sum-tol 1e-12 %t/p.one %t/p.two | FileCheck --check-prefix=SAME %s
// RUN: %python %S/../Inputs/two_tu/cmpprof.py --ignore SiteId --sum-tol 1e-12 %t/p.one %t/p.lib | FileCheck --check-prefix=SAME %s
// RUN: cat %t/p.two/preprocess__Z9hooke_volRK4mat3fPf.poseidon._Z6vol_atPKffi.0.fpprofile | FileCheck --check-prefix=MERGED %s
//
// The same, with relocatable device code, against the one-unit program built
// the same way.
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O3 -fcuda-rdc -poseidon-profile-generate -poseidon-cache=%t/r -I%S/../Inputs/two_tu -c %S/../Inputs/two_tu/a.cu -o %t/ra.o
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O3 -fcuda-rdc -poseidon-profile-generate -poseidon-cache=%t/r -I%S/../Inputs/two_tu -c %S/../Inputs/two_tu/b.cu -o %t/rb.o
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O3 -fcuda-rdc -poseidon-profile-generate -poseidon-cache=%t/r -I%S/../Inputs/two_tu -c %S/../Inputs/two_tu/main.cu -o %t/rmain.o
// RUN: %poseidon_clangxx -poseidon-profile-generate --offload-link %t/ra.o %t/rb.o %t/rmain.o -L%cuda_lib_dir -lcudart -o %t/rtwo
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O3 -fcuda-rdc -poseidon-profile-generate -poseidon-cache=%t/r1 -I%S/../Inputs/two_tu %S/../Inputs/two_tu/one.cu -o %t/rone
// RUN: env POSEIDON_PROFILE_DIR=%t/p.rtwo %t/rtwo | FileCheck --check-prefix=OUT %s
// RUN: env POSEIDON_PROFILE_DIR=%t/p.rone %t/rone | FileCheck --check-prefix=OUT %s
// RUN: %python %S/../Inputs/two_tu/cmpprof.py --ignore SiteId --sum-tol 1e-12 %t/p.rone %t/p.rtwo | FileCheck --check-prefix=SAME %s
//
// Two different functions under one name, one in each unit, are refused when
// the profile is written.
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O3 -poseidon-profile-generate -poseidon-cache=%t/k -DVARIANT=1 -c %S/../Inputs/two_tu/clash.cu -o %t/k1.o
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O3 -poseidon-profile-generate -poseidon-cache=%t/k -DVARIANT=2 -c %S/../Inputs/two_tu/clash.cu -o %t/k2.o
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O3 -poseidon-profile-generate -poseidon-cache=%t/k -c %S/../Inputs/two_tu/clash_main.cu -o %t/kmain.o
// RUN: %poseidon_clangxx -poseidon-profile-generate %t/k1.o %t/k2.o %t/kmain.o -L%cuda_lib_dir -lcudart -o %t/clash
// RUN: not env POSEIDON_PROFILE_DIR=%t/p.clash %t/clash > %t/clash.out 2> %t/clash.err
// RUN: FileCheck --check-prefix=STATUS %s < %t/clash.out
// RUN: FileCheck --check-prefix=CLASH %s < %t/clash.err
// RUN: not ls %t/p.clash/preprocess__ZL4stepPdi_poseidon_body.fpprofile

// A program of two CUDA translation units and a main, compiled one unit at a
// time without relocatable device code, and linked as objects or with the
// units in a static library, records the profile the same program records as
// one unit: an annotated kernel in each unit, an annotated mfem::forall-shaped
// lambda in each, and a device call site in an inline function both units
// run. Each unit's device module has its own copy of that site; its record is
// both copies', 2 x 2048 executions from each of two kernels.

// OUT: axpb 2703.7489499999992 lift 367.08090963891368 forall_a 121902.63888375004 forall_b -7593.8872846631084 vol_a -0.549021492 vol_b -0.219608603
// OUT-NEXT: status no error

// SAME: overall: exact fields identical

// MERGED: MaxBlockDims = 128 1 1
// MERGED-NEXT: MaxGridDims = 32 1 1
// MERGED-NEXT: LaunchCount = 96
// MERGED: Exec = 8192

// STATUS: status no error
// CLASH: Poseidon FPProfiler: two different functions were profiled under the name 'preprocess__ZL4stepPdi_poseidon_body' in two device modules (CanonicalHash = {{[0-9a-f]+}}, CanonicalHash = {{[0-9a-f]+}}), and one profile file cannot hold both.
