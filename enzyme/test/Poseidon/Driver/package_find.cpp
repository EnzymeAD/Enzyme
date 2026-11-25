// REQUIRES: enzyme
// RUN: rm -rf %t && mkdir %t
// RUN: %cmake %cmake_generator -S %S/../Inputs/package -B %t/plain -DCMAKE_CXX_COMPILER=%clang++ -DPoseidon_DIR=%poseidon_package_dir -DPOSEIDON_MODE=plain > %t/plain.log
// RUN: FileCheck --check-prefix=TREE -DDIR=%poseidon_package_dir %s < %t/plain.log
// RUN: %cmake --build %t/plain -v > %t/plain.build.log
// RUN: FileCheck --check-prefix=PLAINCMD %s < %t/plain.build.log
// RUN: %t/plain/host | FileCheck --check-prefix=OUT %s
// RUN: %cmake %cmake_generator -S %S/../Inputs/package -B %t/gen -DCMAKE_CXX_COMPILER=%clang++ -DPoseidon_DIR=%poseidon_package_dir -DPOSEIDON_MODE=instrument > /dev/null
// RUN: %cmake --build %t/gen > /dev/null
// RUN: env POSEIDON_PROFILE_DIR=%t/prof %t/gen/host | FileCheck --check-prefix=OUT %s
// RUN: ls %t/prof | FileCheck --check-prefix=PROF %s
// RUN: %cmake %cmake_generator -S %S/../Inputs/package -B %t/use -DCMAKE_CXX_COMPILER=%clang++ -DPoseidon_DIR=%poseidon_package_dir -DPOSEIDON_MODE=optimize -DPOSEIDON_PROFILE=%t/prof "-DPOSEIDON_FLAGS=-poseidon-tau=1e-6 -poseidon-cost-model=%host_cost_model -poseidon-enable-herbie=0 -poseidon-print" > /dev/null
// RUN: %cmake --build %t/use -v 2>&1 | FileCheck --check-prefix=USE %s
// RUN: %t/use/host | FileCheck --check-prefix=OUT %s
// RUN: %cmake %cmake_generator -S %S/../Inputs/package -B %t/none -DCMAKE_CXX_COMPILER=%clang++ -DPoseidon_DIR=%poseidon_package_dir -DPOSEIDON_MODE=optimize -DPOSEIDON_GPU_RUNTIME=NONE -DPOSEIDON_PROFILE=%t/prof "-DPOSEIDON_FLAGS=-poseidon-tau=1e-6 -poseidon-cost-model=%host_cost_model -poseidon-enable-herbie=0" > /dev/null
// RUN: %cmake --build %t/none -v 2>&1 | FileCheck --check-prefix=NONE --implicit-check-not=cublas --implicit-check-not=cudart --implicit-check-not=poseidon_rt %s
// RUN: not %cmake %cmake_generator -S %S/../Inputs/package -B %t/bad -DCMAKE_CXX_COMPILER=%clang++ -DPoseidon_DIR=%poseidon_package_dir -DPOSEIDON_GPU_RUNTIME=OpenCL 2>&1 | FileCheck --check-prefix=BADRT %s
// RUN: not %cmake %cmake_generator -S %S/../Inputs/package -B %t/newer -DCMAKE_CXX_COMPILER=%clang++ -DPoseidon_DIR=%poseidon_package_dir -DPOSEIDON_MIN_VERSION=99 2>&1 | FileCheck --check-prefix=NEWER %s
//
// The same from an installation of this build tree.
// RUN: %cmake --install %enzyme_binary_dir/Enzyme --prefix %t/inst > /dev/null
// RUN: %cmake --install %poseidon_package_dir --prefix %t/inst > /dev/null
// RUN: %cmake %cmake_generator -S %S/../Inputs/package -B %t/igen -DCMAKE_CXX_COMPILER=%clang++ -DCMAKE_PREFIX_PATH=%t/inst -DPOSEIDON_MODE=instrument > %t/igen.log
// RUN: FileCheck --check-prefix=INST -DLIBDIR=%install_libdir -DPREFIX=%t/inst %s < %t/igen.log
// RUN: %cmake --build %t/igen -v > %t/igen.build.log
// RUN: FileCheck --check-prefix=INSTCMD -DLIBDIR=%install_libdir -DPREFIX=%t/inst --implicit-check-not=enzyme-lto-prelink %s < %t/igen.build.log
// RUN: env POSEIDON_PROFILE_DIR=%t/iprof %t/igen/host | FileCheck --check-prefix=OUT %s
// RUN: ls %t/iprof | FileCheck --check-prefix=PROF %s
// RUN: test -x %t/inst/bin/poseidon-clang++
// RUN: env POSEIDON_ECHO=1 %t/inst/bin/poseidon-clang++ -poseidon-profile-generate %S/../Inputs/package/host.cpp -### -o %t/inst.exe 2>&1 | FileCheck --check-prefix=INSTDRV -DLIBDIR=%install_libdir -DPREFIX=%t/inst %s
// RUN: cp -r %t/inst %t/noenzyme && rm %t/noenzyme/lib/ClangEnzyme-*.so
// RUN: not %t/noenzyme/bin/poseidon-clang++ -poseidon-profile-generate -c %S/../Inputs/package/host.cpp -o %t/noenzyme.o 2>&1 | FileCheck --check-prefix=NOENZYME -DLIBDIR=%install_libdir -DPREFIX=%t/noenzyme %s
// RUN: %t/noenzyme/bin/poseidon-clang++ -c %S/../Inputs/package/host.cpp -o %t/noenzyme.o

// find_package(Poseidon) from the build tree and from an installation, and
// one interface target per action: Poseidon's plugin, and Enzyme's after it to
// generate a profile, the public header and POSEIDON_DEVICE_MATH,
// statement-only contraction and the runtimes each action links, for a project
// that names nothing else of Poseidon's. The package leaves the project's
// MPFR_FOUND alone (Inputs/package/CMakeLists.txt stops otherwise).

// TREE: -- Poseidon 0.1 from [[DIR]]
// TREE-NEXT: -- Poseidon plugins [[DIR]]/lib/Poseidon-{{[0-9]+}}.so {{.*}}/Enzyme/ClangEnzyme-{{[0-9]+}}.so
// TREE-NEXT: -- Poseidon include {{.*}}/include

// PLAINCMD: clang++ -DPOSEIDON_DEVICE_MATH {{.*}}-fpass-plugin={{[^ ]+}}/Poseidon-{{[0-9]+}}.so {{(-Xclang -load -Xclang [^ ]+ )?}}-ffp-contract=on
// PLAINCMD-NOT: ClangEnzyme

// OUT: wave 348.384

// PROF: {{^}}_Z4wavePKdPdi.fpprofile{{$}}

// USE: clang++ {{.*}}-Xclang -mllvm -Xclang -poseidon-profile-use={{[^ ]+}}/prof -Xclang -mllvm -Xclang -poseidon-tau=1e-6 -Xclang -mllvm -Xclang -poseidon-cost-model={{[^ ]+}}/cm_cpu_host.csv -Xclang -mllvm -Xclang -poseidon-enable-herbie=0 -Xclang -mllvm -Xclang -poseidon-print -fpass-plugin={{[^ ]+}}/Poseidon-{{[0-9]+}}.so {{(-Xclang -load -Xclang [^ ]+ )?}}-ffp-contract=on {{.*}}-c {{[^ ]+}}/host.cpp{{$}}
// USE: [poseidon] preprocess__Z4wavePKdPdi_poseidon_body: tau=1.000000e-06 S=
// USE: Applying solution for precision change All FP64(0%) + FP32(100%) (#0)
// USE: [poseidon] pin: _Z4wavePKdPdi:PT0=All FP64(0%) + FP32(100%)
// USE: clang++ {{.*}}CMakeFiles/host.dir/host.cpp.o -o host {{.*}}libposeidon_profile.a && :{{$}}

// NEWER: Could not find a configuration file for package "Poseidon" that is
// NEWER: compatible with requested version "99"

// INST: -- Poseidon 0.1 from [[PREFIX]]/[[LIBDIR]]/cmake/Poseidon
// INST-NEXT: -- Poseidon plugins [[PREFIX]]/[[LIBDIR]]/Poseidon-{{[0-9]+}}.so [[PREFIX]]/lib/ClangEnzyme-{{[0-9]+}}.so
// INST-NEXT: -- Poseidon include [[PREFIX]]/include

// INSTCMD: clang++ -DPOSEIDON_DEVICE_MATH {{.*}}-fpass-plugin=[[PREFIX]]/[[LIBDIR]]/Poseidon-{{[0-9]+}}.so {{(-Xclang -load -Xclang [^ ]+ )?}}-fpass-plugin=[[PREFIX]]/lib/ClangEnzyme-{{[0-9]+}}.so {{(-Xclang -load -Xclang [^ ]+ )?}}-Xclang -mllvm -Xclang -poseidon-profile-generate -ffp-contract=on
// INSTCMD: libposeidon_profile_host.a

// The installed driver takes the plugins, the header and the runtimes from its
// installation, and stops when one it needs is missing there: Enzyme's only
// to generate a profile.

// INSTDRV: clang{{.*}} -fpass-plugin=[[PREFIX]]/[[LIBDIR]]/Poseidon-{{[0-9]+}}.so {{(-Xclang -load -Xclang [^ ]+ )?}}-fpass-plugin=[[PREFIX]]/lib/ClangEnzyme-{{[0-9]+}}.so
// INSTDRV-SAME: -isystem [[PREFIX]]/include
// INSTDRV-SAME: -L[[PREFIX]]/[[LIBDIR]] -lposeidon_profile_host

// NOENZYME: poseidon-clang++: error: Poseidon: Enzyme plugin not found at [[PREFIX]]/lib/ClangEnzyme-{{[0-9]+}}.so; install Enzyme to the same prefix or set ENZYME_PLUGIN{{$}}

// A project of host code alone links no GPU runtime of a GPU build.
// NONE: libposeidon_profile.a
// BADRT: POSEIDON_GPU_RUNTIME is OpenCL; it takes CUDA, HIP or NONE
