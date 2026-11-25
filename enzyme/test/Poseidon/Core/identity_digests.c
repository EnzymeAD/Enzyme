// RUN: rm -rf %t && mkdir -p %t/pass
// RUN: cp %S/../../../Poseidon/Techniques/Matmul/MatmulAccuracy.cpp %S/../../../Poseidon/Techniques/Matmul/MatmulSimulation.h %S/../../../Poseidon/Techniques/Matmul/MatmulSimulation.cpp %S/../../../Poseidon/runtime/ozaki/OzakiII.h %S/../../../Poseidon/runtime/tcec/tcec_scale.h %S/../../../Poseidon/Core/FPKind.h %S/../../../Poseidon/Core/FPKind.cpp %S/../../../Poseidon/Core/UniformReal.h %S/../../../Poseidon/Core/UniformReal.cpp %S/../../../Poseidon/Techniques/Matmul/MatmulAccuracyMemo.cpp %S/../../../Poseidon/Solve/Solvers.cpp %t/pass/
// RUN: %cmake -DPASS_DIR=%t/pass -DOUT=%t/before.h -P %S/../../../Poseidon/cmake/BuildIdentity.cmake
// RUN: echo '// changed' >> %t/pass/Solvers.cpp
// RUN: %cmake -DPASS_DIR=%t/pass -DOUT=%t/pass_changed.h -P %S/../../../Poseidon/cmake/BuildIdentity.cmake
// RUN: echo '// changed' >> %t/pass/MatmulAccuracyMemo.cpp
// RUN: %cmake -DPASS_DIR=%t/pass -DOUT=%t/memo_changed.h -P %S/../../../Poseidon/cmake/BuildIdentity.cmake
// RUN: FileCheck %s < %t/before.h
// RUN: grep kPassIdentity %t/before.h > %t/before.pass && grep kPassIdentity %t/pass_changed.h > %t/pass_changed.pass
// RUN: not diff %t/before.pass %t/pass_changed.pass
// RUN: grep kPassIdentity %t/memo_changed.h > %t/memo_changed.pass
// RUN: not diff %t/pass_changed.pass %t/memo_changed.pass
// RUN: mkdir -p %t/tree/matmul
// RUN: cp %S/../../../Poseidon/Techniques/Matmul/MatmulAccuracy.cpp %S/../../../Poseidon/Techniques/Matmul/MatmulSimulation.h %S/../../../Poseidon/Techniques/Matmul/MatmulSimulation.cpp %S/../../../Poseidon/runtime/ozaki/OzakiII.h %S/../../../Poseidon/runtime/tcec/tcec_scale.h %S/../../../Poseidon/Core/FPKind.h %S/../../../Poseidon/Core/FPKind.cpp %S/../../../Poseidon/Core/UniformReal.h %S/../../../Poseidon/Core/UniformReal.cpp %t/tree/
// RUN: cp %S/../../../Poseidon/Techniques/Matmul/MatmulAccuracyMemo.cpp %t/tree/matmul/
// RUN: %cmake -DPASS_DIR=%t/tree -DOUT=%t/tree.h -P %S/../../../Poseidon/cmake/BuildIdentity.cmake
// RUN: echo '// changed' >> %t/tree/matmul/MatmulAccuracyMemo.cpp
// RUN: %cmake -DPASS_DIR=%t/tree -DOUT=%t/tree_sub_changed.h -P %S/../../../Poseidon/cmake/BuildIdentity.cmake
// RUN: grep kPassIdentity %t/tree.h > %t/tree.pass && grep kPassIdentity %t/tree_sub_changed.h > %t/tree_sub_changed.pass
// RUN: not diff %t/tree.pass %t/tree_sub_changed.pass
// RUN: mkdir -p %t/tree/runtime/fpprofiler %t/tree/tools/driver %t/tree/cmake %t/tree/test %t/tree/build-x/identity %t/tree/out/identity
// RUN: cp %S/../../../Poseidon/Solve/Solvers.cpp %t/tree/runtime/fpprofiler/ && cp %S/../../../Poseidon/Solve/Solvers.cpp %t/tree/tools/driver/ && cp %S/../../../Poseidon/Solve/Solvers.cpp %t/tree/cmake/ && cp %S/../../../Poseidon/Solve/Solvers.cpp %t/tree/test/
// RUN: cp %t/tree.h %t/tree/build-x/identity/BuildIdentity.h && cp %t/tree.h %t/tree/out/identity/BuildIdentity.h
// RUN: %cmake -DPASS_DIR=%t/tree -DBINARY_DIR=%t/tree/out -DOUT=%t/tree_others.h -P %S/../../../Poseidon/cmake/BuildIdentity.cmake
// RUN: diff %t/tree_sub_changed.h %t/tree_others.h
// RUN: echo '// changed' >> %t/tree/runtime/fpprofiler/Solvers.cpp
// RUN: %cmake -DPASS_DIR=%t/tree -DBINARY_DIR=%t/tree/out -DOUT=%t/tree_runtime_changed.h -P %S/../../../Poseidon/cmake/BuildIdentity.cmake
// RUN: diff %t/tree_sub_changed.h %t/tree_runtime_changed.h
// RUN: mkdir -p %t/tree/runtime/ozaki && cp %S/../../../Poseidon/runtime/ozaki/OzakiII.h %t/tree/runtime/ozaki/
// RUN: %cmake -DPASS_DIR=%t/tree -DBINARY_DIR=%t/tree/out -DOUT=%t/tree_ozaki.h -P %S/../../../Poseidon/cmake/BuildIdentity.cmake
// RUN: grep kPassIdentity %t/tree_runtime_changed.h > %t/tree_runtime_changed.pass && grep kPassIdentity %t/tree_ozaki.h > %t/tree_ozaki.pass
// RUN: not diff %t/tree_runtime_changed.pass %t/tree_ozaki.pass

// cmake/BuildIdentity.cmake digests the pass identity. It covers every source
// of the pass, in subdirectories too, the accuracy memo's code included, and
// the Ozaki-II header the pass shares with its runtime; files under the
// runtimes, the tools, the CMake scripts, a test or a build directory do not
// count.

// CHECK: kPassIdentity = "{{[0-9A-F]+}}";
