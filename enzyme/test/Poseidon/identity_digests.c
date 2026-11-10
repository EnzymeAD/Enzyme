// RUN: rm -rf %t && mkdir -p %t/pass
// RUN: cp %S/../../Poseidon/Techniques/Matmul/MatmulAccuracy.cpp %S/../../Poseidon/Core/FPKind.h %S/../../Poseidon/Core/FPKind.cpp %S/../../Poseidon/Techniques/Matmul/MatmulAccuracyMemo.cpp %S/../../Poseidon/Solvers.cpp %t/pass/
// RUN: printf '# earlier identities\n0123456789ABCDEF\n\nFEDCBA9876543210\n' > %t/equivalents.txt
// RUN: %cmake -DPASS_DIR=%t/pass -DACCURACY_FILES=MatmulAccuracy.cpp,FPKind.h,FPKind.cpp -DEQUIVALENTS=%t/equivalents.txt -DOUT=%t/before.h -P %S/../../Poseidon/cmake/BuildIdentity.cmake
// RUN: mkdir -p %t/moved && cp %t/pass/MatmulAccuracy.cpp %t/moved/Model.cpp && cp %t/pass/FPKind.h %t/moved/Kinds.h && cp %t/pass/FPKind.cpp %t/moved/Kinds.cpp
// RUN: %cmake -DPASS_DIR=%t/moved -DACCURACY_FILES=Kinds.cpp,Model.cpp,Kinds.h -DEQUIVALENTS=%t/equivalents.txt -DOUT=%t/moved.h -P %S/../../Poseidon/cmake/BuildIdentity.cmake
// RUN: printf '#' | dd of=%t/moved/Kinds.cpp bs=1 seek=0 conv=notrunc 2> /dev/null
// RUN: cmp -l %t/pass/FPKind.cpp %t/moved/Kinds.cpp > %t/onebyte.txt || true
// RUN: FileCheck --check-prefix=ONEBYTE %s < %t/onebyte.txt
// RUN: %cmake -DPASS_DIR=%t/moved -DACCURACY_FILES=Kinds.cpp,Model.cpp,Kinds.h -DEQUIVALENTS=%t/equivalents.txt -DOUT=%t/moved_changed.h -P %S/../../Poseidon/cmake/BuildIdentity.cmake
// RUN: echo '// changed' >> %t/pass/Solvers.cpp
// RUN: %cmake -DPASS_DIR=%t/pass -DACCURACY_FILES=MatmulAccuracy.cpp,FPKind.h,FPKind.cpp -DEQUIVALENTS=%t/equivalents.txt -DOUT=%t/pass_changed.h -P %S/../../Poseidon/cmake/BuildIdentity.cmake
// RUN: echo '// changed' >> %t/pass/MatmulAccuracyMemo.cpp
// RUN: %cmake -DPASS_DIR=%t/pass -DACCURACY_FILES=MatmulAccuracy.cpp,FPKind.h,FPKind.cpp -DEQUIVALENTS=%t/equivalents.txt -DOUT=%t/memo_changed.h -P %S/../../Poseidon/cmake/BuildIdentity.cmake
// RUN: echo '// changed' >> %t/pass/FPKind.cpp
// RUN: %cmake -DPASS_DIR=%t/pass -DACCURACY_FILES=MatmulAccuracy.cpp,FPKind.h,FPKind.cpp -DEQUIVALENTS=%t/equivalents.txt -DOUT=%t/model_changed.h -P %S/../../Poseidon/cmake/BuildIdentity.cmake
// RUN: FileCheck %s < %t/before.h
// RUN: grep kPassIdentity %t/before.h > %t/before.pass && grep kPassIdentity %t/pass_changed.h > %t/pass_changed.pass
// RUN: grep kAccuracyModelIdentity %t/before.h > %t/before.model && grep kAccuracyModelIdentity %t/pass_changed.h > %t/pass_changed.model && grep kAccuracyModelIdentity %t/model_changed.h > %t/model_changed.model
// RUN: not diff %t/before.pass %t/pass_changed.pass
// RUN: diff %t/before.model %t/pass_changed.model
// RUN: grep kPassIdentity %t/memo_changed.h > %t/memo_changed.pass && grep kAccuracyModelIdentity %t/memo_changed.h > %t/memo_changed.model
// RUN: not diff %t/pass_changed.pass %t/memo_changed.pass
// RUN: diff %t/before.model %t/memo_changed.model
// RUN: not diff %t/memo_changed.model %t/model_changed.model
// RUN: grep kAccuracyModelIdentity %t/moved.h > %t/moved.model && grep kAccuracyModelIdentity %t/moved_changed.h > %t/moved_changed.model
// RUN: diff %t/before.model %t/moved.model
// RUN: not diff %t/moved.model %t/moved_changed.model
// RUN: mkdir -p %t/tree/matmul
// RUN: cp %S/../../Poseidon/Techniques/Matmul/MatmulAccuracy.cpp %S/../../Poseidon/Core/FPKind.h %S/../../Poseidon/Core/FPKind.cpp %t/tree/
// RUN: cp %S/../../Poseidon/Techniques/Matmul/MatmulAccuracyMemo.cpp %t/tree/matmul/
// RUN: %cmake -DPASS_DIR=%t/tree -DACCURACY_FILES=MatmulAccuracy.cpp,FPKind.h,FPKind.cpp -DEQUIVALENTS=%t/equivalents.txt -DOUT=%t/tree.h -P %S/../../Poseidon/cmake/BuildIdentity.cmake
// RUN: echo '// changed' >> %t/tree/matmul/MatmulAccuracyMemo.cpp
// RUN: %cmake -DPASS_DIR=%t/tree -DACCURACY_FILES=MatmulAccuracy.cpp,FPKind.h,FPKind.cpp -DEQUIVALENTS=%t/equivalents.txt -DOUT=%t/tree_sub_changed.h -P %S/../../Poseidon/cmake/BuildIdentity.cmake
// RUN: grep kPassIdentity %t/tree.h > %t/tree.pass && grep kPassIdentity %t/tree_sub_changed.h > %t/tree_sub_changed.pass
// RUN: not diff %t/tree.pass %t/tree_sub_changed.pass
// RUN: grep kAccuracyModelIdentity %t/tree_sub_changed.h > %t/tree_sub_changed.model
// RUN: diff %t/before.model %t/tree_sub_changed.model
// RUN: mkdir -p %t/tree/runtime/fpprofiler %t/tree/tools/driver %t/tree/cost_models %t/tree/cmake %t/tree/test %t/tree/build-x/identity %t/tree/out/identity
// RUN: cp %S/../../Poseidon/Solvers.cpp %t/tree/runtime/fpprofiler/ && cp %S/../../Poseidon/Solvers.cpp %t/tree/tools/driver/ && cp %S/../../Poseidon/Solvers.cpp %t/tree/cost_models/ && cp %S/../../Poseidon/Solvers.cpp %t/tree/cmake/ && cp %S/../../Poseidon/Solvers.cpp %t/tree/test/
// RUN: cp %t/tree.h %t/tree/build-x/identity/BuildIdentity.h && cp %t/tree.h %t/tree/out/identity/BuildIdentity.h
// RUN: %cmake -DPASS_DIR=%t/tree -DBINARY_DIR=%t/tree/out -DACCURACY_FILES=MatmulAccuracy.cpp,FPKind.h,FPKind.cpp -DEQUIVALENTS=%t/equivalents.txt -DOUT=%t/tree_others.h -P %S/../../Poseidon/cmake/BuildIdentity.cmake
// RUN: diff %t/tree_sub_changed.h %t/tree_others.h
// RUN: echo '// changed' >> %t/tree/runtime/fpprofiler/Solvers.cpp
// RUN: %cmake -DPASS_DIR=%t/tree -DBINARY_DIR=%t/tree/out -DACCURACY_FILES=MatmulAccuracy.cpp,FPKind.h,FPKind.cpp -DEQUIVALENTS=%t/equivalents.txt -DOUT=%t/tree_runtime_changed.h -P %S/../../Poseidon/cmake/BuildIdentity.cmake
// RUN: diff %t/tree_sub_changed.h %t/tree_runtime_changed.h
// RUN: mkdir -p %t/tree/runtime/ozaki && cp %S/../../Poseidon/runtime/ozaki/OzakiII.h %t/tree/runtime/ozaki/
// RUN: %cmake -DPASS_DIR=%t/tree -DBINARY_DIR=%t/tree/out -DACCURACY_FILES=MatmulAccuracy.cpp,FPKind.h,FPKind.cpp -DEQUIVALENTS=%t/equivalents.txt -DOUT=%t/tree_ozaki.h -P %S/../../Poseidon/cmake/BuildIdentity.cmake
// RUN: grep kPassIdentity %t/tree_runtime_changed.h > %t/tree_runtime_changed.pass && grep kPassIdentity %t/tree_ozaki.h > %t/tree_ozaki.pass
// RUN: not diff %t/tree_runtime_changed.pass %t/tree_ozaki.pass
// REQUIRES: poseidon

// The pass identity is a digest of every source of the pass, the accuracy-model
// identity of the model's own files alone: a change elsewhere in the pass, the
// memo's own code included, leaves the accuracy memo valid, a change to the
// model does not. The model's identity
// is of their contents only: the same bytes under other names, in another
// directory and listed in another order give the same identity, and one byte
// changed gives another. Earlier model identities declared equivalent are read
// from their list, comments and blank lines skipped. A source of the pass in a
// subdirectory is one of its sources; a file under the runtimes, the tools, the
// cost models, the CMake scripts, a test or a build directory is not, nor one
// under the build directory the build names, but the Ozaki-II header the pass
// shares with its runtime is one.

// ONEBYTE: {{^ *1 +57 +43$}}
// ONEBYTE-NOT: {{.}}

// CHECK: kPassIdentity = "{{[0-9A-F]+}}";
// CHECK: kAccuracyModelIdentity = "{{[0-9A-F]+}}";
// CHECK: std::array<const char *, 2> kAccuracyModelEquivalents = {"0123456789ABCDEF", "FEDCBA9876543210", };
