// RUN: rm -rf %t && mkdir -p %t/pass/Solve %t/pass/runtime/fpprofiler %t/pass/tools/driver
// RUN: cp %S/../../../Poseidon/Solve/Solvers.cpp %t/pass/Solve/
// RUN: cp %S/../../../Poseidon/runtime/fpprofiler/FPProfiler.cpp %t/pass/runtime/fpprofiler/
// RUN: cp %S/../../../Poseidon/tools/driver/poseidon-clang.cpp %t/pass/tools/driver/
// RUN: %cmake -DPASS_DIR=%t/pass -DOUT=%t/before.h -P %S/../../../Poseidon/cmake/BuildIdentity.cmake
// RUN: FileCheck %s < %t/before.h
// RUN: echo '// changed' >> %t/pass/Solve/Solvers.cpp
// RUN: %cmake -DPASS_DIR=%t/pass -DOUT=%t/pass.h -P %S/../../../Poseidon/cmake/BuildIdentity.cmake
// RUN: not diff %t/before.h %t/pass.h
// RUN: echo '// changed' >> %t/pass/runtime/fpprofiler/FPProfiler.cpp
// RUN: %cmake -DPASS_DIR=%t/pass -DOUT=%t/runtime.h -P %S/../../../Poseidon/cmake/BuildIdentity.cmake
// RUN: not diff %t/pass.h %t/runtime.h
// RUN: echo '// changed' >> %t/pass/tools/driver/poseidon-clang.cpp
// RUN: %cmake -DPASS_DIR=%t/pass -DOUT=%t/tool.h -P %S/../../../Poseidon/cmake/BuildIdentity.cmake
// RUN: not diff %t/runtime.h %t/tool.h
// RUN: touch %t/pass/Solve/Added.h
// RUN: %cmake -DPASS_DIR=%t/pass -DOUT=%t/added.h -P %S/../../../Poseidon/cmake/BuildIdentity.cmake
// RUN: not diff %t/tool.h %t/added.h
// RUN: rm %t/pass/Solve/Added.h
// RUN: %cmake -DPASS_DIR=%t/pass -DOUT=%t/removed.h -P %S/../../../Poseidon/cmake/BuildIdentity.cmake
// RUN: diff %t/tool.h %t/removed.h
// RUN: cp %S/../../../Poseidon/runtime/fpprofiler/FPProfilerCUDA.cu %t/pass/runtime/fpprofiler/
// RUN: %cmake -DPASS_DIR=%t/pass -DOUT=%t/other.h -P %S/../../../Poseidon/cmake/BuildIdentity.cmake
// RUN: diff %t/tool.h %t/other.h

// cmake/BuildIdentity.cmake digests every *.cpp and *.h under PASS_DIR, at any
// depth: an edit to a source of the pass, of a runtime or of a tool, and a
// source added or removed, change the identity; any other file does not.

// CHECK: kPassIdentity = "{{[0-9A-F]+}}";
