// RUN: rm -rf %t && mkdir -p %t/pass/Solve %t/pass/runtime/profiler %t/pass/tools/driver
// RUN: echo 'int solve();' > %t/pass/Solve/Solve.cpp
// RUN: echo 'int profile();' > %t/pass/runtime/profiler/Profiler.cpp
// RUN: echo 'int main();' > %t/pass/tools/driver/driver.cpp
// RUN: %cmake -DPASS_DIR=%t/pass -DOUT=%t/before.h -P %S/../../../Poseidon/cmake/BuildIdentity.cmake
// RUN: FileCheck %s < %t/before.h
// RUN: echo '// changed' >> %t/pass/Solve/Solve.cpp
// RUN: %cmake -DPASS_DIR=%t/pass -DOUT=%t/pass.h -P %S/../../../Poseidon/cmake/BuildIdentity.cmake
// RUN: not diff %t/before.h %t/pass.h
// RUN: echo '// changed' >> %t/pass/runtime/profiler/Profiler.cpp
// RUN: %cmake -DPASS_DIR=%t/pass -DOUT=%t/runtime.h -P %S/../../../Poseidon/cmake/BuildIdentity.cmake
// RUN: not diff %t/pass.h %t/runtime.h
// RUN: echo '// changed' >> %t/pass/tools/driver/driver.cpp
// RUN: %cmake -DPASS_DIR=%t/pass -DOUT=%t/tool.h -P %S/../../../Poseidon/cmake/BuildIdentity.cmake
// RUN: not diff %t/runtime.h %t/tool.h
// RUN: touch %t/pass/Solve/Added.h
// RUN: %cmake -DPASS_DIR=%t/pass -DOUT=%t/added.h -P %S/../../../Poseidon/cmake/BuildIdentity.cmake
// RUN: not diff %t/tool.h %t/added.h
// RUN: rm %t/pass/Solve/Added.h
// RUN: %cmake -DPASS_DIR=%t/pass -DOUT=%t/removed.h -P %S/../../../Poseidon/cmake/BuildIdentity.cmake
// RUN: diff %t/tool.h %t/removed.h
// RUN: echo '__global__ void k();' > %t/pass/runtime/profiler/Profiler.cu
// RUN: %cmake -DPASS_DIR=%t/pass -DOUT=%t/other.h -P %S/../../../Poseidon/cmake/BuildIdentity.cmake
// RUN: diff %t/tool.h %t/other.h

// cmake/BuildIdentity.cmake digests every *.cpp and *.h under PASS_DIR: an
// edit, an addition or a removal changes the identity; other files do not.

// CHECK: kPassIdentity = "{{[0-9A-F]+}}";
