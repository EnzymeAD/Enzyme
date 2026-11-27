// REQUIRES: enzyme
// DEFINE: %{drv} = env TMPDIR=%t/tmp POSEIDON_ECHO=1 %poseidon_clangxx \
// DEFINE:   -poseidon-profile-generate -x cuda --cuda-gpu-arch=%gpu_arch \
// DEFINE:   --target=x86_64-unknown-linux-gnu \
// DEFINE:   --cuda-path=%t/cuda -nocudainc -nocudalib -### %s
// RUN: rm -rf %t && mkdir -p %t/tmp
// RUN: %{drv} -o %t/k.o 2>&1 | FileCheck -DTMP=%t/tmp --check-prefix=BOTH %s
// RUN: ls -A %t/tmp | count 0
// RUN: %{drv} --cuda-device-only -o %t/k.s 2>&1 | FileCheck --check-prefix=HALF %s
// RUN: %{drv} --cuda-host-only -o %t/k.o 2>&1 | FileCheck --check-prefix=HALF %s
// RUN: %{drv} -poseidon-launch-dir=%t/mine -o %t/k.o 2>&1 | FileCheck -DMINE=%t/mine --check-prefix=MINE %s

// The launch manifest directory: a temporary one per compile of both halves,
// removed with it; none for a compile of one half; the user's when named.
// BOTH: -Xclang -mllvm -Xclang -poseidon-launch-dir=[[TMP]]/poseidon-launches-[[ID:[^ ]+]]
// BOTH: "-cc1" "-triple" "nvptx64-nvidia-cuda"{{.*}} "-mllvm" "-poseidon-launch-dir=[[TMP]]/poseidon-launches-[[ID]]"
// BOTH: "-cc1" "-triple" "x86_64-{{.*}} "-mllvm" "-poseidon-launch-dir=[[TMP]]/poseidon-launches-[[ID]]"
// HALF-NOT: launch-dir
// MINE-NOT: poseidon-launches-
// MINE: -poseidon-launch-dir=[[MINE]]
// MINE-NOT: poseidon-launches-
