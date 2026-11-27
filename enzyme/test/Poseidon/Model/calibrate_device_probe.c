// poseidon-calibrate's GPU probe, with stand-in microbenchmarks: no device of a
// vendor (exit 2) asks the next vendor; a device that fails to initialize or a
// probe that cannot start ends the run with its reason; no device lists all.
// RUN: rm -rf %t && mkdir -p %t/init %t/start %t/none %t/amd %t/cm
// RUN: for d in init start none amd; do cp %poseidon_calibrate %t/$d/; done
// RUN: printf '#!/bin/sh\necho initialization error >&2\nexit 1\n' > %t/init/poseidon-calibrate-ops-cuda
// RUN: printf '#!/bin/sh\nprintf "AMD Instinct MI250X\\tgfx90a\\n"\n' > %t/init/poseidon-calibrate-ops-hip
// RUN: printf '#!/bin/sh\necho no CUDA-capable device is detected >&2\nexit 2\n' > %t/start/poseidon-calibrate-ops-cuda
// RUN: printf '#!/bin/sh\necho "poseidon-calibrate-ops-hip: error while loading shared libraries: libamdhip64.so.7: cannot open shared object file" >&2\nexit 127\n' > %t/start/poseidon-calibrate-ops-hip
// RUN: cp %t/start/poseidon-calibrate-ops-cuda %t/none/ && cp %t/start/poseidon-calibrate-ops-cuda %t/amd/
// RUN: cp %t/init/poseidon-calibrate-ops-hip %t/amd/
// RUN: chmod +x %t/*/poseidon-calibrate-ops-*
// RUN: not %t/init/poseidon-calibrate --gpu 1 --out %t/cm/ 2>&1 | FileCheck --check-prefix=INIT --implicit-check-not=MI250X %s
// RUN: not %t/start/poseidon-calibrate --gpu 1 --out %t/cm/ 2>&1 | FileCheck --check-prefix=START %s
// RUN: not %t/none/poseidon-calibrate --gpu 1 --out %t/cm/ 2>&1 | FileCheck --check-prefix=NONE %s
// RUN: not %t/amd/poseidon-calibrate --gpu 1 --out %t/cm/ 2>&1 | FileCheck --check-prefix=AMD %s

// INIT: poseidon-calibrate: the CUDA microbenchmark could not initialize device 1: initialization error
// START: poseidon-calibrate: the HIP microbenchmark poseidon-calibrate-ops-hip could not be started (No such file or directory): poseidon-calibrate-ops-hip: error while loading shared libraries: libamdhip64.so.7: cannot open shared object file; the microbenchmarks before it found no device:
// START-NEXT: poseidon-calibrate-ops-cuda: no CUDA-capable device is detected
// NONE: poseidon-calibrate: no GPU with id 1 is visible to this build's microbenchmarks:
// NONE-NEXT: poseidon-calibrate-ops-cuda: no CUDA-capable device is detected
// AMD: poseidon-calibrate: device MI250X (gfx90a, id 1)
