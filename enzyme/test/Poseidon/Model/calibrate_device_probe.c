// How poseidon-calibrate finds the GPU it measures, with stand-ins for the
// microbenchmarks' --device probe: one that finds no device of its vendor
// (exit status 2) passes the question to the next vendor's; one that cannot
// initialize the device (exit status 1) or cannot be started ends the run with
// its reason, and the next vendor's is never asked; with no device anywhere the
// run lists what each probe said.
//
// RUN: rm -rf %t && mkdir -p %t/init %t/start %t/none %t/amd %t/cm
// RUN: for d in init start none amd; do cp %poseidon_calibrate %t/$d/; done
// RUN: printf '#!/bin/sh\necho initialization error >&2\nexit 1\n' > %t/init/poseidon-microbm
// RUN: printf '#!/bin/sh\nprintf "AMD Instinct MI250X\\tgfx90a\\n"\n' > %t/init/poseidon-microbm-hip
// RUN: printf '#!/bin/sh\necho no CUDA-capable device is detected >&2\nexit 2\n' > %t/start/poseidon-microbm
// RUN: printf '#!/bin/sh\necho "poseidon-microbm-hip: error while loading shared libraries: libamdhip64.so.7: cannot open shared object file" >&2\nexit 127\n' > %t/start/poseidon-microbm-hip
// RUN: cp %t/start/poseidon-microbm %t/none/ && cp %t/start/poseidon-microbm %t/amd/
// RUN: cp %t/init/poseidon-microbm-hip %t/amd/
// RUN: chmod +x %t/*/poseidon-microbm*
// RUN: not %t/init/poseidon-calibrate --gpu 1 --out %t/cm/ 2>&1 | FileCheck --check-prefix=INIT --implicit-check-not=MI250X %s
// RUN: not %t/start/poseidon-calibrate --gpu 1 --out %t/cm/ 2>&1 | FileCheck --check-prefix=START %s
// RUN: not %t/none/poseidon-calibrate --gpu 1 --out %t/cm/ 2>&1 | FileCheck --check-prefix=NONE %s
// RUN: not %t/amd/poseidon-calibrate --gpu 1 --out %t/cm/ 2>&1 | FileCheck --check-prefix=AMD %s

// INIT: poseidon-calibrate: the CUDA microbenchmark could not initialize device 1: initialization error
// START: poseidon-calibrate: the HIP microbenchmark poseidon-microbm-hip could not be started (No such file or directory): poseidon-microbm-hip: error while loading shared libraries: libamdhip64.so.7: cannot open shared object file; the microbenchmarks before it found no device:
// START-NEXT: poseidon-microbm: no CUDA-capable device is detected
// NONE: poseidon-calibrate: no GPU with id 1 is visible to this build's microbenchmarks:
// NONE-NEXT: poseidon-microbm: no CUDA-capable device is detected
// AMD: [calibrate] device MI250X (gfx90a, id 1)
