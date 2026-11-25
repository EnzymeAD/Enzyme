// A full poseidon-calibrate run on an NVIDIA GPU without --profile measures
// every arm it can and says that the in-kernel arm needs the profile; asking
// for that arm alone without it is an error. A device query that fails says
// why. The GPU tools are stand-ins.
//
// UNSUPPORTED: tcec-cumpsgemm
// RUN: rm -rf %t && mkdir -p %t/bin %t/cm %t/bad
// RUN: cp %poseidon_calibrate %t/bin/ && cp %poseidon_calibrate %t/bad/
// RUN: printf '#!/bin/sh\nif [ "$1" = --device ]; then printf "NVIDIA GeForce RTX 5090\\tsm_120\\n"; else cat %cost_models/cm_sm_120_synthetic.csv; fi\n' > %t/bin/poseidon-microbm
// RUN: printf '#!/bin/sh\ncase "$1" in ozaki) echo ozaki_dispatch_rel,dgemm,1.0;; direct) echo direct_dispatch_rel,f16_f32,0.5;; esac\n' > %t/bin/poseidon-dispatch-calibrate
// RUN: chmod +x %t/bin/poseidon-microbm %t/bin/poseidon-dispatch-calibrate
// RUN: %t/bin/poseidon-calibrate --gpu 0 --out %t/cm/cm.csv | FileCheck --check-prefix=FULL %s
// RUN: not %t/bin/poseidon-calibrate --gpu 0 --only inkernel --out %t/cm/cm.csv 2>&1 | FileCheck --check-prefix=ONLY %s
// RUN: printf '#!/bin/sh\necho "cudaGetDeviceProperties: CUDA driver version is insufficient for CUDA runtime version" >&2\nexit 1\n' > %t/bad/poseidon-microbm
// RUN: chmod +x %t/bad/poseidon-microbm
// RUN: not %t/bad/poseidon-calibrate --gpu 0 --out %t/cm/ 2>&1 | FileCheck --check-prefix=WHY %s

// FULL: [calibrate] device RTX5090 (sm_120, id 0)
// FULL: ozaki_dispatch_rel,dgemm,1.0
// FULL: direct_dispatch_rel,f16_f32,0.5
// FULL: [calibrate] inkernel: not measured; it needs --profile <dir> (see --help)
// FULL-NEXT: [calibrate] cost model: {{.*}}cm.csv
// ONLY: poseidon-calibrate: the in-kernel arm needs --profile <dir>
// WHY: poseidon-calibrate: no GPU with id 0 is visible to this build's microbenchmarks:
// WHY-NEXT: poseidon-microbm: cudaGetDeviceProperties: CUDA driver version is insufficient for CUDA runtime version
