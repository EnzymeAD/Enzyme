// Without cuMpSGEMM the tcec arm writes no row and says why, and --check does
// not count the row as missing.
// UNSUPPORTED: tcec-cumpsgemm
// RUN: rm -rf %t && mkdir -p %t/bin %t/cm
// RUN: cp %poseidon_calibrate %t/bin/
// RUN: printf '#!/bin/sh\nprintf "NVIDIA GeForce RTX 5090\\tsm_120\\n"\n' > %t/bin/poseidon-calibrate-ops-cuda
// RUN: chmod +x %t/bin/poseidon-calibrate-ops-cuda
// RUN: cp %cost_models/cm_sm_120_synthetic_library.csv %t/cm/cm.csv
// RUN: %t/bin/poseidon-calibrate --gpu 0 --only tcec --out %t/cm/cm.csv | FileCheck --check-prefix=ARM %s
// RUN: diff %cost_models/cm_sm_120_synthetic_library.csv %t/cm/cm.csv
// RUN: not %t/bin/poseidon-calibrate --gpu 0 --check --out %t/cm/cm.csv | FileCheck --check-prefix=CHECK %s

// ARM: poseidon-calibrate: tcec: this build has no error-corrected library call, which needs cuMpSGEMM (configure with -DPOSEIDON_CUMPSGEMM_ROOT=<checkout>); its row stays absent
// CHECK-NOT: MISSING {{.*}}library_tcec_rel
// CHECK: n/a         library_tcec_rel  (error-corrected library call: none in a build without cuMpSGEMM)
