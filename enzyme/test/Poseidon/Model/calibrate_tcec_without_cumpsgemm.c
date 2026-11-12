// A poseidon-calibrate built without cuMpSGEMM has no error-corrected dispatch
// to measure: its tcec arm writes no row and says why, and its check does not
// count the row as missing. The device is nvidia-smi's answer, here a
// stand-in.
//
// UNSUPPORTED: tcec-cumpsgemm
// RUN: rm -rf %t && mkdir -p %t/bin %t/cm
// RUN: cp %poseidon_calibrate %t/bin/
// RUN: printf '#!/bin/sh\nprintf "NVIDIA GeForce RTX 5090, 12.0\\n"\n' > %t/bin/nvidia-smi
// RUN: chmod +x %t/bin/nvidia-smi
// RUN: cp %cost_models/cm_sm_120_synthetic_dispatch.csv %t/cm/cm.csv
// RUN: env PATH=%t/bin %t/bin/poseidon-calibrate --gpu 0 --only tcec --out %t/cm/cm.csv | FileCheck --check-prefix=ARM %s
// RUN: diff %cost_models/cm_sm_120_synthetic_dispatch.csv %t/cm/cm.csv
// RUN: (env PATH=%t/bin %t/bin/poseidon-calibrate --gpu 0 --check --out %t/cm/cm.csv || true) | FileCheck --check-prefix=CHECK %s

// ARM: [calibrate] tcec: this build has no error-corrected dispatch, which needs cuMpSGEMM (configure with -DPOSEIDON_TCEC_USE_CUMPSGEMM=ON -DPOSEIDON_CUMPSGEMM_ROOT=<checkout>); its row stays absent
// CHECK: n/a         tcec_dispatch_rel  (error-corrected host dispatch: none in a build without cuMpSGEMM)
