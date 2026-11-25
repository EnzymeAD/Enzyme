// A full poseidon-calibrate run on an AMD GPU: every arm that measures one
// runs the HIP build of its calibrator, under the name tools/CMakeLists.txt
// gives it. The stand-ins exist under those names only, so an arm that looks
// for another name fails the run.
//
// RUN: rm -rf %t && mkdir -p %t/bin %t/cm
// RUN: cp %poseidon_calibrate %t/bin/
// RUN: sed 's/^# native_arch=sm_120$/# native_arch=gfx942/' %cost_models/cm_sm_120_synthetic.csv > %t/ops.csv
// RUN: printf '#!/bin/sh\nif [ "$1" = --device ]; then printf "AMD Instinct MI300A\\tgfx942\\n"; else cat %t/ops.csv; fi\n' > %t/bin/poseidon-calibrate-ops-hip
// RUN: printf '#!/bin/sh\ncase "$1:$4" in ozaki:) echo library_ozaki_rel,dgemm,1.0;; ozaki:*) echo library_ozaki_rect_rel,nm8_m125k648,2.0;; direct:) echo library_direct_rel,f16_f32,0.5;; esac\n' > %t/bin/poseidon-calibrate-library-hip
// RUN: chmod +x %t/bin/poseidon-calibrate-ops-hip %t/bin/poseidon-calibrate-library-hip
// RUN: %t/bin/poseidon-calibrate --gpu 0 --ozaki-rect 125x648 --out %t/cm/cm.csv | FileCheck %s

// CHECK: poseidon-calibrate: device MI300A (gfx942, id 0)
// CHECK: poseidon-calibrate: wrote {{.*}}cm.csv
// CHECK: library_ozaki_rel,dgemm,1.0
// CHECK: library_ozaki_rect_rel,nm8_m125k648,2.0
// CHECK: poseidon-calibrate: tcec: the error-corrected dispatch has no HIP runtime; its row stays absent
// CHECK: library_direct_rel,f16_f32,0.5
// CHECK: poseidon-calibrate: inkernel: tensor-core raises inside a kernel are NVPTX-only; their rows stay absent
// CHECK: poseidon-calibrate: cost model: {{.*}}cm.csv
