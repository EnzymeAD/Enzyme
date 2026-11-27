// A full poseidon-calibrate run on an AMD GPU: every arm runs the HIP build of
// its calibrator, under the only names the stand-ins exist under. A device ID
// several products share is named after the first.
// RUN: rm -rf %t && mkdir -p %t/bin %t/cm %t/name/bin %t/name/cm %t/name/xdg
// RUN: cp %poseidon_calibrate %t/bin/
// RUN: sed 's/^# native_arch=sm_120$/# native_arch=gfx942/' %cost_models/cm_sm_120_synthetic.csv > %t/ops.csv
// RUN: printf '#!/bin/sh\nif [ "$1" = --device ]; then printf "AMD Instinct MI300A\\tgfx942\\n"; else cat %t/ops.csv; fi\n' > %t/bin/poseidon-calibrate-ops-hip
// RUN: printf '#!/bin/sh\ncase "$1:$4" in ozaki:) echo library_ozaki_rel,dgemm,1.0;; ozaki:*) echo library_ozaki_rect_rel,nm8_m125k648,2.0;; direct:) echo library_direct_rel,f16_f32,0.5;; operand-min:f16_f32) echo library_direct_operand_min,f16_f32,6.103515625e-05;; esac\n' > %t/bin/poseidon-calibrate-library-hip
// RUN: printf '#!/bin/sh\nprintf "stream_load,double,0.024\\nstream_load,float,0.012\\nstream_store,double,0.031\\nstream_store,float,0.018\\nstream_combine,pnorm,12\\nstream_combine,band,0.02\\n"\n' > %t/bin/poseidon-calibrate-stream-hip
// RUN: chmod +x %t/bin/poseidon-calibrate-ops-hip %t/bin/poseidon-calibrate-stream-hip %t/bin/poseidon-calibrate-library-hip
// RUN: %t/bin/poseidon-calibrate --gpu 0 --ozaki-rect 125x648 --out %t/cm/cm.csv | FileCheck %s
// RUN: cp %poseidon_calibrate %t/name/bin/
// RUN: printf '#!/bin/sh\nprintf "AMD Instinct MI250X / MI250\\tgfx90a\\n"\n' > %t/name/bin/poseidon-calibrate-ops-hip
// RUN: chmod +x %t/name/bin/poseidon-calibrate-ops-hip
// RUN: sed 's/^# native_arch=sm_120$/# native_arch=gfx90a/' %cost_models/cm_sm_120_synthetic.csv > %t/name/cm/cm_gfx90a_MI250X.csv
// RUN: not env PATH=%t/name/bin XDG_CACHE_HOME=%t/name/xdg %t/name/bin/poseidon-calibrate --gpu 0 --check --out %t/name/cm/ | FileCheck --check-prefix=NAME %s

// CHECK: poseidon-calibrate: device MI300A (gfx942, id 0)
// CHECK: poseidon-calibrate: wrote {{.*}}cm.csv
// CHECK: poseidon-calibrate: measuring streamed memory traffic
// CHECK-NEXT: stream_load,double,0.024
// CHECK-NEXT: stream_load,float,0.012
// CHECK-NEXT: stream_store,double,0.031
// CHECK-NEXT: stream_store,float,0.018
// CHECK-NEXT: stream_combine,pnorm,12
// CHECK-NEXT: stream_combine,band,0.02
// CHECK: library_ozaki_rel,dgemm,1.0
// CHECK: library_ozaki_rect_rel,nm8_m125k648,2.0
// CHECK: poseidon-calibrate: tcec: the error-corrected library call has no HIP runtime; its row stays absent
// CHECK: library_direct_rel,f16_f32,0.5
// CHECK: poseidon-calibrate: inkernel: tensor-core raises inside a kernel are NVPTX-only; their rows stay absent
// CHECK: poseidon-calibrate: operand-min: tensor-core raises inside a kernel are NVPTX-only; their rows stay absent
// CHECK: library_direct_operand_min,f16_f32,6.103515625e-05
// CHECK: poseidon-calibrate: cost model: {{.*}}cm.csv

// NAME: cost model  {{.*}}/cm/cm_gfx90a_MI250X.csv
// NAME-NEXT: device      MI250X (gfx90a, id 0)
// NAME-NEXT: native_arch gfx90a  ok
