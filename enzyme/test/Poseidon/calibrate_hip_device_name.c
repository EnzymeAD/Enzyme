// poseidon-calibrate names an AMD GPU after the marketing name the HIP runtime
// reports (poseidon-microbm-hip --device, here a stand-in): a device ID that
// several products share lists them all, and the first names the device and
// so its cost model.
//
// RUN: rm -rf %t && mkdir -p %t/bin %t/cm %t/xdg
// RUN: cp %poseidon_calibrate %t/bin/
// RUN: printf '#!/bin/sh\nprintf "AMD Instinct MI250X / MI250\\tgfx90a\\n"\n' > %t/bin/poseidon-microbm-hip
// RUN: chmod +x %t/bin/poseidon-microbm-hip
// RUN: sed 's/^# native_arch=sm_120$/# native_arch=gfx90a/' %S/Inputs/cm_gpu_fixture_x1e6.csv > %t/cm/cm_gfx90a_MI250X.csv
// RUN: (env PATH=%t/bin XDG_CACHE_HOME=%t/xdg %t/bin/poseidon-calibrate --gpu 0 --check --out %t/cm/ || true) | FileCheck %s

// CHECK: cost model  {{.*}}/cm/cm_gfx90a_MI250X.csv
// CHECK-NEXT: device      MI250X (gfx90a, id 0)
// CHECK-NEXT: native_arch gfx90a  ok
