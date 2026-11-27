// A full poseidon-calibrate run on an NVIDIA GPU (stand-ins for the GPU tools
// and poseidon-clang++) measures every arm it can, the in-kernel rows and then
// the operand floors last.
// UNSUPPORTED: tcec-cumpsgemm
// RUN: rm -rf %t && mkdir -p %t/bin %t/cm
// RUN: cp %poseidon_calibrate %t/bin/
// RUN: cp %S/../../Inputs/fake-poseidon-clang.sh %t/bin/poseidon-clang++
// RUN: printf '#!/bin/sh\nif [ "$1" = --device ]; then printf "NVIDIA GeForce RTX 5090\\tsm_120\\n"; else cat %cost_models/cm_sm_120_synthetic.csv; fi\n' > %t/bin/poseidon-calibrate-ops-cuda
// RUN: printf '#!/bin/sh\ncase "$1" in ozaki) echo library_ozaki_rel,dgemm,1.0;; direct) echo library_direct_rel,f16_f32,0.5;; operand-min) echo library_direct_operand_min,f16_f32,6.103515625e-05;; esac\n' > %t/bin/poseidon-calibrate-library-cuda
// RUN: printf '#!/bin/sh\nprintf "stream_load,double,0.024\\nstream_load,float,0.012\\nstream_store,double,0.031\\nstream_store,float,0.018\\nstream_combine,pnorm,12\\nstream_combine,band,0.02\\n"\n' > %t/bin/poseidon-calibrate-stream-cuda
// RUN: chmod +x %t/bin/poseidon-calibrate-ops-cuda %t/bin/poseidon-calibrate-stream-cuda %t/bin/poseidon-calibrate-library-cuda %t/bin/poseidon-clang++
// RUN: %t/bin/poseidon-calibrate --gpu 0 --out %t/cm/cm.csv | FileCheck --check-prefix=FULL %s

// FULL: poseidon-calibrate: device RTX5090 (sm_120, id 0)
// FULL: poseidon-calibrate: measuring streamed memory traffic
// FULL-NEXT: stream_load,double,0.024
// FULL-NEXT: stream_load,float,0.012
// FULL-NEXT: stream_store,double,0.031
// FULL-NEXT: stream_store,float,0.018
// FULL-NEXT: stream_combine,pnorm,12
// FULL-NEXT: stream_combine,band,0.02
// FULL: library_ozaki_rel,dgemm,1.0
// FULL: library_direct_rel,f16_f32,0.5
// FULL: poseidon-calibrate: in-kernel raises at 2048x2048x2048, B strided along K
// FULL: + tcec_n2_tf32_f32 0.5000 ms  rel 0.250000
// FULL-NEXT: poseidon-calibrate: operand-min workdir {{.*}}
// FULL: + inkernel_operand_min,tcec_n2_tf32_f32,1.1479437019748901e-41
// FULL-NEXT: poseidon-calibrate: measuring the operand floors of the library calls
// FULL-NEXT: + library_direct_operand_min,f16_f32,6.103515625e-05
// FULL-NEXT: poseidon-calibrate: cost model: {{.*}}cm.csv
