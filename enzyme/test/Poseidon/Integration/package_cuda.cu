// RUN: rm -rf %t && mkdir -p %t
// RUN: %cmake %cmake_generator -S %S/../Inputs/package -B %t/gen -DCMAKE_CXX_COMPILER=%clangxx -DCMAKE_CUDA_COMPILER=%clangxx -DCMAKE_CUDA_ARCHITECTURES=OFF "-DCMAKE_CUDA_FLAGS=--cuda-gpu-arch=%gpu_arch -O3" -DCUDAToolkit_ROOT=/usr/local/cuda -DWITH_CUDA=ON -DTWO_TU=%S/../Inputs/two_tu -DPoseidon_DIR=%poseidon_package_dir -DPOSEIDON_MODE=instrument > /dev/null
// RUN: %cmake --build %t/gen > %t/gen.log 2>&1
// RUN: env POSEIDON_PROFILE_DIR=%t/prof %t/gen/app | FileCheck --check-prefix=OUT %s
// RUN: ls %t/prof | FileCheck --check-prefix=PROF %s
// RUN: cat %t/prof/preprocess__Z9hooke_volRK4mat3fPf.poseidon._Z6vol_atPKffi.0.fpprofile | FileCheck --check-prefix=MERGED %s
// RUN: %cmake %cmake_generator -S %S/../Inputs/package -B %t/use -DCMAKE_CXX_COMPILER=%clangxx -DCMAKE_CUDA_COMPILER=%clangxx -DCMAKE_CUDA_ARCHITECTURES=OFF "-DCMAKE_CUDA_FLAGS=--cuda-gpu-arch=%gpu_arch -O3" -DCUDAToolkit_ROOT=/usr/local/cuda -DWITH_CUDA=ON -DTWO_TU=%S/../Inputs/two_tu -DPoseidon_DIR=%poseidon_package_dir -DPOSEIDON_MODE=optimize -DPOSEIDON_PROFILE=%t/prof "-DPOSEIDON_FLAGS=-poseidon-tau=1e-6 -poseidon-cost-model=%gpu_cost_model -poseidon-enable-herbie=0 -poseidon-print" > /dev/null
// RUN: %cmake --build %t/use > %t/use.log 2>&1
// RUN: FileCheck --check-prefix=USE %s < %t/use.log
// RUN: %t/use/app | FileCheck --check-prefix=STATUS %s
//
// REQUIRES: poseidon, enzyme, cuda-runtime

// MFEM's shape through the CMake package: a static library of two CUDA units
// built without relocatable device code against Poseidon::ProfileGenerate,
// linked into an application, profiles every site of both units; the build
// against Poseidon::ProfileUse reads that profile back.

// OUT: axpb 2703.7489499999992 lift 367.08090963891368 forall_a 121902.63888375004 forall_b -7593.8872846631084 vol_a -0.549021492 vol_b -0.219608603
// OUT-NEXT: status no error

// PROF-DAG: preprocess__Z4axpbPdPKdS1_i_poseidon_body.fpprofile
// PROF-DAG: preprocess__Z4liftPdPKdi_poseidon_body.fpprofile
// PROF-DAG: preprocess__Z9hooke_volRK4mat3fPf.poseidon._Z6vol_atPKffi.0.fpprofile
// PROF-DAG: preprocess__ZL10CuKernel1DIZ5run_aPdS0_PKdS2_PKfPfiiEUliE_EviT__poseidon_body.fpprofile
// PROF-DAG: preprocess__ZL10CuKernel1DIZ5run_bPdS0_PKdPKfPfiiEUliE_EviT__poseidon_body.fpprofile

// MERGED: LaunchCount = 96
// MERGED: Exec = 8192

// USE-DAG: [poseidon] preprocess__Z4liftPdPKdi_poseidon_body: tau=1.000000e-06
// USE-DAG: [poseidon] preprocess__ZL10CuKernel1DIZ5run_aPdS0_PKdS2_PKfPfiiEUliE_EviT__poseidon_body: tau=1.000000e-06

// STATUS: status no error
