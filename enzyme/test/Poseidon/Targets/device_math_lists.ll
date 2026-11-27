; RUN: sed -n 's/^ *{"\([a-z0-9]*\)", [a-z]*, [a-z]*, true,.*/\1/p' %S/../../../Poseidon/Targets/MathLibrary.h | sort > %t.device
; RUN: sed -n '/__CLANG_CUDA_LIBDEVICE_DECLARES_H__/,/^#elif/s/^ *POSEIDON_KEEP(\([a-z0-9]*\));$/\1/p' %S/../../../include/poseidon/poseidon.h | sort > %t.nvptx
; RUN: sed -n '/__CLANG_HIP_LIBDEVICE_DECLARES_H__/,/^#endif/s/^ *POSEIDON_KEEP(\([a-z0-9]*\));$/\1/p' %S/../../../include/poseidon/poseidon.h | sort > %t.amdgpu
; RUN: grep -x sin %t.nvptx
; RUN: grep -x sin %t.amdgpu
; RUN: diff %t.device %t.nvptx
; RUN: grep -vxF -e ceil -e copysign -e fabs -e floor -e fma -e fmax -e fmin -e powi -e rint -e round -e sqrt -e trunc %t.device > %t.amdgpu.expected
; RUN: diff %t.amdgpu.expected %t.amdgpu

; The device math functions include/poseidon/poseidon.h links are its own
; literal lists, which have to be the Device column of kMathFunctions
; (Targets/MathLibrary.h): all of it on NVPTX, and on AMDGPU all of it but the
; functions the header comment of MathLibrary.h names.
