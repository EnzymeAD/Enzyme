//===- KokkosUtils.h - Helpers for Kokkos differentiation -------*- C++ -*-===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_KOKKOS_UTILS_H
#define ENZYME_KOKKOS_UTILS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"

#include <optional>
#include <string>

#include "Utils.h"

namespace enzyme_kokkos {

/// Quick pre-check before invoking llvm::demangle so non-Kokkos symbols pay
/// near-zero cost.
inline bool mayBeKokkosRelevantSymbol(llvm::StringRef Name) {
  return startsWith(Name, "__enzyme_kokkos_") || Name.contains("Kokkos") ||
         Name.contains("desul") || Name.contains("kokkosp") ||
         Name.contains("mutex") || Name.contains("cuda") ||
         Name.contains("hip");
}

/// Demangles \p Name (if potentially Kokkos/Desul/runtime related) using
/// \p DemangleFn and normalizes C++ ABI inline namespaces (`std::__u::`,
/// `std::__1::`, `std::__cxx11::` -> `std::`) and `> >` -> `>>`.
template <typename DemangleFnT>
inline std::string normalizeKokkosDemangledNameWith(llvm::StringRef Name,
                                                    DemangleFnT &&DemangleFn) {
  if (Name.empty() || !mayBeKokkosRelevantSymbol(Name))
    return {};
  std::string D = DemangleFn(Name);
  auto replaceAll = [&](const char *From, size_t FromLen, const char *To,
                        size_t ToLen) {
    size_t Pos = 0;
    while ((Pos = D.find(From, Pos)) != std::string::npos) {
      D.replace(Pos, FromLen, To, ToLen);
      Pos += ToLen;
    }
  };
  replaceAll("> >", 3, ">>", 2);
  replaceAll("std::__u::", 10, "std::", 5);
  replaceAll("std::__1::", 10, "std::", 5);
  replaceAll("std::__cxx11::", 14, "std::", 5);
  return D;
}

#define ENZYME_KOKKOS_DEMANGLE(Name)                                           \
  enzyme_kokkos::normalizeKokkosDemangledNameWith(                             \
      (Name), [](llvm::StringRef S) { return llvm::demangle(S.str()); })

/// Strip leading return type from a normalized demangled C++ function signature
/// up to the qualified name (e.g. `bool desul::Impl::lock_address<...>(...)` ->
/// `desul::Impl::lock_address<...>(...)`).
///
/// The return type may itself live in the namespace being searched for (e.g.
/// `Kokkos::Impl::SharedAllocationRecord<void, void>*
/// Kokkos::Impl::make_shared_allocation_record<...>(...)`), so rather than
/// searching for the first occurrence of \p NS we locate the qualified name
/// as the token following the last top-level space before the parameter list.
inline llvm::StringRef stripReturnTypeToNamespace(llvm::StringRef NormDem,
                                                  llvm::StringRef NS) {
  if (NormDem.empty())
    return {};
  size_t AngleDepth = 0;
  size_t ParenDepth = 0;
  size_t LastTopSpace = llvm::StringRef::npos;
  size_t PrevTopSpace = llvm::StringRef::npos;
  size_t i = 0;
  for (; i < NormDem.size(); ++i) {
    char c = NormDem[i];
    if (c == '<' && ParenDepth == 0) {
      AngleDepth++;
    } else if (c == '>' && ParenDepth == 0) {
      if (AngleDepth > 0)
        AngleDepth--;
    } else if (c == '(') {
      if (AngleDepth == 0 && ParenDepth == 0)
        break;
      ParenDepth++;
    } else if (c == ')') {
      if (ParenDepth > 0)
        ParenDepth--;
    } else if (c == ' ' && AngleDepth == 0 && ParenDepth == 0) {
      PrevTopSpace = LastTopSpace;
      LastTopSpace = i;
    }
  }
  // `operator new` / `operator delete` / `operator new[]` contain a space in
  // the function name itself.
  if (LastTopSpace != llvm::StringRef::npos &&
      endsWith(NormDem.substr(0, LastTopSpace), "operator"))
    LastTopSpace = PrevTopSpace;
  size_t Start = (LastTopSpace == llvm::StringRef::npos) ? 0 : LastTopSpace + 1;
  while (
      Start < NormDem.size() &&
      (NormDem[Start] == '*' || NormDem[Start] == '&' || NormDem[Start] == ' '))
    Start++;
  llvm::StringRef Res = NormDem.substr(Start);
  if (startsWith(Res, NS))
    return Res;
  // Fallback: the first occurrence of NS at a token boundary that is not
  // itself a return type (i.e. the demangled name has no top-level space).
  if (LastTopSpace == llvm::StringRef::npos && startsWith(NormDem, NS))
    return NormDem;
  return {};
}

inline bool isKokkosPreservedFunctionMangled(llvm::StringRef Name) {
  if (Name.empty() || !mayBeKokkosRelevantSymbol(Name))
    return false;
  if (Name.contains("6Kokkos")) {
    if (((Name.contains("6Kokkos5Tools") ||
          Name.contains("6Kokkos9Profiling")) &&
         !Name.contains("6Kokkos5Tools4Impl") &&
         !Name.contains("6Kokkos5Tools12Experimental4Impl") &&
         !Name.contains("19profile_fence_event")) ||
        Name.contains("6Kokkos4Impl29hostspace_parallel_zeromemset") ||
        Name.contains("6Kokkos4Impl27hostspace_parallel_deepcopy") ||
        startsWith(Name, "_ZN6Kokkos5fenceE") ||
        Name.contains("6Kokkos4Impl11HostBarrier") ||
        (Name.contains("6Kokkos4Impl18HostThreadTeamData") &&
         (Name.contains("15team_rendezvous") ||
          Name.contains("10rendezvous"))) ||
        (Name.contains("6Kokkos4Impl20HostThreadTeamMember") &&
         Name.contains("12team_barrierEv")) ||
        ((Name.contains("6Kokkos6Serial") || Name.contains("6Kokkos6OpenMP") ||
          Name.contains("6Kokkos7Threads") || Name.contains("6Kokkos4Cuda") ||
          Name.contains("6Kokkos3HIPE") ||
          Name.contains("6Kokkos12Experimental3HIP")) &&
         (Name.contains("17impl_static_fenceE") || Name.contains("5fenceE")) &&
         !Name.contains("19profile_fence_event")))
      return true;
  }
  return false;
}

/// Synchronization / fence calls across Kokkos, Desul, CUDA, and HIP.
/// Excludes `ThreadsInternal::{fence,internal_fence}` which have custom
/// reverse-mode sequencing.
inline bool isKokkosFenceCall(llvm::StringRef Name, llvm::StringRef NormDem) {
  if (Name == "cudaDeviceSynchronize" || Name == "cudaStreamSynchronize" ||
      Name == "hipDeviceSynchronize" || Name == "hipStreamSynchronize")
    return true;
  if (!NormDem.empty()) {
    llvm::StringRef D = stripReturnTypeToNamespace(NormDem, "desul::Impl::");
    if (startsWith(D, "desul::Impl::host_atomic_thread_fence(") ||
        startsWith(D, "desul::Impl::device_atomic_thread_fence("))
      return true;
    llvm::StringRef K = stripReturnTypeToNamespace(NormDem, "Kokkos::");
    if (!K.empty()) {
      if (startsWith(K, "Kokkos::fence(") ||
          startsWith(K, "Kokkos::memory_fence(") ||
          startsWith(K, "Kokkos::Impl::hostspace_fence(") ||
          startsWith(K, "Kokkos::Serial::fence(") ||
          startsWith(K, "Kokkos::Serial::impl_static_fence(") ||
          startsWith(K, "Kokkos::Impl::SerialInternal::fence(") ||
          startsWith(K, "Kokkos::OpenMP::fence(") ||
          startsWith(K, "Kokkos::OpenMP::impl_static_fence(") ||
          startsWith(K, "Kokkos::Threads::fence(") ||
          startsWith(K, "Kokkos::Threads::impl_static_fence(") ||
          startsWith(K, "Kokkos::Cuda::fence(") ||
          startsWith(K, "Kokkos::Cuda::impl_static_fence(") ||
          startsWith(K, "Kokkos::Impl::CudaInternal::fence(") ||
          startsWith(K, "Kokkos::HIP::fence(") ||
          startsWith(K, "Kokkos::HIP::impl_static_fence(") ||
          startsWith(K, "Kokkos::Impl::HIPInternal::fence(") ||
          startsWith(K, "Kokkos::Experimental::SYCL::fence(") ||
          startsWith(K, "Kokkos::Experimental::SYCL::impl_static_fence("))
        return true;
    }
  }
  return false;
}

/// Team-level barriers (`HostThreadTeamMember::team_barrier`,
/// `HostThreadTeamData::team_rendezvous[_release]`). Replayed verbatim in the
/// reverse pass so reverse team loops keep the same synchronization.
inline bool isKokkosTeamBarrierCall(llvm::StringRef Name,
                                    llvm::StringRef NormDem) {
  if (NormDem.empty())
    return false;
  llvm::StringRef K = stripReturnTypeToNamespace(NormDem, "Kokkos::");
  if (K.empty())
    return false;
  if ((startsWith(K, "Kokkos::Impl::HostThreadTeamMember<") ||
       startsWith(K, "Kokkos::Impl::ThreadsExecTeamMember::") ||
       startsWith(K, "Kokkos::Impl::OpenMPExecTeamMember::")) &&
      (K.contains("::team_barrier(") || K.contains("::team_rendezvous(") ||
       K.contains("::team_rendezvous_release(")))
    return true;
  if (startsWith(K, "Kokkos::Impl::HostThreadTeamData::") &&
      (K.contains("::team_rendezvous(") ||
       K.contains("::team_rendezvous_release(") ||
       K.contains("::rendezvous(") || K.contains("::rendezvous_release(")))
    return true;
  return false;
}

inline bool isKokkosInactiveCall(llvm::StringRef Name,
                                 llvm::StringRef NormDem) {
  if (startsWith(Name, "__enzyme_kokkos_") ||
      isKokkosFenceCall(Name, NormDem) ||
      isKokkosTeamBarrierCall(Name, NormDem))
    return true;

  if (Name == "cudaFuncGetAttributes" ||
      Name == "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags" ||
      Name == "cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags" ||
      Name == "cudaGetLastError" || Name == "cudaPeekAtLastError" ||
      Name == "cudaGetDevice" || Name == "cudaGetDeviceCount" ||
      Name == "cudaGetDeviceProperties" || Name == "cudaDeviceGetAttribute" ||
      Name == "hipFuncGetAttributes" ||
      Name == "hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags" ||
      Name == "hipGetLastError" || Name == "hipPeekAtLastError" ||
      Name == "hipGetDevice" || Name == "hipGetDeviceCount")
    return true;

  if (NormDem.empty())
    return false;
  llvm::StringRef K = stripReturnTypeToNamespace(NormDem, "Kokkos::");
  if (K.empty())
    return false;

  if (startsWith(K, "Kokkos::Tools::Impl::begin_parallel_") ||
      startsWith(K, "Kokkos::Tools::Impl::ToolResponse<"))
    return false;

  static const char *const InactivePrefixes[] = {
      "Kokkos::Tools::",
      "Kokkos::Profiling::",
      "Kokkos::is_initialized(",
      "Kokkos::is_finalized(",
      "Kokkos::print_configuration(",
      "Kokkos::device_id(",
      "Kokkos::num_devices(",
      "Kokkos::num_threads(",
      "Kokkos::show_warnings(",
      "Kokkos::tune_internals(",
      "Kokkos::abort(",
      "Kokkos::Impl::host_abort(",
      "Kokkos::Impl::throw_runtime_exception(",
      "Kokkos::Impl::throw_bad_alloc(",
      "Kokkos::Impl::log_warning(",
      "Kokkos::Impl::CheckUsage<",
      "Kokkos::Impl::ParallelConstructName<",
      "Kokkos::Impl::check_execution_space_constructor_precondition(",
      "Kokkos::Impl::check_execution_space_destructor_precondition(",
      "Kokkos::Impl::fill_host_accessible_header_info(",
      "Kokkos::Impl::team_policy_check_valid_",
      "Kokkos::Impl::find_hash_size(",
      "Kokkos::Impl::spinwait_while_equal",
      "Kokkos::Impl::spinwait_until_equal",
      "Kokkos::Impl::wait_yield(",
      "Kokkos::Impl::HostBarrier::",
      "Kokkos::Impl::HostThreadTeamData::organize_pool(",
      "Kokkos::Impl::HostThreadTeamData::disband_pool(",
      "Kokkos::Impl::HostThreadTeamData::organize_team(",
      "Kokkos::Impl::HostThreadTeamData::disband_team(",
      "Kokkos::OpenMP::OpenMP(",
      "Kokkos::OpenMP::~OpenMP(",
      "Kokkos::OpenMP::name(",
      "Kokkos::OpenMP::concurrency(",
      "Kokkos::OpenMP::print_configuration(",
      "Kokkos::OpenMP::impl_is_initialized(",
      "Kokkos::OpenMP::in_parallel(",
      "Kokkos::OpenMP::impl_thread_pool_size(",
      "Kokkos::OpenMP::impl_thread_pool_rank(",
      "Kokkos::OpenMP::impl_hardware_thread_id(",
      "Kokkos::OpenMP::impl_max_hardware_threads(",
      "Kokkos::OpenMP::impl_instance_id(",
      "Kokkos::Cuda::Cuda(",
      "Kokkos::Cuda::~Cuda(",
      "Kokkos::Cuda::name(",
      "Kokkos::Cuda::concurrency(",
      "Kokkos::Cuda::cuda_stream(",
      "Kokkos::Cuda::cuda_device(",
      "Kokkos::Cuda::cuda_device_prop(",
      "Kokkos::Cuda::impl_instance_id(",
      "Kokkos::Cuda::impl_internal_space_instance(",
      "Kokkos::Cuda::print_configuration(",
      "Kokkos::Impl::cuda_internal_error_throw(",
      "Kokkos::Impl::cuda_prefetch_pointer(",
      "Kokkos::Impl::CudaInternal::verify_is_initialized(",
      "Kokkos::HIP::HIP(",
      "Kokkos::HIP::~HIP(",
      "Kokkos::HIP::name(",
      "Kokkos::HIP::concurrency(",
      "Kokkos::HIP::hip_stream(",
      "Kokkos::HIP::hip_device(",
      "Kokkos::HIP::hip_device_prop(",
      "Kokkos::HIP::impl_instance_id(",
      "Kokkos::HIP::impl_internal_space_instance(",
      "Kokkos::HIP::print_configuration(",
      "Kokkos::HostSpace::name(",
      "Kokkos::CudaSpace::name(",
      "Kokkos::CudaUVMSpace::name(",
      "Kokkos::CudaHostPinnedSpace::name(",
      "Kokkos::HIPSpace::name(",
      "Kokkos::HIPHostPinnedSpace::name(",
      "Kokkos::HIPManagedSpace::name(",
      // Execution policies are pure iteration-space bookkeeping (bounds,
      // tiles, chunk sizes, execution-space handles); nothing differentiable
      // flows through their constructors or accessors.
      "Kokkos::RangePolicy<",
      "Kokkos::Impl::ImplRangePolicy<",
      "Kokkos::MDRangePolicy<",
      "Kokkos::TeamPolicy<",
      "Kokkos::Impl::TeamPolicyInternal<",
      "Kokkos::Impl::PolicyTraits<",
      "Kokkos::ChunkSize::",
      "Kokkos::Impl::TileSizeProperties",
      "Kokkos::Impl::get_tile_size_properties",
      "Kokkos::Impl::to_array_potentially_narrowing<",
      "Kokkos::Impl::MDRangeTile",
      "Kokkos::PerTeam(",
      "Kokkos::PerThread(",
      "Kokkos::Impl::PerTeamValue",
      "Kokkos::Impl::PerThreadValue",
  };

  for (const char *P : InactivePrefixes) {
    if (startsWith(K, P))
      return true;
  }

  if (K.contains("::get_label(") || K.contains("::label(") ||
      K.contains("::use_count(") ||
      (K.contains("SharedAllocationRecord") &&
       K.contains("::print_records(")) ||
      (startsWith(K, "Kokkos::DualView<") && K.contains("::impl_report_")) ||
      (startsWith(K, "Kokkos::Impl::ImplRangePolicy<") &&
       K.contains("::check_")) ||
      (startsWith(K, "Kokkos::MDRangePolicy<") && K.contains("::check_")) ||
      (startsWith(K, "Kokkos::Impl::TeamPolicyInternal<") &&
       K.contains("::check_")))
    return true;

  return false;
}

/// Operand index and runtime metadata for a Kokkos / CUDA / HIP deep-copy call.
struct KokkosDeepCopyInfo {
  unsigned DstIdx = 0;
  unsigned SrcIdx = 0;
  unsigned LenIdx = 0;
  bool IsDeviceRuntimeCopy = false;
};

/// Identifies Kokkos / CUDA / HIP deep-copy functions and extracts the
/// (dst, src, byteLength) operand indices from the CallBase.
inline std::optional<KokkosDeepCopyInfo>
getKokkosDeepCopyInfo(const llvm::CallBase &CB, llvm::StringRef Name,
                      llvm::StringRef NormDem) {
  if (Name == "cudaMemcpy" || Name == "cudaMemcpyAsync" ||
      Name == "hipMemcpy" || Name == "hipMemcpyAsync") {
    if (CB.arg_size() >= 3 && CB.getArgOperand(0)->getType()->isPointerTy() &&
        CB.getArgOperand(1)->getType()->isPointerTy() &&
        CB.getArgOperand(2)->getType()->isIntegerTy()) {
      return KokkosDeepCopyInfo{0, 1, 2, true};
    }
  }
  if (NormDem.empty())
    return std::nullopt;
  llvm::StringRef K = stripReturnTypeToNamespace(NormDem, "Kokkos::");
  if (startsWith(K, "Kokkos::Impl::hostspace_parallel_deepcopy(") ||
      startsWith(K, "Kokkos::Impl::hostspace_parallel_deepcopy_async") ||
      startsWith(K, "Kokkos::Impl::DeepCopyCuda(") ||
      startsWith(K, "Kokkos::Impl::DeepCopyAsyncCuda(") ||
      startsWith(K, "Kokkos::Impl::DeepCopyHIP(") ||
      startsWith(K, "Kokkos::Impl::DeepCopyAsyncHIP(")) {
    // Either (dst, src, n[, ...]) or (exec, dst, src, n)
    if (CB.arg_size() >= 3 && CB.getArgOperand(0)->getType()->isPointerTy() &&
        CB.getArgOperand(1)->getType()->isPointerTy() &&
        CB.getArgOperand(2)->getType()->isIntegerTy()) {
      return KokkosDeepCopyInfo{0, 1, 2, false};
    }
    if (CB.arg_size() >= 4 && CB.getArgOperand(1)->getType()->isPointerTy() &&
        CB.getArgOperand(2)->getType()->isPointerTy() &&
        CB.getArgOperand(3)->getType()->isIntegerTy()) {
      return KokkosDeepCopyInfo{1, 2, 3, false};
    }
  }
  return std::nullopt;
}

/// Operand index and runtime metadata for a Kokkos / CUDA / HIP zero-memset
/// call.
struct KokkosZeroMemsetInfo {
  unsigned DstIdx = 0;
  unsigned LenIdx = 0;
  int ValIdx = -1;
  bool IsDeviceRuntimeMemset = false;
};

/// Identifies Kokkos / CUDA / HIP zero-memset functions and extracts the
/// (dst, byteLength) operand indices (and optional ValIdx for cudaMemset).
inline std::optional<KokkosZeroMemsetInfo>
getKokkosZeroMemsetInfo(const llvm::CallBase &CB, llvm::StringRef Name,
                        llvm::StringRef NormDem) {
  if (Name == "cudaMemset" || Name == "cudaMemsetAsync" ||
      Name == "hipMemset" || Name == "hipMemsetAsync") {
    if (CB.arg_size() >= 3 && CB.getArgOperand(0)->getType()->isPointerTy() &&
        CB.getArgOperand(1)->getType()->isIntegerTy() &&
        CB.getArgOperand(2)->getType()->isIntegerTy()) {
      return KokkosZeroMemsetInfo{0, 2, 1, true};
    }
  }
  if (NormDem.empty())
    return std::nullopt;
  llvm::StringRef K = stripReturnTypeToNamespace(NormDem, "Kokkos::");
  if (startsWith(K, "Kokkos::Impl::hostspace_parallel_zeromemset") ||
      startsWith(K, "Kokkos::Impl::ZeroMemset<")) {
    if (CB.arg_size() >= 3 && CB.getArgOperand(1)->getType()->isPointerTy() &&
        CB.getArgOperand(2)->getType()->isIntegerTy()) {
      return KokkosZeroMemsetInfo{1, 2, -1, false};
    }
    if (CB.arg_size() >= 2 && CB.getArgOperand(0)->getType()->isPointerTy() &&
        CB.getArgOperand(1)->getType()->isIntegerTy()) {
      return KokkosZeroMemsetInfo{0, 1, -1, false};
    }
  }
  return std::nullopt;
}

} // namespace enzyme_kokkos

#endif // ENZYME_KOKKOS_UTILS_H
