//===- ActivityAnalysis.cpp - Implementation of Activity Analysis ---------===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// If using this code in an academic setting, please cite the following:
// @incollection{enzymeNeurips,
// title = {Instead of Rewriting Foreign Code for Machine Learning,
//          Automatically Synthesize Fast Gradients},
// author = {Moses, William S. and Churavy, Valentin},
// booktitle = {Advances in Neural Information Processing Systems 33},
// year = {2020},
// note = {To appear in},
// }
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of Activity Analysis -- an AD-specific
// analysis that deduces if a given instruction or value can impact the
// calculation of a derivative. This file consists of two mutually recurive
// functions that compute this for values and instructions, respectively.
//
//===----------------------------------------------------------------------===//
#include <cstdint>
#include <deque>

#include <llvm/Config/llvm-config.h>

#include "llvm/ADT/ImmutableSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

#include "llvm/ADT/STLExtras.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

#include "llvm/IR/InstIterator.h"

#include "llvm/Support/raw_ostream.h"

#include "llvm/IR/InlineAsm.h"

#include "ActivityAnalysis.h"
#include "Utils.h"

#include "llvm/Demangle/Demangle.h"

#include "FunctionUtils.h"
#include "LibraryFuncs.h"
#include "TypeAnalysis/TBAA.h"

#include "llvm/Analysis/ValueTracking.h"

using namespace llvm;

#if LLVM_VERSION_MAJOR >= 14
#define addAttribute addAttributeAtIndex
#define removeAttribute removeAttributeAtIndex
#define getAttribute getAttributeAtIndex
#define hasAttribute hasAttributeAtIndex
#endif

extern "C" {
cl::opt<bool>
    EnzymePrintActivity("enzyme-print-activity", cl::init(false), cl::Hidden,
                        cl::desc("Print activity analysis algorithm"));

cl::opt<bool> EnzymeNonmarkedGlobalsInactive(
    "enzyme-globals-default-inactive", cl::init(false), cl::Hidden,
    cl::desc("Consider all nonmarked globals to be inactive"));

cl::opt<bool>
    EnzymeEmptyFnInactive("enzyme-emptyfn-inactive", cl::init(false),
                          cl::Hidden,
                          cl::desc("Empty functions are considered inactive"));

cl::opt<bool>
    EnzymeGlobalActivity("enzyme-global-activity", cl::init(false), cl::Hidden,
                         cl::desc("Enable correct global activity analysis"));

cl::opt<bool>
    EnzymeDisableActivityAnalysis("enzyme-disable-activity-analysis",
                                  cl::init(false), cl::Hidden,
                                  cl::desc("Disable activity analysis"));
}

#include "llvm/IR/InstIterator.h"
#include <map>
#include <set>
#include <unordered_map>

// clang-format off
const char *KnownInactiveFunctionsStartingWith[] = {
    "f90io",
    "$ss5print",
    "_ZTv0_n24_NSoD", //"1Ev, 0Ev
    "_ZNSt16allocator_traitsISaIdEE10deallocate",
    "_ZNSaIcED1Ev",
    "_ZNSaIcEC1Ev",
};

const char *KnownInactiveFunctionsContains[] = {
    "__enzyme_float", "__enzyme_double", "__enzyme_integer",
    "__enzyme_pointer"};

const StringSet<> InactiveGlobals = {
    "small_typeof",
    "ompi_request_null",
    "ompi_mpi_double",
    "ompi_mpi_comm_world",
    "stderr",
    "stdout",
    "stdin",
    "_ZSt3cin",
    "_ZSt4cout",
    "_ZNSt3__14coutE",
    "_ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEE6sentryC1ERS3_",
    "_ZSt5wcout",
    "_ZSt4cerr",
    "_ZTVNSt7__cxx1115basic_stringbufIcSt11char_traitsIcESaIcEEE",
    "_ZTVSt15basic_streambufIcSt11char_traitsIcEE",
    "_ZTVSt9basic_iosIcSt11char_traitsIcEE",
    // istream
    "_ZTVNSt7__cxx1119basic_istringstreamIcSt11char_traitsIcESaIcEEE",
    "_ZTTNSt7__cxx1119basic_istringstreamIcSt11char_traitsIcESaIcEEE",
    // ostream
    "_ZTVNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEE",
    "_ZTTNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEE",
    // stringstream
    "_ZTVNSt7__cxx1118basic_stringstreamIcSt11char_traitsIcESaIcEEE",
    "_ZTTNSt7__cxx1118basic_stringstreamIcSt11char_traitsIcESaIcEEE",
    // ifstream
    "_ZTTSt14basic_ifstreamIcSt11char_traitsIcEE",
    // ofstream
    "_ZTTSt14basic_ofstreamIcSt11char_traitsIcEE",
    // vtable for __cxxabiv1::__si_class_type_info
    "_ZTVN10__cxxabiv120__si_class_type_infoE",
    "_ZTVN10__cxxabiv117__class_type_infoE",
    "_ZTVN10__cxxabiv121__vmi_class_type_infoE"
};

const llvm::StringMap<size_t> MPIInactiveCommAllocators = {
    {"MPI_Graph_create", 5},
    {"MPI_Comm_split", 2},
    {"MPI_Intercomm_create", 6},
    {"MPI_Comm_spawn", 6},
    {"MPI_Comm_spawn_multiple", 7},
    {"MPI_Comm_accept", 4},
    {"MPI_Comm_connect", 4},
    {"MPI_Comm_create", 2},
    {"MPI_Comm_create_group", 3},
    {"MPI_Comm_dup", 1},
    {"MPI_Comm_dup", 2},
    {"MPI_Comm_idup", 1},
    {"MPI_Comm_join", 1},
};

// Instructions which themselves are inactive
// the returned value, however, may still be active
const StringSet<> KnownInactiveFunctionInsts = {
    "__dynamic_cast",
    "_ZSt18_Rb_tree_decrementPKSt18_Rb_tree_node_base",
    "_ZSt18_Rb_tree_incrementPKSt18_Rb_tree_node_base",
    "_ZSt18_Rb_tree_decrementPSt18_Rb_tree_node_base",
    "_ZSt18_Rb_tree_incrementPSt18_Rb_tree_node_base",
    "jl_ptr_to_array",
    "jl_ptr_to_array_1d"};

const StringSet<> KnownInactiveFunctions = {
    "__nv_isfinitel",
    "__nv_isfinited",
    "cublasCreate_v2",
    "cublasSetMathMode",
    "cublasSetStream_v2",
    "cuMemPoolTrimTo",
    "cuDeviceGetMemPool",
    "cuStreamCreate",
    "cuStreamSynchronize",
    "cuStreamDestroy",
    "cuStreamQuery",
    "cuCtxGetCurrent",
    "enzyme_zerotype",
    "abort",
    "time",
    "memcmp",
    "memchr",
    "gettimeofday",
    "stat",
    "mkdir",
    "compress2",
    "__assert_fail",
    "__cxa_atexit",
    "__cxa_guard_acquire",
    "__cxa_guard_release",
    "__cxa_guard_abort",
    "snprintf",
    "sprintf",
    "printf",
    "fprintf",
    "putchar",
    "fprintf",
    "vprintf",
    "vsnprintf",
    "puts",
    "fflush",
    "__kmpc_for_static_init_4",
    "__kmpc_for_static_init_4u",
    "__kmpc_for_static_init_8",
    "__kmpc_for_static_init_8u",
    "__kmpc_for_static_fini",
    "__kmpc_dispatch_init_4",
    "__kmpc_dispatch_init_4u",
    "__kmpc_dispatch_init_8",
    "__kmpc_dispatch_init_8u",
    "__kmpc_dispatch_next_4",
    "__kmpc_dispatch_next_4u",
    "__kmpc_dispatch_next_8",
    "__kmpc_dispatch_next_8u",
    "__kmpc_dispatch_fini_4",
    "__kmpc_dispatch_fini_4u",
    "__kmpc_dispatch_fini_8",
    "__kmpc_dispatch_fini_8u",
    "__kmpc_barrier",
    "__kmpc_barrier_master",
    "__kmpc_barrier_master_nowait",
    "__kmpc_barrier_end_barrier_master",
    "__kmpc_global_thread_num",
    "omp_get_max_threads",
    "malloc_usable_size",
    "malloc_size",
    "MPI_Init",
    "MPI_Comm_size",
    "PMPI_Comm_size",
    "MPI_Comm_rank",
    "PMPI_Comm_rank",
    "MPI_Get_processor_name",
    "MPI_Finalize",
    "MPI_Test",
    "MPI_Probe", // double check potential syncronization
    "MPI_Barrier",
    "MPI_Abort",
    "MPI_Get_count",
    "MPI_Comm_free",
    "MPI_Comm_get_parent",
    "MPI_Comm_get_name",
    "MPI_Comm_get_info",
    "MPI_Comm_remote_size",
    "MPI_Comm_set_info",
    "MPI_Comm_set_name",
    "MPI_Comm_compare",
    "MPI_Comm_call_errhandler",
    "MPI_Comm_create_errhandler",
    "MPI_Comm_disconnect",
    "MPI_Wtime",
    "_msize",
    "ftnio_fmt_write64",
    "f90_strcmp_klen",
    "__swift_instantiateConcreteTypeFromMangledName",
    "logb",
    "logbf",
    "logbl",
    "cuCtxGetCurrent",
    "cuDeviceGet",
    "cuDeviceGetName",
    "cuDriverGetVersion",
    "cudaRuntimeGetVersion",
    "cuDeviceGetCount",
    "cuMemPoolGetAttribute",
    "cuMemGetInfo_v2",
    "cuDeviceGetAttribute",
    "cuDevicePrimaryCtxRetain"
};

const std::set<Intrinsic::ID> KnownInactiveIntrinsics = {
#if LLVM_VERSION_MAJOR >= 12
    Intrinsic::experimental_noalias_scope_decl,
#endif
    Intrinsic::floor,
    Intrinsic::ceil,
    Intrinsic::trunc,
    Intrinsic::rint,
    Intrinsic::lrint,
    Intrinsic::llrint,
    Intrinsic::nearbyint,
    Intrinsic::round,
    Intrinsic::roundeven,
    Intrinsic::lround,
    Intrinsic::llround,
    Intrinsic::nvvm_barrier0,
    Intrinsic::nvvm_barrier0_popc,
    Intrinsic::nvvm_barrier0_and,
    Intrinsic::nvvm_barrier0_or,
    Intrinsic::nvvm_membar_cta,
    Intrinsic::nvvm_membar_gl,
    Intrinsic::nvvm_membar_sys,
    Intrinsic::amdgcn_s_barrier,
    Intrinsic::assume,
    Intrinsic::stacksave,
    Intrinsic::stackrestore,
    Intrinsic::lifetime_start,
    Intrinsic::lifetime_end,
#if LLVM_VERSION_MAJOR <= 16
    Intrinsic::dbg_addr,
#endif

    Intrinsic::dbg_declare,
    Intrinsic::dbg_value,
    Intrinsic::dbg_label,
    Intrinsic::invariant_start,
    Intrinsic::invariant_end,
    Intrinsic::var_annotation,
    Intrinsic::ptr_annotation,
    Intrinsic::annotation,
    Intrinsic::codeview_annotation,
    Intrinsic::expect,
    Intrinsic::type_test,
    Intrinsic::donothing,
    Intrinsic::prefetch,
    Intrinsic::trap,
    Intrinsic::is_constant,
    Intrinsic::memset};

const char *DemangledKnownInactiveFunctionsStartingWith[] = {
    // TODO this returns allocated memory and thus can be an active value
    // "std::allocator",
    "std::chrono::_V2::steady_clock::now",
    "std::string",
    "std::cerr",
    "std::istream",
    "std::ostream",
    "std::ios_base",
    "std::locale",
    "std::ctype<char>",
    "std::__basic_file",
    "std::__ioinit",
    "std::__basic_file",
    "std::hash",
    "std::_Hash_bytes",

    // __cxx11
    "std::__cxx11::basic_string",
    "std::__cxx11::basic_ios",
    "std::__cxx11::basic_ostringstream",
    "std::__cxx11::basic_istringstream",
    "std::__cxx11::basic_istream",
    "std::__cxx11::basic_ostream",
    "std::__cxx11::basic_ifstream",
    "std::__cxx11::basic_ofstream",
    "std::__cxx11::basic_stringbuf",
    "std::__cxx11::basic_filebuf",
    "std::__cxx11::basic_streambuf",

    // non __cxx11
    "std::basic_string",
    "std::to_string",
    "std::basic_ios",
    "std::basic_ostringstream",
    "std::basic_istringstream",
    "std::basic_istream",
    "std::basic_ostream",
    "std::basic_ifstream",
    "std::basic_ofstream",
    "std::basic_stringbuf",
    "std::basic_filebuf",
    "std::basic_streambuf",
    "std::random_device",
    "std::mersenne_twister_engine",
    "std::linear_congruential_engine",
    "std::subtract_with_carry_engine",
    "std::discard_block_engine",
    "std::independent_bits_engine",
    "std::shuffle_order_engine",
  
  
    // libc++
    "std::__1::basic_string",
    "std::__1::__do_string_hash",
    "std::__1::hash",
    "std::__1::__unordered_map_hasher",
    "std::__1::to_string",
    "std::__1::basic_ostream",
    "std::__1::cout",
    "std::__1::random_device",
    "std::__1::mersenne_twister_engine",
    "std::__1::linear_congruential_engine",
    "std::__1::subtract_with_carry_engine",
    "std::__1::discard_block_engine",
    "std::__1::independent_bits_engine",
    "std::__1::shuffle_order_engine",
  

    "std::__detail::_Prime_rehash_policy",
    "std::__detail::_Hash_code_base",
};
// clang-format on

/// Is the use of value val as an argument of call CI known to be inactive
/// This tool can only be used when in DOWN mode
bool ActivityAnalyzer::isFunctionArgumentConstant(CallInst *CI, Value *val) {
  assert(directions & DOWN);
  if (CI->hasFnAttr("enzyme_inactive"))
    return true;

  auto F = getFunctionFromCall(CI);

  bool all_inactive = val != CI->getCalledOperand();

#if LLVM_VERSION_MAJOR >= 14
  for (size_t i = 0; i < CI->arg_size(); i++)
#else
  for (size_t i = 0; i < CI->getNumArgOperands(); i++)
#endif
  {
    if (val == CI->getArgOperand(i)) {
      if (!CI->getAttributes().hasParamAttr(i, "enzyme_inactive") &&
          !(F && F->getCallingConv() == CI->getCallingConv() &&
            F->getAttributes().hasParamAttr(i, "enzyme_inactive"))) {
        all_inactive = false;
        break;
      }
    }
  }

  if (all_inactive)
    return true;

  // Indirect function calls may actively use the argument
  if (F == nullptr)
    return false;

  if (F->hasFnAttribute("enzyme_inactive")) {
    return true;
  }

  auto Name = getFuncNameFromCall(CI);

  // Only the 1-th arg impacts activity
  if (Name == "jl_reshape_array" || Name == "ijl_reshape_array")
    return val != CI->getArgOperand(1);

  // Allocations, deallocations, and c++ guards don't impact the activity
  // of arguments
  if (isAllocationFunction(Name, TLI) || isDeallocationFunction(Name, TLI))
    return true;

  std::string demangledName = llvm::demangle(Name.str());
  auto dName = StringRef(demangledName);
  for (auto FuncName : DemangledKnownInactiveFunctionsStartingWith) {
    if (dName.startswith(FuncName)) {
      return true;
    }
  }
  if (demangledName == Name.str()) {
    // Either demangeling failed
    // or they are equal but matching failed
    // if (!Name.startswith("llvm."))
    //  llvm::errs() << "matching failed: " << Name.str() << " "
    //               << demangledName << "\n";
  }

  for (auto FuncName : KnownInactiveFunctionsStartingWith) {
    if (Name.startswith(FuncName)) {
      return true;
    }
  }

  for (auto FuncName : KnownInactiveFunctionsContains) {
    if (Name.contains(FuncName)) {
      return true;
    }
  }
  if (KnownInactiveFunctions.count(Name)) {
    return true;
  }

  if (MPIInactiveCommAllocators.find(Name) != MPIInactiveCommAllocators.end()) {
    return true;
  }
  if (KnownInactiveIntrinsics.count(F->getIntrinsicID())) {
    return true;
  }

  /// Only the first argument (magnitude) of copysign is active
  if (F->getIntrinsicID() == Intrinsic::copysign &&
      CI->getArgOperand(0) != val) {
    return true;
  }

  if (F->getIntrinsicID() == Intrinsic::memcpy && CI->getArgOperand(0) != val &&
      CI->getArgOperand(1) != val)
    return true;
  if (F->getIntrinsicID() == Intrinsic::memmove &&
      CI->getArgOperand(0) != val && CI->getArgOperand(1) != val)
    return true;

  // only the buffer is active for mpi send/recv
  if (Name == "MPI_Recv" || Name == "PMPI_Recv" || Name == "MPI_Send" ||
      Name == "PMPI_Send") {
    return val != CI->getOperand(0);
  }
  // only the recv buffer and request is active for mpi isend/irecv
  if (Name == "MPI_Irecv" || Name == "MPI_Isend") {
    return val != CI->getOperand(0) && val != CI->getOperand(6);
  }

  // only request is active
  if (Name == "MPI_Wait" || Name == "PMPI_Wait")
    return val != CI->getOperand(0);

  if (Name == "MPI_Waitall" || Name == "PMPI_Waitall")
    return val != CI->getOperand(1);

  // TODO interprocedural detection
  // Before potential introprocedural detection, any function without definition
  // may to be assumed to have an active use
  if (F->empty())
    return false;

  // With all other options exhausted we have to assume this function could
  // actively use the value
  return false;
}

/// Call the function propagateFromOperand on all operands of CI
/// that could impact the activity of the call instruction
static inline void propagateArgumentInformation(
    TargetLibraryInfo &TLI, CallInst &CI,
    llvm::function_ref<bool(Value *)> propagateFromOperand) {

  // These functions are known to only have the first argument impact
  // the activity of the call instruction
  auto Name = getFuncNameFromCall(&CI);
  if (Name == "lgamma" || Name == "lgammaf" || Name == "lgammal" ||
      Name == "lgamma_r" || Name == "lgammaf_r" || Name == "lgammal_r" ||
      Name == "__lgamma_r_finite" || Name == "__lgammaf_r_finite" ||
      Name == "__lgammal_r_finite") {

    propagateFromOperand(CI.getArgOperand(0));
    return;
  }

  if (Name == "julia.call" || Name == "julia.call2") {
#if LLVM_VERSION_MAJOR >= 14
    for (size_t i = 1; i < CI.arg_size(); i++)
#else
    for (size_t i = 1; i < CI.getNumArgOperands(); i++)
#endif
    {
      propagateFromOperand(CI.getOperand(i));
    }
    return;
  }

  // Only the 1-th arg impacts activity
  if (Name == "jl_reshape_array" || Name == "ijl_reshape_array") {
    propagateFromOperand(CI.getArgOperand(1));
    return;
  }

  // Allocations, deallocations, and c++ guards are fully inactive
  if (isAllocationFunction(Name, TLI) || isDeallocationFunction(Name, TLI) ||
      Name == "__cxa_guard_acquire" || Name == "__cxa_guard_release" ||
      Name == "__cxa_guard_abort")
    return;

  auto F = getFunctionFromCall(&CI);

  if (F) {

    /// Only the first argument (magnitude) of copysign is active
    if (F->getIntrinsicID() == Intrinsic::copysign) {
      propagateFromOperand(CI.getOperand(0));
      return;
    }

    // Certain intrinsics are inactive by definition
    // and have nothing to propagate.
    if (KnownInactiveIntrinsics.count(F->getIntrinsicID())) {
      return;
    }

    if (F->getIntrinsicID() == Intrinsic::memcpy ||
        F->getIntrinsicID() == Intrinsic::memmove) {
      propagateFromOperand(CI.getOperand(0));
      propagateFromOperand(CI.getOperand(1));
      return;
    }
  }

  // For other calls, check all operands of the instruction
  // as conservatively they may impact the activity of the call
  size_t i = 0;
#if LLVM_VERSION_MAJOR >= 14
  for (auto &a : CI.args())
#else
  for (auto &a : CI.arg_operands())
#endif
  {

    if (CI.getAttributes().hasParamAttr(i, "enzyme_inactive") ||
        (F && F->getCallingConv() == CI.getCallingConv() &&
         F->getAttributes().hasParamAttr(i, "enzyme_inactive"))) {
      i++;
      continue;
    }

    if (propagateFromOperand(a))
      break;
    i++;
  }
}

bool isPossibleFloat(const TypeResults &TR, Value *I, const DataLayout &DL) {
  bool possibleFloat = false;
  if (!I->getType()->isVoidTy()) {
    auto Size = (DL.getTypeSizeInBits(I->getType()) + 7) / 8;
    auto vd = TR.query(I);
    auto ct0 = vd[{-1}];
    if (ct0.isPossibleFloat() && ct0 != BaseType::Anything) {
      for (unsigned i = 0; i < Size;) {
        auto ct = vd[{(int)i}];
        if (ct.isPossibleFloat() && ct != BaseType::Anything) {
          possibleFloat = true;
          break;
        }
        size_t chunk = 1;
        // Implicit pointer
        if (ct == BaseType::Pointer)
          chunk = DL.getPointerSizeInBits() / 8;
        i += chunk;
      }
    }
  }
  return possibleFloat;
}

/// Return whether this instruction is known not to propagate adjoints
/// Note that instructions could return an active pointer, but
/// do not propagate adjoints themselves
bool ActivityAnalyzer::isConstantInstruction(TypeResults const &TR,
                                             Instruction *I) {
  // This analysis may only be called by instructions corresponding to
  // the function analyzed by TypeInfo
  assert(I);
  assert(TR.getFunction() == I->getParent()->getParent());

  // The return instruction doesn't impact activity (handled specifically
  // during adjoint generation)
  if (isa<ReturnInst>(I))
    return true;

  // Branch, unreachable, and previously computed constants are inactive
  if (isa<UnreachableInst>(I) || isa<BranchInst>(I) ||
      (ConstantInstructions.find(I) != ConstantInstructions.end())) {
    return true;
  }

  /// Previously computed inactives remain inactive
  if ((ActiveInstructions.find(I) != ActiveInstructions.end())) {
    return false;
  }

  if (notForAnalysis.count(I->getParent())) {
    if (EnzymePrintActivity)
      llvm::errs() << " constant instruction as dominates unreachable " << *I
                   << "\n";
    InsertConstantInstruction(TR, I);
    return true;
  }

  if (isa<FenceInst>(I)) {
    if (EnzymePrintActivity)
      llvm::errs() << " constant fence instruction " << *I << "\n";
    InsertConstantInstruction(TR, I);
    return true;
  }

  if (auto CI = dyn_cast<CallInst>(I)) {
    if (CI->hasFnAttr("enzyme_active") || CI->hasFnAttr("enzyme_active_inst")) {
      if (EnzymePrintActivity)
        llvm::errs() << "forced active " << *I << "\n";
      ActiveInstructions.insert(I);
      return false;
    }
    if (CI->hasFnAttr("enzyme_inactive") ||
        CI->hasFnAttr("enzyme_inactive_inst")) {
      if (EnzymePrintActivity)
        llvm::errs() << "forced inactive " << *I << "\n";
      InsertConstantInstruction(TR, I);
      return true;
    }
    auto called = getFunctionFromCall(CI);

    if (called) {
      if (called->hasFnAttribute("enzyme_active") ||
          called->hasFnAttribute("enzyme_active_inst")) {
        if (EnzymePrintActivity)
          llvm::errs() << "forced active " << *I << "\n";
        ActiveInstructions.insert(I);
        return false;
      }
      if (called->hasFnAttribute("enzyme_inactive") ||
          called->hasFnAttribute("enzyme_inactive_inst")) {
        if (EnzymePrintActivity)
          llvm::errs() << "forced inactive " << *I << "\n";
        InsertConstantInstruction(TR, I);
        return true;
      }
      if (KnownInactiveFunctionInsts.count(called->getName())) {
        InsertConstantInstruction(TR, I);
        return true;
      }
    }
  }

  if (auto II = dyn_cast<IntrinsicInst>(I)) {
    if (KnownInactiveIntrinsics.count(II->getIntrinsicID())) {
      if (EnzymePrintActivity)
        llvm::errs() << "known inactive intrinsic " << *I << "\n";
      InsertConstantInstruction(TR, I);
      return true;
    } else if (isIntelSubscriptIntrinsic(*II)) {
      // The intrinsic "llvm.intel.subscript" does not propogate deriviative
      // information directly. But its returned pointer may be active.
      InsertConstantInstruction(TR, I);
      return true;
    }
  }

  if (EnzymeDisableActivityAnalysis)
    return false;

  /// A store into all integral memory is inactive
  if (auto SI = dyn_cast<StoreInst>(I)) {
    auto StoreSize = SI->getParent()
                         ->getParent()
                         ->getParent()
                         ->getDataLayout()
                         .getTypeSizeInBits(SI->getValueOperand()->getType()) /
                     8;

    bool AllIntegral = true;
    bool SeenInteger = false;
    auto q = TR.query(SI->getPointerOperand()).Data0();
    for (int i = -1; i < (int)StoreSize; ++i) {
      auto dt = q[{i}];
      if (dt.isIntegral() || dt == BaseType::Anything) {
        SeenInteger = true;
        if (i == -1)
          break;
      } else if (dt.isKnown()) {
        AllIntegral = false;
        break;
      }
    }

    if (AllIntegral && SeenInteger) {
      if (EnzymePrintActivity)
        llvm::errs() << " constant instruction from TA " << *I << "\n";
      InsertConstantInstruction(TR, I);
      return true;
    }
  }
  if (auto SI = dyn_cast<AtomicRMWInst>(I)) {
    auto StoreSize = SI->getParent()
                         ->getParent()
                         ->getParent()
                         ->getDataLayout()
                         .getTypeSizeInBits(I->getType()) /
                     8;

    bool AllIntegral = true;
    bool SeenInteger = false;
    auto q = TR.query(SI->getOperand(0)).Data0();
    for (int i = -1; i < (int)StoreSize; ++i) {
      auto dt = q[{i}];
      if (dt.isIntegral() || dt == BaseType::Anything) {
        SeenInteger = true;
        if (i == -1)
          break;
      } else if (dt.isKnown()) {
        AllIntegral = false;
        break;
      }
    }

    if (AllIntegral && SeenInteger) {
      if (EnzymePrintActivity)
        llvm::errs() << " constant instruction from TA " << *I << "\n";
      InsertConstantInstruction(TR, I);
      return true;
    }
  }

  if (EnzymePrintActivity)
    llvm::errs() << "checking if is constant[" << (int)directions << "] " << *I
                 << "\n";

  // Analyzer for inductive assumption where we attempt to prove this is
  // inactive from a lack of active users
  std::shared_ptr<ActivityAnalyzer> DownHypothesis;

  // If this instruction does not write to memory that outlives itself
  // (potentially propagating derivative information), the only way to propagate
  // derivative information is through the return value
  // TODO the "doesn't write to active memory" can be made more aggressive than
  // doesn't write to any memory
  bool noActiveWrite = false;
  if (!I->mayWriteToMemory())
    noActiveWrite = true;
  else if (auto CI = dyn_cast<CallInst>(I)) {
    if (AA.onlyReadsMemory(CI) || isReadOnly(CI)) {
      noActiveWrite = true;
    } else {
      StringRef funcName = getFuncNameFromCall(CI);
      if (isMemFreeLibMFunction(funcName)) {
        noActiveWrite = true;
      } else if (funcName == "frexp" || funcName == "frexpf" ||
                 funcName == "frexpl") {
        noActiveWrite = true;
      }
    }
  }
  if (noActiveWrite) {
    auto &DL = I->getParent()->getParent()->getParent()->getDataLayout();
    bool possibleFloat = isPossibleFloat(TR, I, DL);
    // Even if returning a pointer, this instruction is considered inactive
    // since the instruction doesn't prop gradients. Thus, so long as we don't
    // return an object containing a float, this instruction is inactive
    if (!possibleFloat) {
      if (EnzymePrintActivity)
        llvm::errs()
            << " constant instruction from known non-float non-writing "
               "instruction "
            << *I << "\n";
      InsertConstantInstruction(TR, I);
      return true;
    }

    // If the value returned is constant otherwise, the instruction is inactive
    if (isConstantValue(TR, I)) {
      if (EnzymePrintActivity)
        llvm::errs() << " constant instruction from known constant non-writing "
                        "instruction "
                     << *I << "\n";
      InsertConstantInstruction(TR, I);
      return true;
    }

    // Even if the return is nonconstant, it's worth checking explicitly the
    // users since unlike isConstantValue, returning a pointer does not make the
    // instruction active
    if (directions & DOWN) {
      // We shall now induct on this instruction being inactive and try to prove
      // this fact from a lack of active users.

      // If we aren't a phi node (and thus potentially recursive on uses) and
      // already equal to the current direction, we don't need to induct,
      // reducing runtime.
      if (directions == DOWN && !isa<PHINode>(I)) {
        if (isValueInactiveFromUsers(TR, I, UseActivity::None)) {
          if (EnzymePrintActivity)
            llvm::errs() << " constant instruction[" << (int)directions
                         << "] from users instruction " << *I << "\n";
          InsertConstantInstruction(TR, I);
          return true;
        }
      } else {
        DownHypothesis = std::shared_ptr<ActivityAnalyzer>(
            new ActivityAnalyzer(*this, DOWN));
        DownHypothesis->ConstantInstructions.insert(I);
        if (DownHypothesis->isValueInactiveFromUsers(TR, I,
                                                     UseActivity::None)) {
          if (EnzymePrintActivity)
            llvm::errs() << " constant instruction[" << (int)directions
                         << "] from users instruction " << *I << "\n";
          InsertConstantInstruction(TR, I);
          insertConstantsFrom(TR, *DownHypothesis);
          return true;
        }
      }
    }
  }

  std::shared_ptr<ActivityAnalyzer> UpHypothesis;
  if (directions & UP) {
    // If this instruction has no active operands, the instruction
    // is active.
    // TODO This isn't 100% accurate and will incorrectly mark a no-argument
    // function that reads from active memory as constant
    // Technically the additional constraint is that this does not read from
    // active memory, where we have assumed that the only active memory
    // we care about is accessible from arguments passed (and thus not globals)
    UpHypothesis =
        std::shared_ptr<ActivityAnalyzer>(new ActivityAnalyzer(*this, UP));
    UpHypothesis->ConstantInstructions.insert(I);
    assert(directions & UP);
    if (UpHypothesis->isInstructionInactiveFromOrigin(TR, I, false)) {
      if (EnzymePrintActivity)
        llvm::errs() << " constant instruction from origin "
                        "instruction "
                     << *I << "\n";
      InsertConstantInstruction(TR, I);
      insertConstantsFrom(TR, *UpHypothesis);
      if (DownHypothesis)
        insertConstantsFrom(TR, *DownHypothesis);
      return true;
    } else if (directions == 3) {
      if (isa<LoadInst>(I) || isa<StoreInst>(I) || isa<BinaryOperator>(I)) {
        for (auto &op : I->operands()) {
          if (!UpHypothesis->isConstantValue(TR, op)) {
            ReEvaluateInstIfInactiveValue[op].insert(I);
          }
        }
      }
    }
  }

  // Otherwise we must fall back and assume this instruction to be active.
  ActiveInstructions.insert(I);
  if (EnzymePrintActivity)
    llvm::errs() << "couldnt decide fallback as nonconstant instruction("
                 << (int)directions << "):" << *I << "\n";
  if (noActiveWrite && directions == 3)
    ReEvaluateInstIfInactiveValue[I].insert(I);
  return false;
}

bool isValuePotentiallyUsedAsPointer(llvm::Value *val) {
  std::deque<llvm::Value *> todo = {val};
  SmallPtrSet<Value *, 3> seen;
  while (todo.size()) {
    auto cur = todo.back();
    todo.pop_back();
    if (seen.count(cur))
      continue;
    seen.insert(cur);
    for (auto u : cur->users()) {
      if (isa<ReturnInst>(u))
        return true;
      if (!cast<Instruction>(u)->mayReadOrWriteMemory()) {
        todo.push_back(u);
        continue;
      }
      if (EnzymePrintActivity)
        llvm::errs() << " VALUE potentially used as pointer " << *val << " by "
                     << *u << "\n";
      return true;
    }
  }
  return false;
}

bool ActivityAnalyzer::isConstantValue(TypeResults const &TR, Value *Val) {
  // This analysis may only be called by instructions corresponding to
  // the function analyzed by TypeInfo -- however if the Value
  // was created outside a function (e.g. global, constant), that is allowed
  assert(Val);
  if (auto I = dyn_cast<Instruction>(Val)) {
    if (TR.getFunction() != I->getParent()->getParent()) {
      llvm::errs() << *TR.getFunction() << "\n";
      llvm::errs() << *I << "\n";
    }
    assert(TR.getFunction() == I->getParent()->getParent());
  }
  if (auto Arg = dyn_cast<Argument>(Val)) {
    assert(TR.getFunction() == Arg->getParent());
  }

  // Void values are definitionally inactive
  if (Val->getType()->isVoidTy())
    return true;

  // Token values are definitionally inactive
  if (Val->getType()->isTokenTy())
    return true;

  // All function pointers are considered active in case an augmented primal
  // or reverse is needed
  if (isa<Function>(Val) || isa<InlineAsm>(Val)) {
    return false;
  }

  /// If we've already shown this value to be inactive
  if (ConstantValues.find(Val) != ConstantValues.end()) {
    return true;
  }

  /// If we've already shown this value to be active
  if (ActiveValues.find(Val) != ActiveValues.end()) {
    return false;
  }

  // We do this check down here so we can go past asserted constant values from
  // arguments, and also allow void/tokens to be inactive.
  if (!EnzymeDisableActivityAnalysis) {

    if (auto CD = dyn_cast<ConstantDataSequential>(Val)) {
      // inductively assume inactive
      ConstantValues.insert(CD);
      for (size_t i = 0, len = CD->getNumElements(); i < len; i++) {
        if (!isConstantValue(TR, CD->getElementAsConstant(i))) {
          ConstantValues.erase(CD);
          ActiveValues.insert(CD);
          return false;
        }
      }
      return true;
    }
    if (auto CD = dyn_cast<ConstantAggregate>(Val)) {
      // inductively assume inactive
      ConstantValues.insert(CD);
      for (size_t i = 0, len = CD->getNumOperands(); i < len; i++) {
        if (!isConstantValue(TR, CD->getOperand(i))) {
          ConstantValues.erase(CD);
          ActiveValues.insert(CD);
          return false;
        }
      }
      return true;
    }

    // Undef, metadata, non-global constants, and blocks are inactive
    if (isa<UndefValue>(Val) || isa<MetadataAsValue>(Val) ||
        isa<ConstantData>(Val) || isa<ConstantAggregate>(Val) ||
        isa<BasicBlock>(Val)) {
      return true;
    }

    if (auto II = dyn_cast<IntrinsicInst>(Val)) {
      if (KnownInactiveIntrinsics.count(II->getIntrinsicID())) {
        InsertConstantValue(TR, Val);
        return true;
      }
    }

    // All arguments must be marked constant/nonconstant ahead of time
    if (isa<Argument>(Val) && !cast<Argument>(Val)->hasByValAttr()) {
      llvm::errs() << *(cast<Argument>(Val)->getParent()) << "\n";
      llvm::errs() << *Val << "\n";
      assert(0 && "must've put arguments in constant/nonconstant");
    }

    // This value is certainly an integer (and only and integer, not a pointer
    // or float). Therefore its value is constant
    if (TR.query(Val)[{-1}] == BaseType::Integer) {
      if (EnzymePrintActivity)
        llvm::errs() << " Value const as integral " << (int)directions << " "
                     << *Val << " "
                     << TR.intType(1, Val, /*errIfNotFound*/ false).str()
                     << "\n";
      InsertConstantValue(TR, Val);
      return true;
    }

#if 0
  // This value is certainly a pointer to an integer (and only and integer, not
  // a pointer or float). Therefore its value is constant
  // TODO use typeInfo for more aggressive activity analysis
  if (val->getType()->isPointerTy() &&
      cast<PointerType>(val->getType())->isIntOrIntVectorTy() &&
      TR.firstPointer(1, val, /*errifnotfound*/ false).isIntegral()) {
    if (EnzymePrintActivity)
      llvm::errs() << " Value const as integral pointer" << (int)directions
                   << " " << *val << "\n";
    InsertConstantValue(TR, Val);
    return true;
  }
#endif

    if (auto GA = dyn_cast<GlobalAlias>(Val))
      return isConstantValue(TR, GA->getAliasee());

    if (auto GI = dyn_cast<GlobalVariable>(Val)) {
      // If operating under the assumption globals are inactive unless
      // explicitly marked as active, this is inactive
      if (!hasMetadata(GI, "enzyme_shadow") && EnzymeNonmarkedGlobalsInactive) {
        InsertConstantValue(TR, Val);
        return true;
      }
      if (hasMetadata(GI, "enzyme_inactive")) {
        InsertConstantValue(TR, Val);
        return true;
      }

      if (GI->getName().contains("enzyme_const") ||
          InactiveGlobals.count(GI->getName())) {
        InsertConstantValue(TR, Val);
        return true;
      }

      // If this global is unchanging and the internal constant data
      // is inactive, the global is inactive
      if (GI->isConstant() && GI->hasInitializer() &&
          isConstantValue(TR, GI->getInitializer())) {
        InsertConstantValue(TR, Val);
        if (EnzymePrintActivity)
          llvm::errs() << " VALUE const global " << *Val
                       << " init: " << *GI->getInitializer() << "\n";
        return true;
      }

      // If this global is a pointer to an integer, it is inactive
      // TODO note this may need updating to consider the size
      // of the global
      auto res = TR.query(GI).Data0();
      auto dt = res[{-1}];
      if (dt.isIntegral()) {
        if (EnzymePrintActivity)
          llvm::errs() << " VALUE const as global int pointer " << *Val
                       << " type - " << res.str() << "\n";
        InsertConstantValue(TR, Val);
        return true;
      }

      // If this is a global local to this translation unit with inactive
      // initializer and no active uses, it is definitionally inactive
      bool usedJustInThisModule =
          GI->hasInternalLinkage() || GI->hasPrivateLinkage();

      if (EnzymePrintActivity)
        llvm::errs() << "pre attempting(" << (int)directions
                     << ") just used in module for: " << *GI << " dir"
                     << (int)directions
                     << " justusedin:" << usedJustInThisModule << "\n";

      if (directions == 3 && usedJustInThisModule) {
        // TODO this assumes global initializer cannot refer to itself (lest
        // infinite loop)
        if (!GI->hasInitializer() ||
            isConstantValue(TR, GI->getInitializer())) {

          if (EnzymePrintActivity)
            llvm::errs() << "attempting just used in module for: " << *GI
                         << "\n";
          // Not looking at users to prove inactive (definition of down)
          // If all users are inactive, this is therefore inactive.
          // Since we won't look at origins to prove, we can inductively assume
          // this is inactive

          // As an optimization if we are going down already
          // and we won't use ourselves (done by PHI's), we
          // dont need to inductively assume we're true
          // and can instead use this object!
          // This pointer is inactive if it is either not actively stored to or
          // not actively loaded from
          // See alloca logic to explain why OnlyStores is insufficient here
          if (directions == DOWN) {
            if (isValueInactiveFromUsers(TR, Val, UseActivity::OnlyLoads)) {
              InsertConstantValue(TR, Val);
              return true;
            }
          } else {
            Instruction *LoadReval = nullptr;
            Instruction *StoreReval = nullptr;
            auto DownHypothesis = std::shared_ptr<ActivityAnalyzer>(
                new ActivityAnalyzer(*this, DOWN));
            DownHypothesis->ConstantValues.insert(Val);
            if (DownHypothesis->isValueInactiveFromUsers(
                    TR, Val, UseActivity::OnlyLoads, &LoadReval) ||
                (TR.query(GI)[{-1, -1}].isFloat() &&
                 DownHypothesis->isValueInactiveFromUsers(
                     TR, Val, UseActivity::OnlyStores, &StoreReval))) {
              insertConstantsFrom(TR, *DownHypothesis);
              InsertConstantValue(TR, Val);
              return true;
            } else {
              if (LoadReval) {
                if (EnzymePrintActivity)
                  llvm::errs() << " global activity of " << *Val
                               << " dependant on " << *LoadReval << "\n";
                ReEvaluateValueIfInactiveInst[LoadReval].insert(Val);
              }
              if (StoreReval)
                ReEvaluateValueIfInactiveInst[StoreReval].insert(Val);
            }
          }
        }
      }

      // Otherwise we have to assume this global is active since it can
      // be arbitrarily used in an active way
      // TODO we can be more aggressive here in the future
      if (EnzymePrintActivity)
        llvm::errs() << " VALUE nonconst unknown global " << *Val << " type - "
                     << res.str() << "\n";
      ActiveValues.insert(Val);
      return false;
    }

    // ConstantExpr's are inactive if their arguments are inactive
    // Note that since there can't be a recursive constant this shouldn't
    // infinite loop
    if (auto ce = dyn_cast<ConstantExpr>(Val)) {
      if (ce->isCast()) {
        if (isConstantValue(TR, ce->getOperand(0))) {
          if (EnzymePrintActivity)
            llvm::errs() << " VALUE const cast from from operand " << *Val
                         << "\n";
          InsertConstantValue(TR, Val);
          return true;
        }
      }
      if (ce->getOpcode() == Instruction::GetElementPtr &&
          llvm::all_of(ce->operand_values(),
                       [&](Value *v) { return isConstantValue(TR, v); })) {
        if (isConstantValue(TR, ce->getOperand(0))) {
          if (EnzymePrintActivity)
            llvm::errs() << " VALUE const cast from gep operand " << *Val
                         << "\n";
          InsertConstantValue(TR, Val);
          return true;
        }
      }
      if (EnzymePrintActivity)
        llvm::errs() << " VALUE nonconst unknown expr " << *Val << "\n";
      ActiveValues.insert(Val);
      return false;
    }

    if (auto I = dyn_cast<Instruction>(Val)) {
      if (hasMetadata(I, "enzyme_active") ||
          hasMetadata(I, "enzyme_active_val")) {
        if (EnzymePrintActivity)
          llvm::errs() << "forced active val (MD)" << *Val << "\n";
        InsertConstantValue(TR, Val);
        return true;
      }
      if (hasMetadata(I, "enzyme_inactive") ||
          hasMetadata(I, "enzyme_inactive_val")) {
        if (EnzymePrintActivity)
          llvm::errs() << "forced inactive val (MD)" << *Val << "\n";
        InsertConstantValue(TR, Val);
        return true;
      }
    }
    if (auto CI = dyn_cast<CallInst>(Val)) {
      if (CI->hasFnAttr("enzyme_active") ||
          CI->hasFnAttr("enzyme_active_val") ||
          CI->getAttributes().hasAttribute(llvm::AttributeList::ReturnIndex,
                                           "enzyme_active")) {
        if (EnzymePrintActivity)
          llvm::errs() << "forced active val " << *Val << "\n";
        ActiveValues.insert(Val);
        return false;
      }
      if (CI->hasFnAttr("enzyme_inactive") ||
          CI->hasFnAttr("enzyme_inactive_val") ||
          CI->getAttributes().hasAttribute(llvm::AttributeList::ReturnIndex,
                                           "enzyme_inactive")) {
        if (EnzymePrintActivity)
          llvm::errs() << "forced inactive val " << *Val << "\n";
        InsertConstantValue(TR, Val);
        return true;
      }
      auto called = getFunctionFromCall(CI);

      if (called) {
        if (called->hasFnAttribute("enzyme_active") ||
            called->hasFnAttribute("enzyme_active_val") ||
            called->getAttributes().hasAttribute(
                llvm::AttributeList::ReturnIndex, "enzyme_active")) {
          if (EnzymePrintActivity)
            llvm::errs() << "forced active val " << *Val << "\n";
          ActiveValues.insert(Val);
          return false;
        }
        if (called->hasFnAttribute("enzyme_inactive") ||
            called->hasFnAttribute("enzyme_inactive_val") ||
            called->getAttributes().hasAttribute(
                llvm::AttributeList::ReturnIndex, "enzyme_inactive")) {
          if (EnzymePrintActivity)
            llvm::errs() << "forced inactive val " << *Val << "\n";
          InsertConstantValue(TR, Val);
          return true;
        }
      }
    }
    if (auto BO = dyn_cast<BinaryOperator>(Val)) {
      // x & 0b100000 is definitionally inactive
      //  + if floating point, this returns either +/- 0
      //  if int/pointer, this contains no info
      if (BO->getOpcode() == Instruction::And) {
        auto &DL = BO->getParent()->getParent()->getParent()->getDataLayout();
        for (int i = 0; i < 2; ++i) {
          auto FT =
              TR.query(BO->getOperand(1 - i))
                  .IsAllFloat((DL.getTypeSizeInBits(BO->getType()) + 7) / 8);
          // If ^ against 0b10000000000 and a float the result is a float
          if (FT)
            if (containsOnlyAtMostTopBit(BO->getOperand(i), FT, DL)) {
              if (EnzymePrintActivity)
                llvm::errs() << " inactive bithack " << *Val << "\n";
              InsertConstantValue(TR, Val);
              return true;
            }
        }
      }
    }
  }

  std::shared_ptr<ActivityAnalyzer> UpHypothesis;

  // Handle types that could contain pointers
  //  Consider all types except
  //   * floating point types (since those are assumed not pointers)
  //   * integers that we know are not pointers
  bool containsPointer = true;
  if (Val->getType()->isFPOrFPVectorTy())
    containsPointer = false;
  if (!TR.intType(1, Val, /*errIfNotFound*/ false).isPossiblePointer())
    containsPointer = false;

  if (containsPointer && !isValuePotentiallyUsedAsPointer(Val)) {
    containsPointer = false;
  }

  // We do this pointer dance here to ensure that any derived pointers from
  // constant arguments are still constant, even id ATA is disabled.
  if (EnzymeDisableActivityAnalysis) {
    if (!containsPointer)
      return false;

    auto TmpOrig = getBaseObject(Val);

    if (auto LI = dyn_cast<LoadInst>(TmpOrig))
      return isConstantValue(TR, LI->getPointerOperand());
    if (isa<IntrinsicInst>(TmpOrig) &&
        (cast<IntrinsicInst>(TmpOrig)->getIntrinsicID() ==
             Intrinsic::nvvm_ldu_global_i ||
         cast<IntrinsicInst>(TmpOrig)->getIntrinsicID() ==
             Intrinsic::nvvm_ldu_global_p ||
         cast<IntrinsicInst>(TmpOrig)->getIntrinsicID() ==
             Intrinsic::nvvm_ldu_global_f ||
         cast<IntrinsicInst>(TmpOrig)->getIntrinsicID() ==
             Intrinsic::nvvm_ldg_global_i ||
         cast<IntrinsicInst>(TmpOrig)->getIntrinsicID() ==
             Intrinsic::nvvm_ldg_global_p ||
         cast<IntrinsicInst>(TmpOrig)->getIntrinsicID() ==
             Intrinsic::nvvm_ldg_global_f))
      return isConstantValue(TR, cast<Instruction>(TmpOrig)->getOperand(0));

    if (TmpOrig == Val)
      return false;
    return isConstantValue(TR, TmpOrig);
  }

  if (containsPointer) {
    // This value is certainly an integer (and only and integer, not a pointer
    // or float). Therefore its value is constant
    if (TR.query(Val)[{-1, -1}] == BaseType::Integer) {
      if (EnzymePrintActivity)
        llvm::errs() << " Value const as pointer to integer " << (int)directions
                     << " " << *Val << " " << TR.query(Val).str() << "\n";
      InsertConstantValue(TR, Val);
      return true;
    }

    auto TmpOrig = getBaseObject(Val);

    // If we know that our origin is inactive from its arguments,
    // we are definitionally inactive
    if (directions & UP) {
      // If we are derived from an argument our activity is equal to the
      // activity of the argument by definition
      if (auto arg = dyn_cast<Argument>(TmpOrig)) {
        if (!arg->hasByValAttr()) {
          bool res = isConstantValue(TR, TmpOrig);
          if (res) {
            if (EnzymePrintActivity)
              llvm::errs() << " arg const from orig val=" << *Val
                           << " orig=" << *TmpOrig << "\n";
            InsertConstantValue(TR, Val);
          } else {
            if (EnzymePrintActivity)
              llvm::errs() << " arg active from orig val=" << *Val
                           << " orig=" << *TmpOrig << "\n";
            ActiveValues.insert(Val);
          }
          return res;
        }
      }

      UpHypothesis =
          std::shared_ptr<ActivityAnalyzer>(new ActivityAnalyzer(*this, UP));
      UpHypothesis->ConstantValues.insert(Val);

      // If our origin is a load of a known inactive (say inactive argument), we
      // are also inactive
      if (auto PN = dyn_cast<PHINode>(TmpOrig)) {
        // Not taking fast path incase phi is recursive.
        Value *active = nullptr;
        for (auto &V : PN->incoming_values()) {
          if (!UpHypothesis->isConstantValue(TR, V.get())) {
            active = V.get();
            break;
          }
        }
        if (!active) {
          InsertConstantValue(TR, Val);
          if (TmpOrig != Val) {
            InsertConstantValue(TR, TmpOrig);
          }
          insertConstantsFrom(TR, *UpHypothesis);
          return true;
        } else {
          ReEvaluateValueIfInactiveValue[active].insert(Val);
          if (TmpOrig != Val) {
            ReEvaluateValueIfInactiveValue[active].insert(TmpOrig);
          }
        }
      } else if (auto LI = dyn_cast<LoadInst>(TmpOrig)) {

        if (directions == UP) {
          if (isConstantValue(TR, LI->getPointerOperand())) {
            InsertConstantValue(TR, Val);
            return true;
          }
        } else {
          if (UpHypothesis->isConstantValue(TR, LI->getPointerOperand())) {
            InsertConstantValue(TR, Val);
            insertConstantsFrom(TR, *UpHypothesis);
            return true;
          }
        }
        ReEvaluateValueIfInactiveValue[LI->getPointerOperand()].insert(Val);
        if (TmpOrig != Val) {
          ReEvaluateValueIfInactiveValue[LI->getPointerOperand()].insert(
              TmpOrig);
        }
      } else if (isa<IntrinsicInst>(TmpOrig) &&
                 (cast<IntrinsicInst>(TmpOrig)->getIntrinsicID() ==
                      Intrinsic::nvvm_ldu_global_i ||
                  cast<IntrinsicInst>(TmpOrig)->getIntrinsicID() ==
                      Intrinsic::nvvm_ldu_global_p ||
                  cast<IntrinsicInst>(TmpOrig)->getIntrinsicID() ==
                      Intrinsic::nvvm_ldu_global_f ||
                  cast<IntrinsicInst>(TmpOrig)->getIntrinsicID() ==
                      Intrinsic::nvvm_ldg_global_i ||
                  cast<IntrinsicInst>(TmpOrig)->getIntrinsicID() ==
                      Intrinsic::nvvm_ldg_global_p ||
                  cast<IntrinsicInst>(TmpOrig)->getIntrinsicID() ==
                      Intrinsic::nvvm_ldg_global_f)) {
        auto II = cast<IntrinsicInst>(TmpOrig);
        if (directions == UP) {
          if (isConstantValue(TR, II->getOperand(0))) {
            InsertConstantValue(TR, Val);
            return true;
          }
        } else {
          if (UpHypothesis->isConstantValue(TR, II->getOperand(0))) {
            InsertConstantValue(TR, Val);
            insertConstantsFrom(TR, *UpHypothesis);
            return true;
          }
        }
        ReEvaluateValueIfInactiveValue[II->getOperand(0)].insert(Val);
        if (TmpOrig != Val) {
          ReEvaluateValueIfInactiveValue[II->getOperand(0)].insert(TmpOrig);
        }
      } else if (auto op = dyn_cast<CallInst>(TmpOrig)) {
        if (op->hasFnAttr("enzyme_inactive") ||
            op->hasFnAttr("enzyme_inactive_val") ||
            op->getAttributes().hasAttribute(llvm::AttributeList::ReturnIndex,
                                             "enzyme_inactive")) {
          InsertConstantValue(TR, Val);
          insertConstantsFrom(TR, *UpHypothesis);
          return true;
        }
        auto called = getFunctionFromCall(op);

        StringRef funcName = getFuncNameFromCall(op);

        if (called &&
            (called->hasFnAttribute("enzyme_inactive") ||
             called->hasFnAttribute("enzyme_inactive_val") ||
             called->getAttributes().hasAttribute(
                 llvm::AttributeList::ReturnIndex, "enzyme_inactive"))) {
          InsertConstantValue(TR, Val);
          insertConstantsFrom(TR, *UpHypothesis);
          return true;
        }
        if (funcName == "free" || funcName == "_ZdlPv" ||
            funcName == "_ZdlPvm" || funcName == "munmap") {
          InsertConstantValue(TR, Val);
          insertConstantsFrom(TR, *UpHypothesis);
          return true;
        }

        auto dName = demangle(funcName.str());
        for (auto FuncName : DemangledKnownInactiveFunctionsStartingWith) {
          if (StringRef(dName).startswith(FuncName)) {
            InsertConstantValue(TR, Val);
            insertConstantsFrom(TR, *UpHypothesis);
            return true;
          }
        }

        for (auto FuncName : KnownInactiveFunctionsStartingWith) {
          if (funcName.startswith(FuncName)) {
            InsertConstantValue(TR, Val);
            insertConstantsFrom(TR, *UpHypothesis);
            return true;
          }
        }

        for (auto FuncName : KnownInactiveFunctionsContains) {
          if (funcName.contains(FuncName)) {
            InsertConstantValue(TR, Val);
            insertConstantsFrom(TR, *UpHypothesis);
            return true;
          }
        }

        if (KnownInactiveFunctions.count(funcName) ||
            MPIInactiveCommAllocators.find(funcName) !=
                MPIInactiveCommAllocators.end()) {
          InsertConstantValue(TR, Val);
          insertConstantsFrom(TR, *UpHypothesis);
          return true;
        }

        if (called && called->getIntrinsicID() == Intrinsic::trap) {
          InsertConstantValue(TR, Val);
          insertConstantsFrom(TR, *UpHypothesis);
          return true;
        }

        // If requesting empty unknown functions to be considered inactive,
        // abide by those rules
        if (called && EnzymeEmptyFnInactive && called->empty() &&
            !hasMetadata(called, "enzyme_gradient") &&
            !hasMetadata(called, "enzyme_derivative") &&
            !isAllocationFunction(funcName, TLI) &&
            !isDeallocationFunction(funcName, TLI) && !isa<IntrinsicInst>(op)) {
          InsertConstantValue(TR, Val);
          insertConstantsFrom(TR, *UpHypothesis);
          return true;
        }
        if (isAllocationFunction(funcName, TLI)) {
          // This pointer is inactive if it is either not actively stored to
          // and not actively loaded from.
          if (directions == DOWN) {
            for (auto UA :
                 {UseActivity::OnlyLoads, UseActivity::OnlyNonPointerStores,
                  UseActivity::AllStores, UseActivity::None}) {
              Instruction *LoadReval = nullptr;
              if (isValueInactiveFromUsers(TR, TmpOrig, UA, &LoadReval)) {
                InsertConstantValue(TR, Val);
                return true;
              }
              if (LoadReval && UA != UseActivity::AllStores) {
                ReEvaluateValueIfInactiveInst[LoadReval].insert(TmpOrig);
              }
            }
          } else if (directions & DOWN) {
            auto DownHypothesis = std::shared_ptr<ActivityAnalyzer>(
                new ActivityAnalyzer(*this, DOWN));
            DownHypothesis->ConstantValues.insert(TmpOrig);
            for (auto UA :
                 {UseActivity::OnlyLoads, UseActivity::OnlyNonPointerStores,
                  UseActivity::AllStores, UseActivity::None}) {
              Instruction *LoadReval = nullptr;
              if (DownHypothesis->isValueInactiveFromUsers(TR, TmpOrig, UA,
                                                           &LoadReval)) {
                insertConstantsFrom(TR, *DownHypothesis);
                InsertConstantValue(TR, Val);
                return true;
              } else {
                if (LoadReval && UA != UseActivity::AllStores) {
                  ReEvaluateValueIfInactiveInst[LoadReval].insert(TmpOrig);
                }
              }
            }
          }
          // If allocation function doesn't initialize inner pointers.
          // For example, julia allocations initialize inner pointers, but
          // malloc/etc just allocate the immediate memory.
          if (directions & DOWN &&
              (funcName == "malloc" || funcName == "calloc" ||
               funcName == "_Znwm" || funcName == "julia.gc_alloc_obj" ||
               funcName == "??2@YAPAXI@Z" || funcName == "??2@YAPEAX_K@Z" ||
               funcName == "jl_gc_alloc_typed" ||
               funcName == "ijl_gc_alloc_typed")) {
            std::shared_ptr<ActivityAnalyzer> Hypothesis =
                std::shared_ptr<ActivityAnalyzer>(
                    new ActivityAnalyzer(*this, directions));
            Hypothesis->ActiveValues.insert(Val);
            Instruction *LoadReval = nullptr;
            if (Hypothesis->isValueInactiveFromUsers(
                    TR, TmpOrig, UseActivity::OnlyStores, &LoadReval)) {
              insertConstantsFrom(TR, *Hypothesis);
              InsertConstantValue(TR, Val);
              return true;
            } else {
              if (LoadReval) {
                ReEvaluateValueIfInactiveInst[LoadReval].insert(TmpOrig);
              }
            }
          }
        }
        if (funcName == "jl_array_copy" || funcName == "ijl_array_copy") {
          // This pointer is inactive if it is either not actively stored to
          // and not actively loaded from.
          if (directions & DOWN && directions & UP) {
            if (UpHypothesis->isConstantValue(TR, op->getOperand(0))) {
              auto DownHypothesis = std::shared_ptr<ActivityAnalyzer>(
                  new ActivityAnalyzer(*this, DOWN));
              DownHypothesis->ConstantValues.insert(TmpOrig);
              for (auto UA :
                   {UseActivity::OnlyLoads, UseActivity::OnlyNonPointerStores,
                    UseActivity::AllStores, UseActivity::None}) {
                Instruction *LoadReval = nullptr;
                if (DownHypothesis->isValueInactiveFromUsers(TR, TmpOrig, UA,
                                                             &LoadReval)) {
                  insertConstantsFrom(TR, *DownHypothesis);
                  InsertConstantValue(TR, Val);
                  return true;
                } else {
                  if (LoadReval && UA != UseActivity::AllStores) {
                    ReEvaluateValueIfInactiveInst[LoadReval].insert(TmpOrig);
                  }
                }
              }
            }
          }
        }
      } else if (isa<AllocaInst>(Val)) {
        // This pointer is inactive if it is either not actively stored to or
        // not actively loaded from and is nonescaping by definition of being
        // alloca.
        //   When presuming the value is constant,
        //     OnlyStores is insufficient. This is because one could allocate
        //     memory, assumed inactive by definition since it is only stored
        //     into the hypothesized inactive alloca. However, one could load
        //     that pointer, and then use it as an active buffer.
        //   When presuming the value is active,
        //     OnlyStores should be fine, since any store will assume that
        //     its use by storing to the active alloca will be active unless
        //     the pointer being stored is otherwise guaranteed inactive (e.g.
        //     from the argument).
        if (directions == DOWN) {
          for (auto UA :
               {UseActivity::OnlyLoads, UseActivity::OnlyNonPointerStores,
                UseActivity::AllStores, UseActivity::None}) {
            Instruction *LoadReval = nullptr;
            if (isValueInactiveFromUsers(TR, TmpOrig, UA, &LoadReval)) {
              InsertConstantValue(TR, Val);
              return true;
            }
            if (LoadReval && UA != UseActivity::AllStores) {
              ReEvaluateValueIfInactiveInst[LoadReval].insert(TmpOrig);
            }
          }
        } else if (directions & DOWN) {
          auto DownHypothesis = std::shared_ptr<ActivityAnalyzer>(
              new ActivityAnalyzer(*this, DOWN));
          DownHypothesis->ConstantValues.insert(TmpOrig);
          for (auto UA :
               {UseActivity::OnlyLoads, UseActivity::OnlyNonPointerStores,
                UseActivity::AllStores, UseActivity::None}) {
            Instruction *LoadReval = nullptr;
            if (DownHypothesis->isValueInactiveFromUsers(TR, TmpOrig, UA,
                                                         &LoadReval)) {
              insertConstantsFrom(TR, *DownHypothesis);
              InsertConstantValue(TR, Val);
              return true;
            } else {
              if (LoadReval && UA != UseActivity::AllStores) {
                ReEvaluateValueIfInactiveInst[LoadReval].insert(TmpOrig);
              }
            }
          }
        }

        if (directions & DOWN) {
          std::shared_ptr<ActivityAnalyzer> Hypothesis =
              std::shared_ptr<ActivityAnalyzer>(
                  new ActivityAnalyzer(*this, directions));
          Hypothesis->ActiveValues.insert(Val);
          Instruction *LoadReval = nullptr;
          if (Hypothesis->isValueInactiveFromUsers(
                  TR, TmpOrig, UseActivity::OnlyStores, &LoadReval)) {
            insertConstantsFrom(TR, *Hypothesis);
            InsertConstantValue(TR, Val);
            return true;
          } else {
            if (LoadReval) {
              ReEvaluateValueIfInactiveInst[LoadReval].insert(TmpOrig);
            }
          }
        }
      }

      // otherwise if the origin is a previously derived known inactive value
      // assess
      // TODO here we would need to potentially consider loading an active
      // global as we again assume that active memory is passed explicitly as an
      // argument
      if (TmpOrig != Val) {
        if (isConstantValue(TR, TmpOrig)) {
          if (EnzymePrintActivity)
            llvm::errs() << " Potential Pointer(" << (int)directions << ") "
                         << *Val << " inactive from inactive origin "
                         << *TmpOrig << "\n";
          InsertConstantValue(TR, Val);
          return true;
        }
      }
      if (auto inst = dyn_cast<Instruction>(Val)) {
        if (!inst->mayReadFromMemory() && !isa<AllocaInst>(Val)) {
          if (directions == UP && !isa<PHINode>(inst)) {
            if (isInstructionInactiveFromOrigin(TR, inst, true)) {
              InsertConstantValue(TR, Val);
              return true;
            }
          } else {
            if (UpHypothesis->isInstructionInactiveFromOrigin(TR, inst, true)) {
              InsertConstantValue(TR, Val);
              insertConstantsFrom(TR, *UpHypothesis);
              return true;
            }
          }
        }
      }
    }

    // If not capable of looking at both users and uses, all the ways a pointer
    // can be loaded/stored cannot be assesed and therefore we default to assume
    // it to be active
    if (directions != 3) {
      if (EnzymePrintActivity)
        llvm::errs() << " <Potential Pointer assumed active at "
                     << (int)directions << ">" << *Val << "\n";
      ActiveValues.insert(Val);
      return false;
    }

    if (EnzymePrintActivity)
      llvm::errs() << " < MEMSEARCH" << (int)directions << ">" << *Val << "\n";
    // A pointer value is active if two things hold:
    //   an potentially active value is stored into the memory
    //   memory loaded from the value is used in an active way
    Instruction *potentiallyActiveStore = nullptr;
    bool potentialStore = false;
    Instruction *potentiallyActiveLoad = nullptr;

    // Assume the value (not instruction) is itself active
    // In spite of that can we show that there are either no active stores
    // or no active loads
    std::shared_ptr<ActivityAnalyzer> Hypothesis =
        std::shared_ptr<ActivityAnalyzer>(
            new ActivityAnalyzer(*this, directions));
    Hypothesis->ActiveValues.insert(Val);
    if (auto VI = dyn_cast<Instruction>(Val)) {
      if (UpHypothesis->isInstructionInactiveFromOrigin(TR, VI, true)) {
        Hypothesis->DeducingPointers.insert(Val);
        if (EnzymePrintActivity)
          llvm::errs() << " constant instruction hypothesis: " << *VI << "\n";
      } else {
        if (EnzymePrintActivity)
          llvm::errs() << " cannot show constant instruction hypothesis: "
                       << *VI << "\n";
      }
    }

    auto checkActivity = [&](Instruction *I) {
      if (notForAnalysis.count(I->getParent()))
        return false;

      if (isa<FenceInst>(I))
        return false;

      // If this is a malloc or free, this doesn't impact the activity
      if (auto CI = dyn_cast<CallInst>(I)) {
        if (CI->hasFnAttr("enzyme_inactive") ||
            CI->hasFnAttr("enzyme_inactive_inst"))
          return false;

        if (auto iasm = dyn_cast<InlineAsm>(CI->getCalledOperand())) {
          if (StringRef(iasm->getAsmString()).contains("exit") ||
              StringRef(iasm->getAsmString()).contains("cpuid"))
            return false;
        }

        auto F = getFunctionFromCall(CI);
        StringRef funcName = getFuncNameFromCall(CI);

        if (F && (F->hasFnAttribute("enzyme_inactive") ||
                  F->hasFnAttribute("enzyme_inactive_inst"))) {
          return false;
        }
        if (isAllocationFunction(funcName, TLI) ||
            isDeallocationFunction(funcName, TLI)) {
          return false;
        }
        if (KnownInactiveFunctions.count(funcName) ||
            MPIInactiveCommAllocators.find(funcName) !=
                MPIInactiveCommAllocators.end()) {
          return false;
        }
        if (KnownInactiveFunctionInsts.count(funcName)) {
          return false;
        }
        if (isMemFreeLibMFunction(funcName)) {
          return false;
        }

        auto dName = demangle(funcName.str());
        for (auto FuncName : DemangledKnownInactiveFunctionsStartingWith) {
          if (StringRef(dName).startswith(FuncName)) {
            return false;
          }
        }

        for (auto FuncName : KnownInactiveFunctionsStartingWith) {
          if (funcName.startswith(FuncName)) {
            return false;
          }
        }
        for (auto FuncName : KnownInactiveFunctionsContains) {
          if (funcName.contains(FuncName)) {
            return false;
          }
        }

        if (funcName == "__cxa_guard_acquire" ||
            funcName == "__cxa_guard_release" ||
            funcName == "__cxa_guard_abort" || funcName == "posix_memalign" ||
            funcName == "cuMemAllocAsync" || funcName == "cuMemAlloc" ||
            funcName == "cuMemAlloc_v2" || funcName == "cudaMallocAsync" ||
            funcName == "cudaMallocHost" ||
            funcName == "cudaMallocFromPoolAsync") {
          return false;
        }

        if (F) {
          if (KnownInactiveIntrinsics.count(F->getIntrinsicID())) {
            return false;
          }
        }
      }

      Value *memval = Val;

      // BasicAA stupidy assumes that non-pointer's don't alias
      // if this is a nonpointer, use something else to force alias
      // consideration
      if (!memval->getType()->isPointerTy()) {
        if (auto ci = dyn_cast<CastInst>(Val)) {
          if (ci->getOperand(0)->getType()->isPointerTy()) {
            memval = ci->getOperand(0);
          }
        }
        for (auto user : Val->users()) {
          if (isa<CastInst>(user) && user->getType()->isPointerTy()) {
            memval = user;
            break;
          }
        }
      }

#if LLVM_VERSION_MAJOR >= 12
      auto AARes = AA.getModRefInfo(
          I, MemoryLocation(memval, LocationSize::beforeOrAfterPointer()));
#else
      auto AARes =
          AA.getModRefInfo(I, MemoryLocation(memval, LocationSize::unknown()));
#endif

      // Still having failed to replace the location used by AA, fall back to
      // getModref against any location.
      if (!memval->getType()->isPointerTy()) {
        if (auto CB = dyn_cast<CallInst>(I)) {
#if LLVM_VERSION_MAJOR >= 16
          AARes = AA.getMemoryEffects(CB).getModRef();
#else
          AARes = createModRefInfo(AA.getModRefBehavior(CB));
#endif
        } else {
          bool mayRead = I->mayReadFromMemory();
          bool mayWrite = I->mayWriteToMemory();
          AARes = mayRead ? (mayWrite ? ModRefInfo::ModRef : ModRefInfo::Ref)
                          : (mayWrite ? ModRefInfo::Mod : ModRefInfo::NoModRef);
        }
      }

      if (auto CB = dyn_cast<CallInst>(I)) {
        if (CB->onlyAccessesInaccessibleMemory())
          AARes = ModRefInfo::NoModRef;

        bool ReadOnly = isReadOnly(CB);

        bool WriteOnly = isWriteOnly(CB);

        if (ReadOnly && WriteOnly)
          AARes = ModRefInfo::NoModRef;
        else if (WriteOnly) {
          if (isRefSet(AARes)) {
            AARes = isModSet(AARes) ? ModRefInfo::Mod : ModRefInfo::NoModRef;
          }
        } else if (ReadOnly) {
          if (isModSet(AARes)) {
            AARes = isRefSet(AARes) ? ModRefInfo::Ref : ModRefInfo::NoModRef;
          }
        }
      }

      // TODO this aliasing information is too conservative, the question
      // isn't merely aliasing but whether there is a path for THIS value to
      // eventually be loaded by it not simply because there isnt aliasing

      // If we haven't already shown a potentially active load
      // check if this loads the given value and is active
      if ((!potentiallyActiveLoad || !potentiallyActiveStore) &&
          isRefSet(AARes)) {
        if (EnzymePrintActivity)
          llvm::errs() << "potential active load: " << *I << "\n";
        if (isa<LoadInst>(I) || (isa<IntrinsicInst>(I) &&
                                 (cast<IntrinsicInst>(I)->getIntrinsicID() ==
                                      Intrinsic::nvvm_ldu_global_i ||
                                  cast<IntrinsicInst>(I)->getIntrinsicID() ==
                                      Intrinsic::nvvm_ldu_global_p ||
                                  cast<IntrinsicInst>(I)->getIntrinsicID() ==
                                      Intrinsic::nvvm_ldu_global_f ||
                                  cast<IntrinsicInst>(I)->getIntrinsicID() ==
                                      Intrinsic::nvvm_ldg_global_i ||
                                  cast<IntrinsicInst>(I)->getIntrinsicID() ==
                                      Intrinsic::nvvm_ldg_global_p ||
                                  cast<IntrinsicInst>(I)->getIntrinsicID() ==
                                      Intrinsic::nvvm_ldg_global_f))) {
          // If the ref'ing value is a load check if the loaded value is
          // active
          if (!Hypothesis->isConstantValue(TR, I)) {
            potentiallyActiveLoad = I;
            // returns whether seen
            std::function<bool(Value * V, SmallPtrSetImpl<Value *> &)>
                loadCheck = [&](Value *V, SmallPtrSetImpl<Value *> &Seen) {
                  if (Seen.count(V))
                    return false;
                  Seen.insert(V);
                  if (TR.query(V)[{-1}].isPossiblePointer()) {
                    for (auto UU : V->users()) {
                      auto U = cast<Instruction>(UU);
                      if (U->mayWriteToMemory()) {
                        if (!Hypothesis->isConstantInstruction(TR, U)) {
                          if (EnzymePrintActivity)
                            llvm::errs() << "potential active store via "
                                            "pointer in load: "
                                         << *I << " of " << *Val << " via "
                                         << *U << "\n";
                          potentiallyActiveStore = U;
                          return true;
                        }
                      }

                      if (U != Val && !Hypothesis->isConstantValue(TR, U)) {
                        if (loadCheck(U, Seen))
                          return true;
                      }
                    }
                  }
                  return false;
                };
            SmallPtrSet<Value *, 2> Seen;
            loadCheck(I, Seen);
          }
        } else if (auto MTI = dyn_cast<MemTransferInst>(I)) {
          if (!Hypothesis->isConstantValue(TR, MTI->getArgOperand(0))) {
            potentiallyActiveLoad = MTI;
            if (TR.query(Val)[{-1, -1}].isPossiblePointer()) {
              if (EnzymePrintActivity)
                llvm::errs()
                    << "potential active store via pointer in memcpy: " << *I
                    << " of " << *Val << "\n";
              potentiallyActiveStore = MTI;
            }
          }
        } else {
          // Otherwise fallback and check any part of the instruction is
          // active
          // TODO: note that this can be optimized (especially for function
          // calls)
          // Notably need both to check the result and instruction since
          // A load that has as result an active pointer is not an active
          // instruction, but does have an active value
          if (!Hypothesis->isConstantInstruction(TR, I) ||
              (I != Val && !Hypothesis->isConstantValue(TR, I))) {
            potentiallyActiveLoad = I;
            // If this a potential pointer of pointer AND
            //     double** Val;
            //
            if (TR.query(Val)[{-1, -1}].isPossiblePointer()) {
              // If this instruction either:
              //   1) can actively store into the inner pointer, even
              //      if it doesn't store into the outer pointer. Actively
              //      storing into the outer pointer is handled by the isMod
              //      case.
              //        I(double** readonly Val, double activeX) {
              //            double* V0 = Val[0]
              //            V0 = activeX;
              //        }
              //   2) may return an active pointer loaded from Val
              //        double* I = *Val;
              //        I[0] = active;
              //
              if ((I->mayWriteToMemory() &&
                   !Hypothesis->isConstantInstruction(TR, I)) ||
                  (!Hypothesis->DeducingPointers.count(I) &&
                   !Hypothesis->isConstantValue(TR, I) &&
                   TR.query(I)[{-1}].isPossiblePointer())) {
                if (EnzymePrintActivity)
                  llvm::errs() << "potential active store via pointer in "
                                  "unknown inst: "
                               << *I << " of " << *Val << "\n";
                potentiallyActiveStore = I;
              }
            }
          }
        }
      }
      if ((!potentiallyActiveStore || !potentialStore) && isModSet(AARes)) {
        if (EnzymePrintActivity)
          llvm::errs() << "potential active store: " << *I << " Val=" << *Val
                       << "\n";
        if (auto SI = dyn_cast<StoreInst>(I)) {
          bool cop = !Hypothesis->isConstantValue(TR, SI->getValueOperand());
          if (EnzymePrintActivity)
            llvm::errs() << " -- store potential activity: " << (int)cop
                         << " - " << *SI << " of "
                         << " Val=" << *Val << "\n";
          potentialStore = true;
          if (cop)
            potentiallyActiveStore = SI;
        } else if (auto MTI = dyn_cast<MemTransferInst>(I)) {
          bool cop = !Hypothesis->isConstantValue(TR, MTI->getArgOperand(1));
          potentialStore = true;
          if (cop)
            potentiallyActiveStore = MTI;
        } else if (isa<MemSetInst>(I)) {
          potentialStore = true;
        } else {
          // Otherwise fallback and check if the instruction is active
          // TODO: note that this can be optimized (especially for function
          // calls)
          auto cop = !Hypothesis->isConstantInstruction(TR, I);
          if (EnzymePrintActivity)
            llvm::errs() << " -- unknown store potential activity: " << (int)cop
                         << " - " << *I << " of "
                         << " Val=" << *Val << "\n";
          potentialStore = true;
          if (cop)
            potentiallyActiveStore = I;
        }
      }
      if (potentiallyActiveStore && potentiallyActiveLoad)
        return true;
      return false;
    };

    // Search through all the instructions in this function
    // for potential loads / stores of this value.
    //
    // We can choose to only look at potential follower instructions
    // if the value is created by the instruction (alloca, noalias)
    // since no potentially active store to the same location can occur
    // prior to its creation. Otherwise, check all instructions in the
    // function as a store to an aliasing location may have occured
    // prior to the instruction generating the value.

    if (auto VI = dyn_cast<AllocaInst>(Val)) {
      allFollowersOf(VI, checkActivity);
    } else if (auto VI = dyn_cast<CallInst>(Val)) {
      if (VI->hasRetAttr(Attribute::NoAlias))
        allFollowersOf(VI, checkActivity);
      else {
        for (BasicBlock &BB : *TR.getFunction()) {
          if (notForAnalysis.count(&BB))
            continue;
          for (Instruction &I : BB) {
            if (checkActivity(&I))
              goto activeLoadAndStore;
          }
        }
      }
    } else if (isa<Argument>(Val) || isa<Instruction>(Val)) {
      for (BasicBlock &BB : *TR.getFunction()) {
        if (notForAnalysis.count(&BB))
          continue;
        for (Instruction &I : BB) {
          if (checkActivity(&I))
            goto activeLoadAndStore;
        }
      }
    } else {
      llvm::errs() << "unknown pointer value type: " << *Val << "\n";
      assert(0 && "unknown pointer value type");
      llvm_unreachable("unknown pointer value type");
    }

  activeLoadAndStore:;
    if (EnzymePrintActivity)
      llvm::errs() << " </MEMSEARCH" << (int)directions << ">" << *Val
                   << " potentiallyActiveLoad=" << potentiallyActiveLoad
                   << " potentiallyActiveStore=" << potentiallyActiveStore
                   << " potentialStore=" << potentialStore << "\n";
    if (potentiallyActiveLoad && potentiallyActiveStore) {
      ReEvaluateValueIfInactiveInst[potentiallyActiveLoad].insert(Val);
      ReEvaluateValueIfInactiveInst[potentiallyActiveStore].insert(Val);
      insertAllFrom(TR, *Hypothesis, Val, TmpOrig);
      if (TmpOrig != Val) {
        ReEvaluateValueIfInactiveValue[TmpOrig].insert(Val);
        ReEvaluateValueIfInactiveInst[potentiallyActiveLoad].insert(TmpOrig);
        ReEvaluateValueIfInactiveInst[potentiallyActiveStore].insert(TmpOrig);
      }
      return false;
    } else {
      // We now know that there isn't a matching active load/store pair in this
      // function. Now the only way that this memory can facilitate a transfer
      // of active information is if it is done outside of the function

      // This can happen if either:
      // a) the memory had an active load or store before this function was
      // called b) the memory had an active load or store after this function
      // was called

      // Case a) can occur if:
      //    1) this memory came from an active global
      //    2) this memory came from an active argument
      //    3) this memory came from a load from active memory
      // In other words, assuming this value is inactive, going up this
      // location's argument must be inactive

      assert(UpHypothesis);
      // UpHypothesis.ConstantValues.insert(val);
      if (DeducingPointers.size() == 0)
        UpHypothesis->insertConstantsFrom(TR, *Hypothesis);
      assert(directions & UP);
      bool ActiveUp =
          !isa<Argument>(Val) &&
          !UpHypothesis->isInstructionInactiveFromOrigin(TR, Val, true);

      // Case b) can occur if:
      //    1) this memory is used as part of an active return
      //    2) this memory is stored somewhere

      // We never verify that an origin wasn't stored somewhere or returned.
      // to remedy correctness for now let's do something extremely simple
      std::shared_ptr<ActivityAnalyzer> DownHypothesis =
          std::shared_ptr<ActivityAnalyzer>(new ActivityAnalyzer(*this, DOWN));
      DownHypothesis->ConstantValues.insert(Val);
      DownHypothesis->insertConstantsFrom(TR, *Hypothesis);
      bool ActiveDown =
          DownHypothesis->isValueActivelyStoredOrReturned(TR, Val);
      // BEGIN TEMPORARY

      if (!ActiveDown && TmpOrig != Val) {

        if (isa<Argument>(TmpOrig) || isa<GlobalVariable>(TmpOrig) ||
            isa<AllocaInst>(TmpOrig) || isAllocationCall(TmpOrig, TLI)) {
          std::shared_ptr<ActivityAnalyzer> DownHypothesis2 =
              std::shared_ptr<ActivityAnalyzer>(
                  new ActivityAnalyzer(*DownHypothesis, DOWN));
          DownHypothesis2->ConstantValues.insert(TmpOrig);
          if (DownHypothesis2->isValueActivelyStoredOrReturned(TR, TmpOrig)) {
            if (EnzymePrintActivity)
              llvm::errs() << " active from ivasor: " << *TmpOrig << "\n";
            ActiveDown = true;
          }
        } else {
          // unknown origin that could've been stored/returned/etc
          if (EnzymePrintActivity)
            llvm::errs() << " active from unknown origin: " << *TmpOrig << "\n";
          ActiveDown = true;
        }
      }

      // END TEMPORARY

      // We can now consider the three places derivative information can be
      // transferred
      //   Case A) From the origin
      //   Case B) Though the return
      //   Case C) Within the function (via either load or store)

      bool ActiveMemory = false;

      // If it is transferred via active origin and return, clearly this is
      // active
      ActiveMemory |= (ActiveUp && ActiveDown);

      // If we come from an active origin and load, memory is clearly active
      ActiveMemory |= (ActiveUp && potentiallyActiveLoad);

      // If we come from an active origin and only store into it, it changes
      // future state
      ActiveMemory |= (ActiveUp && potentialStore);

      // If we go to an active return and store active memory, this is active
      ActiveMemory |= (ActiveDown && potentialStore);
      // Actually more generally, if we are ActiveDown (returning memory that is
      // used) in active return, we must be active. This is necessary to ensure
      // mallocs have their differential shadows created when returned [TODO
      // investigate more]
      ActiveMemory |= ActiveDown;

      // If we go to an active return and only load it, however, that doesnt
      // transfer derivatives and we can say this memory is inactive

      if (EnzymePrintActivity)
        llvm::errs() << " @@MEMSEARCH" << (int)directions << ">" << *Val
                     << " potentiallyActiveLoad=" << potentiallyActiveLoad
                     << " potentialStore=" << potentialStore
                     << " ActiveUp=" << ActiveUp << " ActiveDown=" << ActiveDown
                     << " ActiveMemory=" << ActiveMemory << "\n";

      if (ActiveMemory) {
        ActiveValues.insert(Val);
        assert(Hypothesis->directions == directions);
        assert(Hypothesis->ActiveValues.count(Val));
        insertAllFrom(TR, *Hypothesis, Val, TmpOrig);
        if (TmpOrig != Val)
          ReEvaluateValueIfInactiveValue[TmpOrig].insert(Val);
        return false;
      } else {
        InsertConstantValue(TR, Val);
        insertConstantsFrom(TR, *Hypothesis);
        if (DeducingPointers.size() == 0)
          insertConstantsFrom(TR, *UpHypothesis);
        insertConstantsFrom(TR, *DownHypothesis);
        return true;
      }
    }
  }

  // For all non-pointers, it is now sufficient to simply prove that
  // either activity does not flow in, or activity does not flow out
  // This alone cuts off the flow (being unable to flow through memory)

  // Not looking at uses to prove inactive (definition of up), if the creator of
  // this value is inactive, we are inactive Since we won't look at uses to
  // prove, we can inductively assume this is inactive
  if (directions & UP) {
    if (directions == UP && !isa<PHINode>(Val)) {
      if (isInstructionInactiveFromOrigin(TR, Val, true)) {
        InsertConstantValue(TR, Val);
        return true;
      } else if (auto I = dyn_cast<Instruction>(Val)) {
        if (directions == 3) {
          for (auto &op : I->operands()) {
            if (!UpHypothesis->isConstantValue(TR, op)) {
              ReEvaluateValueIfInactiveValue[op].insert(I);
            }
          }
        }
      }
    } else {
      UpHypothesis =
          std::shared_ptr<ActivityAnalyzer>(new ActivityAnalyzer(*this, UP));
      UpHypothesis->ConstantValues.insert(Val);
      if (UpHypothesis->isInstructionInactiveFromOrigin(TR, Val, true)) {
        insertConstantsFrom(TR, *UpHypothesis);
        InsertConstantValue(TR, Val);
        return true;
      } else if (auto I = dyn_cast<Instruction>(Val)) {
        if (directions == 3) {
          for (auto &op : I->operands()) {
            if (!UpHypothesis->isConstantValue(TR, op)) {
              ReEvaluateValueIfInactiveValue[op].insert(I);
            }
          }
        }
      }
    }
  }

  if (directions & DOWN) {
    // Not looking at users to prove inactive (definition of down)
    // If all users are inactive, this is therefore inactive.
    // Since we won't look at origins to prove, we can inductively assume this
    // is inactive

    // As an optimization if we are going down already
    // and we won't use ourselves (done by PHI's), we
    // dont need to inductively assume we're true
    // and can instead use this object!
    if (directions == DOWN && !isa<PHINode>(Val)) {
      if (isValueInactiveFromUsers(TR, Val, UseActivity::None)) {
        if (UpHypothesis)
          insertConstantsFrom(TR, *UpHypothesis);
        InsertConstantValue(TR, Val);
        return true;
      }
    } else {
      auto DownHypothesis =
          std::shared_ptr<ActivityAnalyzer>(new ActivityAnalyzer(*this, DOWN));
      DownHypothesis->ConstantValues.insert(Val);
      if (DownHypothesis->isValueInactiveFromUsers(TR, Val,
                                                   UseActivity::None)) {
        insertConstantsFrom(TR, *DownHypothesis);
        if (UpHypothesis)
          insertConstantsFrom(TR, *UpHypothesis);
        InsertConstantValue(TR, Val);
        return true;
      }
    }
  }

  if (EnzymePrintActivity)
    llvm::errs() << " Value nonconstant (couldn't disprove)[" << (int)directions
                 << "]" << *Val << "\n";
  ActiveValues.insert(Val);
  return false;
}

/// Is the instruction guaranteed to be inactive because of its operands
bool ActivityAnalyzer::isInstructionInactiveFromOrigin(TypeResults const &TR,
                                                       llvm::Value *val,
                                                       bool considerValue) {
  // Must be an analyzer only searching up
  assert(directions == UP);
  assert(!isa<Argument>(val));
  assert(!isa<GlobalVariable>(val));

  // Not an instruction and thus not legal to search for activity via operands
  if (!isa<Instruction>(val)) {
    llvm::errs() << "unknown pointer source: " << *val << "\n";
    assert(0 && "unknown pointer source");
    llvm_unreachable("unknown pointer source");
    return false;
  }

  Instruction *inst = cast<Instruction>(val);
  if (EnzymePrintActivity)
    llvm::errs() << " < UPSEARCH" << (int)directions << ">" << *inst << "\n";

  // cpuid is explicitly an inactive instruction
  if (auto call = dyn_cast<CallInst>(inst)) {
    if (auto iasm = dyn_cast<InlineAsm>(call->getCalledOperand())) {
      if (StringRef(iasm->getAsmString()).contains("cpuid")) {
        if (EnzymePrintActivity)
          llvm::errs() << " constant instruction from known cpuid instruction "
                       << *inst << "\n";
        return true;
      }
    }
  }

  if (auto SI = dyn_cast<StoreInst>(inst)) {
    // if either src or dst is inactive, there cannot be a transfer of active
    // values and thus the store is inactive
    if (isConstantValue(TR, SI->getValueOperand()) ||
        isConstantValue(TR, SI->getPointerOperand())) {
      if (EnzymePrintActivity)
        llvm::errs() << " constant instruction as store operand is inactive "
                     << *inst << "\n";
      return true;
    }
  }

  if (!considerValue) {
    if (auto IEI = dyn_cast<InsertElementInst>(inst)) {
      auto &DL = IEI->getParent()->getParent()->getParent()->getDataLayout();
      if ((!isPossibleFloat(TR, IEI->getOperand(0), DL) ||
           isConstantValue(TR, IEI->getOperand(0))) &&
          (!isPossibleFloat(TR, IEI->getOperand(1), DL) ||
           isConstantValue(TR, IEI->getOperand(1)))) {
        if (EnzymePrintActivity)
          llvm::errs()
              << " constant instruction as inserting known pointer or inactive"
              << *inst << "\n";
        return true;
      }
    }
    if (auto IEI = dyn_cast<InsertValueInst>(inst)) {
      auto &DL = IEI->getParent()->getParent()->getParent()->getDataLayout();
      if ((!isPossibleFloat(TR, IEI->getAggregateOperand(), DL) ||
           isConstantValue(TR, IEI->getAggregateOperand())) &&
          (!isPossibleFloat(TR, IEI->getInsertedValueOperand(), DL) ||
           isConstantValue(TR, IEI->getInsertedValueOperand()))) {
        if (EnzymePrintActivity)
          llvm::errs()
              << " constant instruction as inserting known pointer or inactive"
              << *inst << "\n";
        return true;
      }
    }
    if (auto PN = dyn_cast<PHINode>(inst)) {
      std::deque<PHINode *> todo = {PN};
      SmallPtrSet<PHINode *, 1> done;
      SmallVector<Value *, 2> incoming;
      while (todo.size()) {
        auto cur = todo.back();
        todo.pop_back();
        if (done.count(cur))
          continue;
        done.insert(cur);
        for (auto &V : cur->incoming_values()) {
          if (auto P = dyn_cast<PHINode>(V)) {
            todo.push_back(P);
            continue;
          }
          incoming.push_back(V);
        }
      }
      bool legal = true;
      auto &DL = PN->getParent()->getParent()->getParent()->getDataLayout();
      for (auto V : incoming) {
        if (isPossibleFloat(TR, V, DL) && !isConstantValue(TR, V)) {
          legal = false;
          break;
        }
      }
      if (legal) {
        if (EnzymePrintActivity)
          llvm::errs()
              << " constant instruction as phi of known pointer or inactive"
              << *inst << "\n";
        return true;
      }
    }
  }

  if (auto MTI = dyn_cast<MemTransferInst>(inst)) {
    // if either src or dst is inactive, there cannot be a transfer of active
    // values and thus the store is inactive
    if (isConstantValue(TR, MTI->getArgOperand(0)) ||
        isConstantValue(TR, MTI->getArgOperand(1))) {
      if (EnzymePrintActivity)
        llvm::errs() << " constant instruction as memtransfer " << *inst
                     << "\n";
      return true;
    }
  }

  if (auto op = dyn_cast<CallInst>(inst)) {
    if (op->hasFnAttr("enzyme_inactive") ||
        op->hasFnAttr("enzyme_inactive_val")) {
      return true;
    }
    // Calls to print/assert/cxa guard are definitionally inactive
    llvm::Value *callVal;
    callVal = op->getCalledOperand();
    StringRef funcName = getFuncNameFromCall(op);
    auto called = getFunctionFromCall(op);

    if (called && (called->hasFnAttribute("enzyme_inactive") ||
                   called->hasFnAttribute("enzyme_inactive_val"))) {
      return true;
    }
    if (funcName == "free" || funcName == "_ZdlPv" || funcName == "_ZdlPvm" ||
        funcName == "munmap") {
      return true;
    }

    auto dName = demangle(funcName.str());
    for (auto FuncName : DemangledKnownInactiveFunctionsStartingWith) {
      if (StringRef(dName).startswith(FuncName)) {
        return true;
      }
    }

    for (auto FuncName : KnownInactiveFunctionsStartingWith) {
      if (funcName.startswith(FuncName)) {
        return true;
      }
    }

    for (auto FuncName : KnownInactiveFunctionsContains) {
      if (funcName.contains(FuncName)) {
        return true;
      }
    }

    if (KnownInactiveFunctions.count(funcName) ||
        MPIInactiveCommAllocators.find(funcName) !=
            MPIInactiveCommAllocators.end()) {
      if (EnzymePrintActivity)
        llvm::errs() << "constant(" << (int)directions
                     << ") up-knowninactivecall " << *inst << "\n";
      return true;
    }

    if (called && called->getIntrinsicID() == Intrinsic::trap)
      return true;

    // If requesting empty unknown functions to be considered inactive, abide
    // by those rules
    if (called && EnzymeEmptyFnInactive && called->empty() &&
        !hasMetadata(called, "enzyme_gradient") &&
        !hasMetadata(called, "enzyme_derivative") &&
        !isAllocationFunction(funcName, TLI) &&
        !isDeallocationFunction(funcName, TLI) && !isa<IntrinsicInst>(op)) {
      if (EnzymePrintActivity)
        llvm::errs() << "constant(" << (int)directions << ") up-emptyconst "
                     << *inst << "\n";
      return true;
    }
    if (!isa<Constant>(callVal) && isConstantValue(TR, callVal)) {
      if (EnzymePrintActivity)
        llvm::errs() << "constant(" << (int)directions << ") up-constfn "
                     << *inst << " - " << *callVal << "\n";
      return true;
    }
  }
  // Intrinsics known always to be inactive
  if (auto II = dyn_cast<IntrinsicInst>(inst)) {
    if (KnownInactiveIntrinsics.count(II->getIntrinsicID())) {
      if (EnzymePrintActivity)
        llvm::errs() << "constant(" << (int)directions << ") up-intrinsic "
                     << *inst << "\n";
      return true;
    }
    if (isIntelSubscriptIntrinsic(*II)) {
      // The only argument that can make an llvm.intel.subscript intrinsic
      // active is the pointer operand
      const unsigned int ptrArgIdx = 3;
      if (isConstantValue(TR, II->getOperand(ptrArgIdx))) {
        if (EnzymePrintActivity)
          llvm::errs() << "constant(" << (int)directions << ") up-intrinsic "
                       << *inst << "\n";
        return true;
      }
      return false;
    }
  }

  if (auto gep = dyn_cast<GetElementPtrInst>(inst)) {
    // A gep's only args that could make it active is the pointer operand
    if (isConstantValue(TR, gep->getPointerOperand())) {
      if (EnzymePrintActivity)
        llvm::errs() << "constant(" << (int)directions << ") up-gep " << *inst
                     << "\n";
      return true;
    }
    return false;
  } else if (auto ci = dyn_cast<CallInst>(inst)) {
    bool seenuse = false;

    propagateArgumentInformation(TLI, *ci, [&](Value *a) {
      if (!isConstantValue(TR, a)) {
        seenuse = true;
        if (EnzymePrintActivity)
          llvm::errs() << "nonconstant(" << (int)directions << ")  up-call "
                       << *inst << " op " << *a << "\n";
        return true;
      }
      return false;
    });
    if (EnzymeGlobalActivity) {
      if (!ci->onlyAccessesArgMemory() && !ci->doesNotAccessMemory()) {
        bool legalUse = false;

        StringRef funcName = getFuncNameFromCall(ci);

        if (funcName == "") {
        } else if (isMemFreeLibMFunction(funcName) ||
                   isDebugFunction(ci->getCalledFunction()) ||
                   isCertainPrint(funcName) ||
                   isAllocationFunction(funcName, TLI) ||
                   isDeallocationFunction(funcName, TLI)) {
          legalUse = true;
        }

        if (!legalUse) {
          if (EnzymePrintActivity)
            llvm::errs() << "nonconstant(" << (int)directions << ")  up-global "
                         << *inst << "\n";
          seenuse = true;
        }
      }
    }

    if (!seenuse) {
      if (EnzymePrintActivity)
        llvm::errs() << "constant(" << (int)directions << ")  up-call:" << *inst
                     << "\n";
      return true;
    }
    return !seenuse;
  } else if (auto si = dyn_cast<SelectInst>(inst)) {

    if (isConstantValue(TR, si->getTrueValue()) &&
        isConstantValue(TR, si->getFalseValue())) {

      if (EnzymePrintActivity)
        llvm::errs() << "constant(" << (int)directions << ") up-sel:" << *inst
                     << "\n";
      return true;
    }
    return false;
  } else if (isa<SIToFPInst>(inst) || isa<UIToFPInst>(inst) ||
             isa<FPToSIInst>(inst) || isa<FPToUIInst>(inst)) {

    if (EnzymePrintActivity)
      llvm::errs() << "constant(" << (int)directions << ") up-fpcst:" << *inst
                   << "\n";
    return true;
  } else {
    bool seenuse = false;
    //! TODO does not consider reading from global memory that is active and not
    //! an argument
    for (auto &a : inst->operands()) {
      bool hypval = isConstantValue(TR, a);
      if (!hypval) {
        if (EnzymePrintActivity)
          llvm::errs() << "nonconstant(" << (int)directions << ")  up-inst "
                       << *inst << " op " << *a << "\n";
        seenuse = true;
        break;
      }
    }

    if (!seenuse) {
      if (EnzymePrintActivity)
        llvm::errs() << "constant(" << (int)directions << ")  up-inst:" << *inst
                     << "\n";
      return true;
    }
    return false;
  }
}

/// Is the value free of any active uses
bool ActivityAnalyzer::isValueInactiveFromUsers(TypeResults const &TR,
                                                llvm::Value *const val,
                                                UseActivity PUA,
                                                Instruction **FoundInst) {
  assert(directions & DOWN);
  // Must be an analyzer only searching down, unless used outside
  // assert(directions == DOWN);

  // To ensure we can call down

  if (EnzymePrintActivity)
    llvm::errs() << " <Value USESEARCH" << (int)directions << ">" << *val
                 << " UA=" << to_string(PUA) << "\n";

  bool seenuse = false;
  // user, predecessor
  std::deque<std::tuple<User *, Value *, UseActivity>> todo;
  for (const auto a : val->users()) {
    todo.push_back(std::make_tuple(a, val, PUA));
  }
  std::set<std::tuple<User *, Value *, UseActivity>> done = {};

  SmallSet<Value *, 1> AllocaSet;

  if (isa<AllocaInst>(val))
    AllocaSet.insert(val);

  if (PUA == UseActivity::None && isAllocationCall(val, TLI))
    AllocaSet.insert(val);

  while (todo.size()) {
    auto pair = todo.front();
    todo.pop_front();
    if (done.count(pair))
      continue;
    done.insert(pair);
    User *a = std::get<0>(pair);
    Value *parent = std::get<1>(pair);
    UseActivity UA = std::get<2>(pair);

    if (auto LI = dyn_cast<LoadInst>(a)) {
      if (UA == UseActivity::OnlyStores)
        continue;
      if (UA == UseActivity::OnlyNonPointerStores ||
          UA == UseActivity::AllStores) {
        if (!TR.query(LI)[{-1}].isPossiblePointer())
          continue;
      }
    }

    if (EnzymePrintActivity)
      llvm::errs() << "      considering use of " << *val << " - " << *a
                   << "\n";

    // Only ignore stores to the operand, not storing the operand
    // somewhere
    if (auto SI = dyn_cast<StoreInst>(a)) {
      if (SI->getValueOperand() != parent) {
        if (UA == UseActivity::OnlyLoads) {
          continue;
        }
        if (UA != UseActivity::AllStores &&
            (ConstantValues.count(SI->getValueOperand()) ||
             isa<ConstantInt>(SI->getValueOperand())))
          continue;
        if (UA == UseActivity::None) {
          // If storing into itself, all potential uses are taken care of
          // elsewhere in the recursion.
          bool shouldContinue = true;
          SmallVector<Value *, 1> vtodo = {SI->getValueOperand()};
          SmallSet<Value *, 1> seen;
          SmallSet<Value *, 1> newAllocaSet;
          while (vtodo.size()) {
            auto TmpOrig = vtodo.back();
            vtodo.pop_back();
            if (seen.count(TmpOrig))
              continue;
            seen.insert(TmpOrig);
            if (AllocaSet.count(TmpOrig)) {
              continue;
            }
            if (isa<AllocaInst>(TmpOrig)) {
              newAllocaSet.insert(TmpOrig);
              continue;
            }
            if (isAllocationCall(TmpOrig, TLI)) {
              newAllocaSet.insert(TmpOrig);
              continue;
            }
            if (isa<UndefValue>(TmpOrig) || isa<ConstantInt>(TmpOrig) ||
                isa<ConstantPointerNull>(TmpOrig) || isa<ConstantFP>(TmpOrig)) {
              continue;
            }
            if (auto LI = dyn_cast<LoadInst>(TmpOrig)) {
              vtodo.push_back(LI->getPointerOperand());
              continue;
            }
            if (auto CD = dyn_cast<ConstantDataSequential>(TmpOrig)) {
              for (size_t i = 0, len = CD->getNumElements(); i < len; i++)
                vtodo.push_back(CD->getElementAsConstant(i));
              continue;
            }
            if (auto CD = dyn_cast<ConstantAggregate>(TmpOrig)) {
              for (size_t i = 0, len = CD->getNumOperands(); i < len; i++)
                vtodo.push_back(CD->getOperand(i));
              continue;
            }
            if (auto GV = dyn_cast<GlobalVariable>(TmpOrig)) {
              // If operating under the assumption globals are inactive unless
              // explicitly marked as active, this is inactive
              if (!hasMetadata(GV, "enzyme_shadow") &&
                  EnzymeNonmarkedGlobalsInactive) {
                continue;
              }
              if (hasMetadata(GV, "enzyme_inactive")) {
                continue;
              }
              if (GV->getName().contains("enzyme_const") ||
                  InactiveGlobals.count(GV->getName())) {
                continue;
              }
            }
            auto TmpOrig_2 = getBaseObject(TmpOrig);
            if (TmpOrig != TmpOrig_2) {
              vtodo.push_back(TmpOrig_2);
              continue;
            }
            if (EnzymePrintActivity)
              llvm::errs() << "      -- cannot continuing indirect store from "
                           << *val << " due to " << *TmpOrig << "\n";
            shouldContinue = false;
            break;
          }
          if (shouldContinue) {
            if (EnzymePrintActivity)
              llvm::errs() << "      -- continuing indirect store from " << *val
                           << " into:\n";
            done.insert(std::make_tuple((User *)SI, SI->getValueOperand(), UA));
            for (auto TmpOrig : newAllocaSet) {

              for (const auto a : TmpOrig->users()) {
                todo.push_back(std::make_tuple(a, TmpOrig, UA));
                if (EnzymePrintActivity)
                  llvm::errs() << "         ** " << *a << "\n";
              }
              AllocaSet.insert(TmpOrig);
              shouldContinue = true;
            }
            continue;
          }
        }
      }
      if (SI->getPointerOperand() != parent) {
        auto TmpOrig = SI->getPointerOperand();
        // If storing into itself, all potential uses are taken care of
        // elsewhere in the recursion.
        bool shouldContinue = false;
        while (1) {
          if (AllocaSet.count(TmpOrig)) {
            shouldContinue = true;
            break;
          }
          if (isa<AllocaInst>(TmpOrig)) {
            done.insert(
                std::make_tuple((User *)SI, SI->getPointerOperand(), UA));
            for (const auto a : TmpOrig->users()) {
              todo.push_back(std::make_tuple(a, TmpOrig, UA));
            }
            AllocaSet.insert(TmpOrig);
            shouldContinue = true;
            break;
          }
          if (PUA == UseActivity::None) {
            if (auto LI = dyn_cast<LoadInst>(TmpOrig)) {
              TmpOrig = LI->getPointerOperand();
              continue;
            }
            if (isAllocationCall(TmpOrig, TLI)) {
              done.insert(
                  std::make_tuple((User *)SI, SI->getPointerOperand(), UA));
              for (const auto a : TmpOrig->users()) {
                todo.push_back(std::make_tuple(a, TmpOrig, UA));
              }
              AllocaSet.insert(TmpOrig);
              shouldContinue = true;
              break;
            }
          }
          auto TmpOrig_2 = getBaseObject(TmpOrig);
          if (TmpOrig != TmpOrig_2) {
            TmpOrig = TmpOrig_2;
            continue;
          }
          break;
        }
        if (shouldContinue) {
          if (EnzymePrintActivity)
            llvm::errs() << "      -- continuing indirect store2 from " << *val
                         << " via " << *TmpOrig << "\n";
          continue;
        }
      }
      if (PUA == UseActivity::OnlyLoads) {
        auto TmpOrig = getBaseObject(SI->getPointerOperand());
        if (TmpOrig == val) {
          continue;
        }
      }
    }

    if (!isa<Instruction>(a)) {
      if (auto CE = dyn_cast<ConstantExpr>(a)) {
        for (auto u : CE->users()) {
          todo.push_back(std::make_tuple(u, (Value *)CE, UA));
        }
        continue;
      }
      if (isa<ConstantData>(a)) {
        continue;
      }

      if (EnzymePrintActivity)
        llvm::errs() << "      unknown non instruction use of " << *val << " - "
                     << *a << "\n";
      return false;
    }

    if (isa<AllocaInst>(a)) {
      if (EnzymePrintActivity)
        llvm::errs() << "found constant(" << (int)directions
                     << ")  allocainst use:" << *val << " user " << *a << "\n";
      continue;
    }

    if (isa<SIToFPInst>(a) || isa<UIToFPInst>(a) || isa<FPToSIInst>(a) ||
        isa<FPToUIInst>(a)) {
      if (EnzymePrintActivity)
        llvm::errs() << "found constant(" << (int)directions
                     << ")  si-fp use:" << *val << " user " << *a << "\n";
      continue;
    }

    // if this instruction is in a different function, conservatively assume
    // it is active
    Function *InstF = cast<Instruction>(a)->getParent()->getParent();
    while (PPC.CloneOrigin.find(InstF) != PPC.CloneOrigin.end())
      InstF = PPC.CloneOrigin[InstF];

    Function *F = TR.getFunction();
    while (PPC.CloneOrigin.find(F) != PPC.CloneOrigin.end())
      F = PPC.CloneOrigin[F];

    if (InstF != F) {
      if (EnzymePrintActivity)
        llvm::errs() << "found use in different function(" << (int)directions
                     << ")  val:" << *val << " user " << *a << " in "
                     << InstF->getName() << "@" << InstF
                     << " self: " << F->getName() << "@" << F << "\n";
      return false;
    }
    if (cast<Instruction>(a)->getParent()->getParent() != TR.getFunction())
      continue;

    // This use is only active if specified
    if (isa<ReturnInst>(a)) {
      if (ActiveReturns == DIFFE_TYPE::CONSTANT &&
          UA != UseActivity::AllStores) {
        continue;
      } else {
        return false;
      }
    }

    if (auto II = dyn_cast<IntrinsicInst>(a)) {
      if (isIntelSubscriptIntrinsic(*II) &&
          (II->getOperand(/*ptrArgIdx=*/3) != parent)) {
        continue;
      }
    }

    if (auto call = dyn_cast<CallInst>(a)) {
      bool mayWrite = false;
      bool mayRead = false;
      bool mayCapture = false;

      auto F = getFunctionFromCall(call);

      size_t idx = 0;
#if LLVM_VERSION_MAJOR >= 14
      for (auto &arg : call->args())
#else
      for (auto &arg : call->arg_operands())
#endif
      {
        if (arg != parent) {
          idx++;
          continue;
        }

        bool NoCapture = isNoCapture(call, idx);

        mayCapture |= !NoCapture;

        bool ReadOnly = isReadOnly(call, idx);

        mayWrite |= !ReadOnly;

        bool WriteOnly = isWriteOnly(call, idx);

        mayRead |= !WriteOnly;
      }

      bool ConstantArg = isFunctionArgumentConstant(call, parent);
      if (ConstantArg && UA != UseActivity::AllStores) {
        if (EnzymePrintActivity) {
          llvm::errs() << "Value found constant callinst use:" << *val
                       << " user " << *call << "\n";
        }
        continue;
      }

      if (!mayCapture) {
        if (!mayRead && UA == UseActivity::OnlyLoads) {
          if (EnzymePrintActivity) {
            llvm::errs() << "Value found non-loading use:" << *val << " user "
                         << *call << "\n";
          }
          continue;
        }
        if (!mayWrite && UA == UseActivity::OnlyStores) {
          if (EnzymePrintActivity) {
            llvm::errs() << "Value found non-writing use:" << *val << " user "
                         << *call << "\n";
          }
          continue;
        }
        if (!mayWrite && (UA == UseActivity::OnlyNonPointerStores ||
                          UA == UseActivity::AllStores)) {
          if (!mayRead || !TR.query(parent)[{-1, -1}].isPossiblePointer()) {
            if (EnzymePrintActivity) {
              llvm::errs()
                  << "Value found non-writing and non pointer loading use:"
                  << *val << " user " << *call << "\n";
            }
            continue;
          }
        }
      }

      if (F) {
        if (UA == UseActivity::AllStores &&
            (F->getName() == "julia.write_barrier" ||
             F->getName() == "julia.write_barrier_binding"))
          continue;
        if (F->getIntrinsicID() == Intrinsic::memcpy ||
            F->getIntrinsicID() == Intrinsic::memmove) {

          // copies of constant string data do not impact activity.
          if (auto cexpr = dyn_cast<ConstantExpr>(call->getArgOperand(1))) {
            if (cexpr->getOpcode() == Instruction::GetElementPtr) {
              if (auto GV = dyn_cast<GlobalVariable>(cexpr->getOperand(0))) {
                if (GV->hasInitializer() && GV->isConstant()) {
                  if (auto CDA =
                          dyn_cast<ConstantDataArray>(GV->getInitializer())) {
                    if (CDA->getType()->getElementType()->isIntegerTy(8))
                      continue;
                  }
                }
              }
            }
          }

          // Only need to care about loads from
          if (UA == UseActivity::OnlyLoads && call->getArgOperand(1) != parent)
            continue;

          // Only need to care about store from
          if (call->getArgOperand(0) != parent) {
            if (UA == UseActivity::OnlyStores)
              continue;
            else if (UA == UseActivity::OnlyNonPointerStores ||
                     UA == UseActivity::AllStores) {
              // todo can change this to query either -1 (all mem) or 0..size
              // (if size of copy is const)
              if (!TR.query(call->getArgOperand(1))[{-1, -1}]
                       .isPossiblePointer())
                continue;
            }
          }

          bool shouldContinue = false;
          if (UA != UseActivity::AllStores)
            for (int arg = 0; arg < 2; arg++)
              if (call->getArgOperand(arg) != parent &&
                  (arg == 0 || (PUA == UseActivity::None))) {
                Value *TmpOrig = call->getOperand(arg);
                while (1) {
                  if (AllocaSet.count(TmpOrig)) {
                    shouldContinue = true;
                    break;
                  }
                  if (isa<AllocaInst>(TmpOrig)) {
                    done.insert(std::make_tuple((User *)call,
                                                call->getArgOperand(arg), UA));
                    for (const auto a : TmpOrig->users()) {
                      todo.push_back(std::make_tuple(a, TmpOrig, UA));
                    }
                    AllocaSet.insert(TmpOrig);
                    shouldContinue = true;
                    break;
                  }
                  if (PUA == UseActivity::None) {
                    if (auto LI = dyn_cast<LoadInst>(TmpOrig)) {
                      TmpOrig = LI->getPointerOperand();
                      continue;
                    }
                    if (isAllocationCall(TmpOrig, TLI)) {
                      done.insert(std::make_tuple(
                          (User *)call, call->getArgOperand(arg), UA));
                      for (const auto a : TmpOrig->users()) {
                        todo.push_back(std::make_tuple(a, TmpOrig, UA));
                      }
                      AllocaSet.insert(TmpOrig);
                      shouldContinue = true;
                      break;
                    }
                  }
                  auto TmpOrig_2 = getBaseObject(TmpOrig);
                  if (TmpOrig != TmpOrig_2) {
                    TmpOrig = TmpOrig_2;
                    continue;
                  }
                  break;
                }
                if (shouldContinue)
                  break;
              }

          if (shouldContinue)
            continue;
        }
      } else if (PUA == UseActivity::None || PUA == UseActivity::OnlyStores) {
        // If calling a function derived from an alloca of this value,
        // the function is only active if the function stored into
        // the allocation is active (all functions not explicitly marked
        // inactive), or one of the args to the call is active
        Value *operand = call->getCalledOperand();

        bool toContinue = false;
        if (isa<LoadInst>(operand)) {
          bool legal = true;

#if LLVM_VERSION_MAJOR >= 14
          for (unsigned i = 0; i < call->arg_size() + 1; ++i)
#else
          for (unsigned i = 0; i < call->getNumArgOperands() + 1; ++i)
#endif
          {
            Value *a = call->getOperand(i);

            if (isa<ConstantInt>(a))
              continue;

            Value *ptr = a;
            bool subValue = false;
            while (ptr) {
              auto TmpOrig2 = getBaseObject(ptr);
              if (AllocaSet.count(TmpOrig2)) {
                subValue = true;
                break;
              }
              if (isa<AllocaInst>(TmpOrig2)) {
                done.insert(std::make_tuple((User *)call, a, UA));
                for (const auto a : TmpOrig2->users()) {
                  todo.push_back(std::make_tuple(a, TmpOrig2, UA));
                }
                AllocaSet.insert(TmpOrig2);
                subValue = true;
                break;
              }

              if (PUA == UseActivity::None) {
                if (isAllocationCall(TmpOrig2, TLI)) {
                  done.insert(std::make_tuple((User *)call, a, UA));
                  for (const auto a : TmpOrig2->users()) {
                    todo.push_back(std::make_tuple(a, TmpOrig2, UA));
                  }
                  AllocaSet.insert(TmpOrig2);
                  subValue = true;
                  break;
                }
                if (auto L = dyn_cast<LoadInst>(TmpOrig2)) {
                  ptr = L->getPointerOperand();
                } else
                  ptr = nullptr;
              } else
                ptr = nullptr;
            }
            if (subValue)
              continue;
            legal = false;
            break;
          }
          if (legal) {
            toContinue = true;
          }
        }
        if (toContinue) {
          if (EnzymePrintActivity) {
            llvm::errs() << "Value found indirect call use which must be "
                            "constant as all stored functions are constant val:"
                         << *val << " user " << *call << "\n";
          }
          for (auto u : call->users()) {
            todo.push_back(std::make_tuple(u, a, UseActivity::None));
          }
          continue;
        }
      }
    }

    // For an inbound gep, args which are not the pointer being offset
    // are not used in an active way by definition.
    if (auto gep = dyn_cast<GetElementPtrInst>(a)) {
      if (gep->isInBounds() && gep->getPointerOperand() != parent)
        continue;
    }

    // If this doesn't write to memory this can only be an active use
    // if its return is used in an active way, therefore add this to
    // the list of users to analyze
    if (auto I = dyn_cast<Instruction>(a)) {
      if (notForAnalysis.count(I->getParent())) {
        if (EnzymePrintActivity) {
          llvm::errs() << "Value found constant unreachable inst use:" << *val
                       << " user " << *I << "\n";
        }
        continue;
      }
      if (UA != UseActivity::AllStores && ConstantInstructions.count(I)) {
        if (I->getType()->isVoidTy() || I->getType()->isTokenTy() ||
            ConstantValues.count(I)) {
          if (EnzymePrintActivity) {
            llvm::errs() << "Value found constant inst use:" << *val << " user "
                         << *I << "\n";
          }
          continue;
        }
        UseActivity NU = UA;
        if (UA == UseActivity::OnlyLoads || UA == UseActivity::OnlyStores ||
            UA == UseActivity::OnlyNonPointerStores) {
          if (!isPointerArithmeticInst(I))
            NU = UseActivity::None;
        }

        for (auto u : I->users()) {
          todo.push_back(std::make_tuple(u, (Value *)I, NU));
        }
        continue;
      }
      if (!I->mayWriteToMemory() || isa<LoadInst>(I)) {
        if (TR.query(I)[{-1}].isIntegral()) {
          continue;
        }
        UseActivity NU = UA;
        if (UA == UseActivity::OnlyLoads || UA == UseActivity::OnlyStores ||
            UA == UseActivity::OnlyNonPointerStores) {
          if (!isPointerArithmeticInst(I))
            NU = UseActivity::None;
        }

        for (auto u : I->users()) {
          todo.push_back(std::make_tuple(u, (Value *)I, NU));
        }
        continue;
      }

      if (FoundInst)
        *FoundInst = I;
    }

    if (EnzymePrintActivity)
      llvm::errs() << "Value nonconstant inst (uses):" << *val << " user " << *a
                   << "\n";
    seenuse = true;
    break;
  }

  if (EnzymePrintActivity)
    llvm::errs() << " </Value USESEARCH" << (int)directions
                 << " const=" << (!seenuse) << ">" << *val << "\n";
  return !seenuse;
}

/// Is the value potentially actively returned or stored
bool ActivityAnalyzer::isValueActivelyStoredOrReturned(TypeResults const &TR,
                                                       llvm::Value *val,
                                                       bool outside) {
  // Must be an analyzer only searching down
  if (!outside)
    assert(directions == DOWN);

  bool ignoreStoresInto = true;
  auto key = std::make_pair(ignoreStoresInto, val);
  if (StoredOrReturnedCache.find(key) != StoredOrReturnedCache.end()) {
    return StoredOrReturnedCache[key];
  }

  if (EnzymePrintActivity)
    llvm::errs() << " <ASOR" << (int)directions
                 << " ignoreStoresinto=" << ignoreStoresInto << ">" << *val
                 << "\n";

  StoredOrReturnedCache[key] = false;

  for (const auto a : val->users()) {
    if (isa<AllocaInst>(a)) {
      continue;
    }
    // Loading a value prevents its pointer from being captured
    if (isa<LoadInst>(a)) {
      continue;
    }

    if (isa<ReturnInst>(a)) {
      if (ActiveReturns == DIFFE_TYPE::CONSTANT)
        continue;

      if (EnzymePrintActivity)
        llvm::errs() << " </ASOR" << (int)directions
                     << " ignoreStoresInto=" << ignoreStoresInto << ">"
                     << " active from-ret>" << *val << "\n";
      StoredOrReturnedCache[key] = true;
      return true;
    }

    if (auto call = dyn_cast<CallInst>(a)) {
      if (!couldFunctionArgumentCapture(call, val)) {
        continue;
      }
      bool ConstantArg = isFunctionArgumentConstant(call, val);
      if (ConstantArg) {
        continue;
      }
    }

    if (auto SI = dyn_cast<StoreInst>(a)) {
      // If we are being stored into, not storing this value
      // this case can be skipped
      if (SI->getValueOperand() != val) {
        if (!ignoreStoresInto) {
          // Storing into active value, return true
          if (!isConstantValue(TR, SI->getValueOperand())) {
            StoredOrReturnedCache[key] = true;
            if (EnzymePrintActivity)
              llvm::errs() << " </ASOR" << (int)directions
                           << " ignoreStoresInto=" << ignoreStoresInto
                           << " active from-store>" << *val
                           << " store into=" << *SI << "\n";
            return true;
          }
        }
        continue;
      } else {
        // Storing into active memory, return true
        if (!isConstantValue(TR, SI->getPointerOperand())) {
          StoredOrReturnedCache[key] = true;
          if (EnzymePrintActivity)
            llvm::errs() << " </ASOR" << (int)directions
                         << " ignoreStoresInto=" << ignoreStoresInto
                         << " active from-store>" << *val << " store=" << *SI
                         << "\n";
          return true;
        }
        continue;
      }
    }

    if (auto inst = dyn_cast<Instruction>(a)) {
      if (!inst->mayWriteToMemory() ||
          (isa<CallInst>(inst) && (AA.onlyReadsMemory(cast<CallInst>(inst)) ||
                                   isReadOnly(cast<CallInst>(inst))))) {
        // if not written to memory and returning a known constant, this
        // cannot be actively returned/stored
        if (inst->getParent()->getParent() == TR.getFunction() &&
            isConstantValue(TR, a)) {
          continue;
        }
        // if not written to memory and returning a value itself
        // not actively stored or returned, this is not actively
        // stored or returned
        if (!isValueActivelyStoredOrReturned(TR, a, outside)) {
          continue;
        }
      }
    }

    if (isAllocationCall(a, TLI)) {
      // if not written to memory and returning a known constant, this
      // cannot be actively returned/stored
      if (isConstantValue(TR, a)) {
        continue;
      }
      // if not written to memory and returning a value itself
      // not actively stored or returned, this is not actively
      // stored or returned
      if (!isValueActivelyStoredOrReturned(TR, a, outside)) {
        continue;
      }
    } else if (isDeallocationCall(a, TLI)) {
      // freeing memory never counts
      continue;
    }
    // fallback and conservatively assume that if the value is written to
    // it is written to active memory
    // TODO handle more memory instructions above to be less conservative

    if (EnzymePrintActivity)
      llvm::errs() << " </ASOR" << (int)directions
                   << " ignoreStoresInto=" << ignoreStoresInto
                   << " active from-unknown>" << *val << " - use=" << *a
                   << "\n";
    return StoredOrReturnedCache[key] = true;
  }

  if (EnzymePrintActivity)
    llvm::errs() << " </ASOR" << (int)directions
                 << " ignoreStoresInto=" << ignoreStoresInto << " inactive>"
                 << *val << "\n";
  return false;
}

void ActivityAnalyzer::InsertConstantInstruction(TypeResults const &TR,
                                                 llvm::Instruction *I) {
  ConstantInstructions.insert(I);
  auto found = ReEvaluateValueIfInactiveInst.find(I);
  if (found == ReEvaluateValueIfInactiveInst.end())
    return;
  auto set = std::move(ReEvaluateValueIfInactiveInst[I]);
  ReEvaluateValueIfInactiveInst.erase(I);
  for (auto toeval : set) {
    if (!ActiveValues.count(toeval))
      continue;
    ActiveValues.erase(toeval);
    if (EnzymePrintActivity)
      llvm::errs() << " re-evaluating activity of val " << *toeval
                   << " due to inst " << *I << "\n";
    isConstantValue(TR, toeval);
  }
}

void ActivityAnalyzer::InsertConstantValue(TypeResults const &TR,
                                           llvm::Value *V) {
  ConstantValues.insert(V);
  if (InsertConstValueRecursionHandler) {
    InsertConstValueRecursionHandler->push_back(V);
    return;
  }
  SmallVector<Value *, 1> InsertConstValueRecursionHandlerTmp;
  InsertConstValueRecursionHandlerTmp.push_back(V);
  InsertConstValueRecursionHandler = &InsertConstValueRecursionHandlerTmp;
  while (InsertConstValueRecursionHandlerTmp.size()) {
    auto V = InsertConstValueRecursionHandlerTmp.back();
    InsertConstValueRecursionHandlerTmp.pop_back();
    auto found = ReEvaluateValueIfInactiveValue.find(V);
    if (found != ReEvaluateValueIfInactiveValue.end()) {
      auto set = std::move(ReEvaluateValueIfInactiveValue[V]);
      ReEvaluateValueIfInactiveValue.erase(V);
      for (auto toeval : set) {
        if (!ActiveValues.count(toeval))
          continue;
        ActiveValues.erase(toeval);
        if (EnzymePrintActivity)
          llvm::errs() << " re-evaluating activity of val " << *toeval
                       << " due to value " << *V << "\n";
        isConstantValue(TR, toeval);
      }
    }
    auto found2 = ReEvaluateInstIfInactiveValue.find(V);
    if (found2 != ReEvaluateInstIfInactiveValue.end()) {
      auto set = std::move(ReEvaluateInstIfInactiveValue[V]);
      ReEvaluateInstIfInactiveValue.erase(V);
      for (auto toeval : set) {
        if (!ActiveInstructions.count(toeval))
          continue;
        ActiveInstructions.erase(toeval);
        if (EnzymePrintActivity)
          llvm::errs() << " re-evaluating activity of inst " << *toeval
                       << " due to value " << *V << "\n";
        isConstantInstruction(TR, toeval);
      }
    }
  }
  InsertConstValueRecursionHandler = nullptr;
}
