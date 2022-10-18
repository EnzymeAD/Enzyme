#include "ActivityAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Intrinsics.h"

using namespace mlir;
using namespace mlir::enzyme;

const char *KnownInactiveFunctionsStartingWith[] = {
    "f90io",
    "$ss5print",
    "_ZTv0_n24_NSoD", //"1Ev, 0Ev
    "_ZNSt16allocator_traitsISaIdEE10deallocate",
    "_ZNSaIcED1Ev",
    "_ZNSaIcEC1Ev",
#if LLVM_VERSION_MAJOR <= 8
    // TODO this returns allocated memory and thus can be an active value
    // "_ZNSt16allocator_traits",
    "_ZN4core3fmt",
    "_ZN3std2io5stdio6_print",
    "_ZNSt7__cxx1112basic_string",
    "_ZNSt7__cxx1118basic_string",
    "_ZNKSt7__cxx1112basic_string",
    "_ZN9__gnu_cxx12__to_xstringINSt7__cxx1112basic_string",
    "_ZNSt12__basic_file",
    "_ZNSt15basic_streambufIcSt11char_traits",
    "_ZNSt13basic_filebufIcSt11char_traits",
    "_ZNSt14basic_ofstreamIcSt11char_traits",
    "_ZNSi4readEPcl",
    "_ZNKSt14basic_ifstreamIcSt11char_traits",
    "_ZNSt14basic_ifstreamIcSt11char_traits",
    "_ZNSo5writeEPKcl",
    "_ZNSt19basic_ostringstreamIcSt11char_traits",
    "_ZStrsIcSt11char_traitsIcESaIcEERSt13basic_istream",
    "_ZStlsIcSt11char_traitsIcESaIcEERSt13basic_ostream",
    "_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traits",
    "_ZNKSt7__cxx1119basic_ostringstreamIcSt11char_traits",
    "_ZNSoD1Ev",
    "_ZNSoC1EPSt15basic_streambufIcSt11char_traits",
    "_ZStlsISt11char_traitsIcEERSt13basic_ostream",
    "_ZSt16__ostream_insert",
    "_ZStlsIwSt11char_traitsIwEERSt13basic_ostream",
    "_ZNSo9_M_insert",
    "_ZNSt13basic_ostream",
    "_ZNSo3put",
    "_ZNKSt5ctypeIcE13_M_widen_init",
    "_ZNSi3get",
    "_ZNSi7getline",
    "_ZNSirsER",
    "_ZNSt7__cxx1115basic_stringbuf",
    "_ZNKSt7__cxx1115basic_stringbuf",
    "_ZNSi6ignore",
    "_ZNSt8ios_base",
    "_ZNKSt9basic_ios",
    "_ZNSt9basic_ios",
    "_ZStorSt13_Ios_OpenmodeS_",
    "_ZNSt6locale",
    "_ZNKSt6locale4name",
    "_ZStL8__ioinit"
    "_ZNSt9basic_ios",
    "_ZSt4cout",
    "_ZSt3cin",
    "_ZNSi10_M_extract",
    "_ZNSolsE",
    "_ZSt5flush",
    "_ZNSo5flush",
    "_ZSt4endl",
    "_ZNSaIcE",
#endif
};

const char *KnownInactiveFunctionsContains[] = {
    "__enzyme_float", "__enzyme_double", "__enzyme_integer",
    "__enzyme_pointer"};

const std::set<std::string> InactiveGlobals = {
    "ompi_request_null", "ompi_mpi_double", "ompi_mpi_comm_world", "stderr",
    "stdout", "stdin", "_ZSt3cin", "_ZSt4cout", "_ZSt5wcout", "_ZSt4cerr",
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
    "_ZTVN10__cxxabiv121__vmi_class_type_infoE"};

const std::map<std::string, size_t> MPIInactiveCommAllocators = {
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
const std::set<std::string> KnownInactiveFunctionInsts = {
    "__dynamic_cast",
    "_ZSt18_Rb_tree_decrementPKSt18_Rb_tree_node_base",
    "_ZSt18_Rb_tree_incrementPKSt18_Rb_tree_node_base",
    "_ZSt18_Rb_tree_decrementPSt18_Rb_tree_node_base",
    "_ZSt18_Rb_tree_incrementPSt18_Rb_tree_node_base",
    "jl_ptr_to_array",
    "jl_ptr_to_array_1d"};

const std::set<std::string> KnownInactiveFunctions = {
    "abort",
    "time",
    "memcmp",
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
};

const char *DemangledKnownInactiveFunctionsStartingWith[] = {
    // TODO this returns allocated memory and thus can be an active value
    // "std::allocator",
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

};

  // TODO: these should be lifted to proper operations and implemented via the
  // interface.
const static unsigned constantIntrinsics[] = {
    llvm::Intrinsic::nvvm_barrier0,
    llvm::Intrinsic::nvvm_barrier0_popc,
    llvm::Intrinsic::nvvm_barrier0_and,
    llvm::Intrinsic::nvvm_barrier0_or,
    llvm::Intrinsic::nvvm_membar_cta,
    llvm::Intrinsic::nvvm_membar_gl,
    llvm::Intrinsic::nvvm_membar_sys,
    llvm::Intrinsic::amdgcn_s_barrier,
    llvm::Intrinsic::assume,
    llvm::Intrinsic::stacksave,
    llvm::Intrinsic::stackrestore,
    llvm::Intrinsic::lifetime_start,
    llvm::Intrinsic::lifetime_end,
    llvm::Intrinsic::dbg_addr,
    llvm::Intrinsic::dbg_declare,
    llvm::Intrinsic::dbg_value,
    llvm::Intrinsic::invariant_start,
    llvm::Intrinsic::invariant_end,
    llvm::Intrinsic::var_annotation,
    llvm::Intrinsic::ptr_annotation,
    llvm::Intrinsic::annotation,
    llvm::Intrinsic::codeview_annotation,
    llvm::Intrinsic::expect,
    llvm::Intrinsic::type_test,
    llvm::Intrinsic::donothing,
    llvm::Intrinsic::prefetch,
    llvm::Intrinsic::trap,
#if LLVM_VERSION_MAJOR >= 8
    llvm::Intrinsic::is_constant,
#endif
    llvm::Intrinsic::memset,
};

static Operation *getFunctionFromCall(CallOpInterface iface) {
  auto symbol = iface.getCallableForCallee().dyn_cast<SymbolRefAttr>();
  if (!symbol)
    return nullptr;

  return SymbolTable::lookupNearestSymbolFrom(iface.getOperation(), symbol);
}

/// Is the use of value val as an argument of call CI known to be inactive
/// This tool can only be used when in DOWN mode
bool ActivityAnalyzer::isFunctionArgumentConstant(CallOpInterface CI, Value val) {
  assert(directions & DOWN);

  if (CI->hasAttr("enzyme_inactive"))
    return true;

  Operation *F = getFunctionFromCall(CI);

  // Indirect function calls may actively use the argument
  if (F == nullptr)
    return false;

  if (F->hasAttr("enzyme_inactive")) {
    return true;
  }

  StringRef Name = cast<SymbolOpInterface>(F).getName();

  // Allocations, deallocations, and c++ guards don't impact the activity
  // of arguments
  // if (isAllocationFunction(Name, TLI) || isDeallocationFunction(Name, TLI))
  //   return true;
  if (Name == "posix_memalign")
    return true;

#if LLVM_VERSION_MAJOR >= 9
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
#endif
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
  if (KnownInactiveFunctions.count(Name.str())) {
    return true;
  }

  if (MPIInactiveCommAllocators.find(Name.str()) !=
      MPIInactiveCommAllocators.end()) {
    return true;
  }

  for (unsigned intrinsicID : constantIntrinsics) {
    if (Name.startswith(llvm::Intrinsic::getBaseName(intrinsicID)))
      return true;
  }

  // TODO: this should be lifted to proper operations via the interface.
  // /// Only the first argument (magnitude) of copysign is active
  // if (F->getIntrinsicID() == Intrinsic::copysign &&
  //     CI->getArgOperand(0) != val) {
  //   return true;
  // }

  // if (F->getIntrinsicID() == Intrinsic::memcpy && CI->getArgOperand(0) != val &&
  //     CI->getArgOperand(1) != val)
  //   return true;
  // if (F->getIntrinsicID() == Intrinsic::memmove &&
  //     CI->getArgOperand(0) != val && CI->getArgOperand(1) != val)
  //   return true;

  // only the float arg input is potentially active
  if (Name == "frexp" || Name == "frexpf" || Name == "frexpl") {
    return val != CI.getArgOperands()[0];
  }

  // The relerr argument is inactive
  if (Name == "Faddeeva_erf" || Name == "Faddeeva_erfc" ||
      Name == "Faddeeva_erfcx" || Name == "Faddeeva_erfi" ||
      Name == "Faddeeva_dawson") {
    for (size_t i = 0; i < CI.getArgOperands().size() - 1; i++)
    {
      if (val == CI.getArgOperands()[i])
        return false;
    }
    return true;
  }

  // only the buffer is active for mpi send/recv
  if (Name == "MPI_Recv" || Name == "PMPI_Recv" || Name == "MPI_Send" ||
      Name == "PMPI_Send") {
    return val != CI.getArgOperands()[0];
  }
  // only the recv buffer and request is active for mpi isend/irecv
  if (Name == "MPI_Irecv" || Name == "MPI_Isend") {
    return val != CI.getArgOperands()[0] && val != CI.getArgOperands()[6];
  }

  // only request is active
  if (Name == "MPI_Wait" || Name == "PMPI_Wait")
    return val != CI.getArgOperands()[0];

  if (Name == "MPI_Waitall" || Name == "PMPI_Waitall")
    return val != CI.getArgOperands()[1];

  // With all other options exhausted we have to assume this function could
  // actively use the value
  return false;
}

/// Call the function propagateFromOperand on all operands of CI
/// that could impact the activity of the call instruction
static inline void propagateArgumentInformation(
    /*TargetLibraryInfo &TLI,*/ CallOpInterface CI,
    std::function<bool(Value)> propagateFromOperand) {

  if (Operation *F = getFunctionFromCall(CI)) {
    // These functions are known to only have the first argument impact
    // the activity of the call instruction
    StringRef Name = cast<SymbolOpInterface>(F).getName();
    if (Name == "lgamma" || Name == "lgammaf" || Name == "lgammal" ||
        Name == "lgamma_r" || Name == "lgammaf_r" || Name == "lgammal_r" ||
        Name == "__lgamma_r_finite" || Name == "__lgammaf_r_finite" ||
        Name == "__lgammal_r_finite") {

      propagateFromOperand(CI.getArgOperands()[0]);
      return;
    }

    // Allocations, deallocations, and c++ guards are fully inactive
    // if (isAllocationFunction(Name, TLI) || isDeallocationFunction(Name, TLI) ||
    //     Name == "__cxa_guard_acquire" || Name == "__cxa_guard_release" ||
    //     Name == "__cxa_guard_abort")
    //   return;

    /// Only the first argument (magnitude) of copysign is active
    if (Name == llvm::Intrinsic::getName(llvm::Intrinsic::copysign)) {
      propagateFromOperand(CI.getArgOperands()[0]);
      return;
    }

    // Certain intrinsics are inactive by definition
    // and have nothing to propagate.
    for (unsigned intrinsicID : constantIntrinsics) {
      if (Name.startswith(llvm::Intrinsic::getBaseName(intrinsicID)))
        return;
    }

    if (Name.startswith(llvm::Intrinsic::getBaseName(llvm::Intrinsic::memcpy)) ||
        Name.startswith(llvm::Intrinsic::getBaseName(llvm::Intrinsic::memmove))) {
      propagateFromOperand(CI.getArgOperands()[0]);
      propagateFromOperand(CI.getArgOperands()[1]);
      return;
    }

    if (Name == "frexp" || Name == "frexpf" || Name == "frexpl") {
      propagateFromOperand(CI.getArgOperands()[0]);
      return;
    }
    if (Name == "Faddeeva_erf" || Name == "Faddeeva_erfc" ||
        Name == "Faddeeva_erfcx" || Name == "Faddeeva_erfi" ||
        Name == "Faddeeva_dawson") {
      for (size_t i = 0; i < CI.getArgOperands().size() - 1; i++) {
        propagateFromOperand(CI.getArgOperands()[i]);
      }
      return;
    }

    if (Name == "julia.call" || Name == "julia.call2") {
      for (size_t i = 1; i < CI.getArgOperands().size(); i++) {
        propagateFromOperand(CI.getArgOperands()[i]);
      }
      return;
    }
  }

  // For other calls, check all operands of the operation
  // as conservatively they may impact the activity of the call
  for (Value a : CI->getOperands()) {
    if (propagateFromOperand(a))
      break;
  }
}

/// Return whether this operation is known not to propagate adjoints
/// Note that operation could return an active pointer, but
/// do not propagate adjoints themselves
bool ActivityAnalyzer::isConstantOperation(TypeResults const &TR,
                                           Operation *I) {
  // This analysis may only be called by instructions corresponding to
  // the function analyzed by TypeInfo
  assert(I);
  // assert(TR.getFunction() == I->getParent()->getParent());

  // The return instruction doesn't impact activity (handled specifically
  // during adjoint generation)
  if (isa<func::ReturnOp>(I))
    return true;

  // Branch, unreachable, and previously computed constants are inactive
  if (isa<LLVM::UnreachableOp>(I) /*|| isa<cf::BranchOp>(I)*/ ||
      ConstantOperations.contains(I)) {
    return true;
  }

  /// Previously computed inactives remain inactive
  if (ActiveOperations.contains(I)) {
    return false;
  }

  if (notForAnalysis.count(I->getBlock())) {
    // if (EnzymePrintActivity)
    //   llvm::errs() << " constant instruction as dominates unreachable " << *I
    //                << "\n";
    InsertConstantOperation(TR, I);
    return true;
  }

  if (auto CI = dyn_cast<CallOpInterface>(I)) {
    if (CI->hasAttr("enzyme_active")) {
      // if (EnzymePrintActivity)
      //   llvm::errs() << "forced active " << *I << "\n";
      ActiveOperations.insert(I);
      return false;
    }
    if (CI->hasAttr("enzyme_inactive")) {
      // if (EnzymePrintActivity)
      //   llvm::errs() << "forced inactive " << *I << "\n";
      InsertConstantOperation(TR, I);
      return true;
    }
    Operation *called = getFunctionFromCall(CI);

    if (called) {
      if (called->hasAttr("enzyme_active")) {
        // if (EnzymePrintActivity)
        //   llvm::errs() << "forced active " << *I << "\n";
        ActiveOperations.insert(I);
        return false;
      }
      if (called->hasAttr("enzyme_inactive")) {
        // if (EnzymePrintActivity)
        //   llvm::errs() << "forced inactive " << *I << "\n";
        InsertConstantOperation(TR, I);
        return true;
      }
      if (KnownInactiveFunctionInsts.count(cast<SymbolOpInterface>(called).getName().str())) {
        InsertConstantOperation(TR, I);
        return true;
      }
    }
  }

  /// A store into all integral memory is inactive
  // TODO: per-operation stuff
  // if (auto SI = dyn_cast<StoreInst>(I)) {
  //   auto StoreSize = SI->getParent()
  //                        ->getParent()
  //                        ->getParent()
  //                        ->getDataLayout()
  //                        .getTypeSizeInBits(SI->getValueOperand()->getType()) /
  //                    8;

  //   bool AllIntegral = true;
  //   bool SeenInteger = false;
  //   auto q = TR.query(SI->getPointerOperand()).Data0();
  //   for (int i = -1; i < (int)StoreSize; ++i) {
  //     auto dt = q[{i}];
  //     if (dt.isIntegral() || dt == BaseType::Anything) {
  //       SeenInteger = true;
  //       if (i == -1)
  //         break;
  //     } else if (dt.isKnown()) {
  //       AllIntegral = false;
  //       break;
  //     }
  //   }

  //   if (AllIntegral && SeenInteger) {
  //     if (EnzymePrintActivity)
  //       llvm::errs() << " constant instruction from TA " << *I << "\n";
  //     InsertConstantInstruction(TR, I);
  //     return true;
  //   }
  // }
  // if (auto SI = dyn_cast<AtomicRMWInst>(I)) {
  //   auto StoreSize = SI->getParent()
  //                        ->getParent()
  //                        ->getParent()
  //                        ->getDataLayout()
  //                        .getTypeSizeInBits(I->getType()) /
  //                    8;

  //   bool AllIntegral = true;
  //   bool SeenInteger = false;
  //   auto q = TR.query(SI->getOperand(0)).Data0();
  //   for (int i = -1; i < (int)StoreSize; ++i) {
  //     auto dt = q[{i}];
  //     if (dt.isIntegral() || dt == BaseType::Anything) {
  //       SeenInteger = true;
  //       if (i == -1)
  //         break;
  //     } else if (dt.isKnown()) {
  //       AllIntegral = false;
  //       break;
  //     }
  //   }

  //   if (AllIntegral && SeenInteger) {
  //     if (EnzymePrintActivity)
  //       llvm::errs() << " constant instruction from TA " << *I << "\n";
  //     InsertConstantInstruction(TR, I);
  //     return true;
  //   }
  // }

  // if (EnzymePrintActivity)
  //   llvm::errs() << "checking if is constant[" << (int)directions << "] " << *I
  //                << "\n";

  if (isa<NVVM::Barrier0Op, LLVM::AssumeOp, LLVM::StackSaveOp,
          LLVM::StackRestoreOp, LLVM::LifetimeStartOp, LLVM::LifetimeEndOp,
          LLVM::Prefetch, LLVM::MemsetOp>(I)) {
    InsertConstantOperation(TR, I);
  }

//   if (auto II = dyn_cast<IntrinsicInst>(I)) {
//     switch (II->getIntrinsicID()) {
//     case Intrinsic::nvvm_barrier0:
//     case Intrinsic::nvvm_barrier0_popc:
//     case Intrinsic::nvvm_barrier0_and:
//     case Intrinsic::nvvm_barrier0_or:
//     case Intrinsic::nvvm_membar_cta:
//     case Intrinsic::nvvm_membar_gl:
//     case Intrinsic::nvvm_membar_sys:
//     case Intrinsic::amdgcn_s_barrier:
//     case Intrinsic::assume:
//     case Intrinsic::stacksave:
//     case Intrinsic::stackrestore:
//     case Intrinsic::lifetime_start:
//     case Intrinsic::lifetime_end:
//     case Intrinsic::dbg_addr:
//     case Intrinsic::dbg_declare:
//     case Intrinsic::dbg_value:
//     case Intrinsic::invariant_start:
//     case Intrinsic::invariant_end:
//     case Intrinsic::var_annotation:
//     case Intrinsic::ptr_annotation:
//     case Intrinsic::annotation:
//     case Intrinsic::codeview_annotation:
//     case Intrinsic::expect:
//     case Intrinsic::type_test:
//     case Intrinsic::donothing:
//     case Intrinsic::prefetch:
//     case Intrinsic::trap:
// #if LLVM_VERSION_MAJOR >= 8
//     case Intrinsic::is_constant:
// #endif
//     case Intrinsic::memset:
//       if (EnzymePrintActivity)
//         llvm::errs() << "known inactive intrinsic " << *I << "\n";
//       InsertConstantInstruction(TR, I);
//       return true;

//     default:
//       break;
//     }
//   }

  // Analyzer for inductive assumption where we attempt to prove this is
  // inactive from a lack of active users
  std::shared_ptr<ActivityAnalyzer> DownHypothesis;

  // If this instruction does not write to memory that outlives itself
  // (potentially propagating derivative information), the only way to propagate
  // derivative information is through the return value
  // TODO the "doesn't write to active memory" can be made more aggressive than
  // doesn't write to any memory
  bool noActiveWrite = false;

  if (isa<MemoryEffectOpInterface>(I) && !hasEffect<MemoryEffects::Write>(I))
    noActiveWrite = true;
  else if (auto CI = dyn_cast<CallOpInterface>(I)) {
    // if (AA.onlyReadsMemory(CI)) {
    //   noActiveWrite = true;
    // } else 
    if (Operation *F = getFunctionFromCall(CI)) {
      // if (isMemFreeLibMFunction(F->getName())) {
      //   noActiveWrite = true;
      // } else
      StringRef Name = cast<SymbolOpInterface>(F).getName();
      if (Name == "frexp" || Name == "frexpf" || Name == "frexpl") {
        noActiveWrite = true;
      }
    }
  }
  if (noActiveWrite) {
    // Even if returning a pointer, this instruction is considered inactive
    // since the instruction doesn't prop gradients. Thus, so long as we don't
    // return an object containing a float, this instruction is inactive
    // if (!TR.intType(1, I, /*errifNotFound*/ false).isPossibleFloat()) {
    //   if (EnzymePrintActivity)
    //     llvm::errs()
    //         << " constant instruction from known non-float non-writing "
    //            "instruction "
    //         << *I << "\n";
    //   InsertConstantInstruction(TR, I);
    //   return true;
    // }

    // If all returned values constant otherwise, the operation is inactive
    if (llvm::all_of(I->getResults(),
                     [&](Value v) { return isConstantValue(TR, v); })) {
      // if (EnzymePrintActivity)
      //   llvm::errs() << " constant instruction from known constant non-writing "
      //                   "instruction "
      //                << *I << "\n";
      InsertConstantOperation(TR, I);
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
      if (directions == DOWN /*&& !isa<PHINode>(I)*/) {
        if (llvm::all_of(I->getResults(), [&](Value val) {
              return isValueInactiveFromUsers(TR, val, UseActivity::None);
            })) {
          // if (EnzymePrintActivity)
          //   llvm::errs() << " constant instruction[" << (int)directions
          //                << "] from users instruction " << *I << "\n";
          InsertConstantOperation(TR, I);
          return true;
        }
      } else {
        DownHypothesis = std::shared_ptr<ActivityAnalyzer>(
            new ActivityAnalyzer(*this, DOWN));
        DownHypothesis->ConstantOperations.insert(I);
        if (llvm::all_of(I->getResults(), [&](Value val) {
              return DownHypothesis->isValueInactiveFromUsers(
                  TR, val, UseActivity::None);
            })) {
          // if (EnzymePrintActivity)
          //   llvm::errs() << " constant instruction[" << (int)directions
          //                << "] from users instruction " << *I << "\n";
          InsertConstantOperation(TR, I);
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
    UpHypothesis->ConstantOperations.insert(I);
    assert(directions & UP);
    if (UpHypothesis->isInstructionInactiveFromOrigin(TR, I)) {
      // if (EnzymePrintActivity)
      //   llvm::errs() << " constant instruction from origin "
      //                   "instruction "
      //                << *I << "\n";
      InsertConstantOperation(TR, I);
      insertConstantsFrom(TR, *UpHypothesis);
      if (DownHypothesis)
        insertConstantsFrom(TR, *DownHypothesis);
      return true;
    } else if (directions == (UP | DOWN)) {
      // TODO: what does this mean for interfaces?
      if (isa<
          // clang-format off
          LLVM::LoadOp,
          LLVM::StoreOp,
          // Integer binary ops.
          LLVM::AddOp,
          LLVM::SubOp,
          LLVM::MulOp,
          LLVM::UDivOp,
          LLVM::SDivOp,
          LLVM::URemOp,
          LLVM::SRemOp,
          LLVM::AndOp,
          LLVM::OrOp,
          LLVM::XOrOp,
          LLVM::ShlOp,
          LLVM::LShrOp,
          LLVM::AShrOp,
          // Float binary ops.
          LLVM::FAddOp,
          LLVM::FSubOp,
          LLVM::FMulOp,
          LLVM::FDivOp,
          LLVM::FRemOp,
          LLVM::FNegOp
          // clang-format on
            >(I)) {
        for (Value operand : I->getOperands()) {
          if (!UpHypothesis->isConstantValue(TR, operand)) {
            ReEvaluateOpIfInactiveValue[operand].insert(I);
          }
        }
      }
    }
  }

  // Otherwise we must fall back and assume this instruction to be active.
  ActiveOperations.insert(I);
  // if (EnzymePrintActivity)
  //   llvm::errs() << "couldnt decide fallback as nonconstant instruction("
  //                << (int)directions << "):" << *I << "\n";
  if (noActiveWrite && (directions == (UP | DOWN)))
    for (Value result : I->getResults())
      ReEvaluateOpIfInactiveValue[result].insert(I);
  return false;
}

static bool isValuePotentiallyUsedAsPointer(Value val) {
  std::deque<Value> todo = {val};
  SmallPtrSet<Value, 3> seen;
  while (todo.size()) {
    auto cur = todo.back();
    todo.pop_back();
    if (seen.count(cur))
      continue;
    seen.insert(cur);
    for (Operation *user : cur.getUsers()) {
      if (isa<func::ReturnOp>(user))
        return true;
      // The operation is known not to read or write memory.
      if (isa<MemoryEffectOpInterface>(user) &&
          !hasEffect<MemoryEffects::Read>(user) &&
          !hasEffect<MemoryEffects::Write>(user)) {
        todo.insert(todo.end(), user->result_begin(), user->result_end());
        continue;
      }
      // if (EnzymePrintActivity)
      //   llvm::errs() << " VALUE potentially used as pointer " << *val << " by "
      //                << *u << "\n";
      return true;
    }
  }
  return false;
}

bool ActivityAnalyzer::isConstantValue(TypeResults const &TR, Value Val) {
  // This analysis may only be called by instructions corresponding to
  // the function analyzed by TypeInfo -- however if the Value
  // was created outside a function (e.g. global, constant), that is allowed
  assert(Val);
  // if (auto I = dyn_cast<Instruction>(Val)) {
  //   if (TR.getFunction() != I->getParent()->getParent()) {
  //     llvm::errs() << *TR.getFunction() << "\n";
  //     llvm::errs() << *I << "\n";
  //   }
  //   assert(TR.getFunction() == I->getParent()->getParent());
  // }
  // if (auto Arg = dyn_cast<Argument>(Val)) {
  //   assert(TR.getFunction() == Arg->getParent());
  // }

  // Void values are definitionally inactive
  if (Val.getType().isa<LLVM::LLVMVoidType>())
    return true;

  // Token values are definitionally inactive
  if (Val.getType().isa<LLVM::LLVMTokenType>())
    return true;

  // All function pointers are considered active in case an augmented primal
  // or reverse is needed
  if (Val.getDefiningOp() &&
      isa<func::ConstantOp, LLVM::InlineAsmOp>(Val.getDefiningOp())) {
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

  // TODO: LLVM global initializers with regions?
  if (matchPattern(Val, m_Constant()))
    return true;

  // if (auto CD = dyn_cast<ConstantDataSequential>(Val)) {
  //   // inductively assume inactive
  //   ConstantValues.insert(CD);
  //   for (size_t i = 0, len = CD->getNumElements(); i < len; i++) {
  //     if (!isConstantValue(TR, CD->getElementAsConstant(i))) {
  //       ConstantValues.erase(CD);
  //       ActiveValues.insert(CD);
  //       return false;
  //     }
  //   }
  //   return true;
  // }
  // if (auto CD = dyn_cast<ConstantAggregate>(Val)) {
  //   // inductively assume inactive
  //   ConstantValues.insert(CD);
  //   for (size_t i = 0, len = CD->getNumOperands(); i < len; i++) {
  //     if (!isConstantValue(TR, CD->getOperand(i))) {
  //       ConstantValues.erase(CD);
  //       ActiveValues.insert(CD);
  //       return false;
  //     }
  //   }
  //   return true;
  // }

  if (Operation *definingOp = Val.getDefiningOp()) {
    // Undef, metadata, non-global constants are inactive.
    if (isa<LLVM::UndefOp, LLVM::MetadataOp, LLVM::ConstantOp>(definingOp)) {
      return true;
    }

    // Ops derived from intrinsics.
    // NOTE: this was written with the assumption that Value is-a Operation,
    // which is not the case in MLIR.
    if (isa<NVVM::Barrier0Op, LLVM::AssumeOp, LLVM::StackSaveOp,
            LLVM::StackRestoreOp, LLVM::LifetimeStartOp, LLVM::LifetimeEndOp,
            LLVM::Prefetch>(definingOp)) {
      return true;
    }
  }

  if (auto arg = Val.dyn_cast<BlockArgument>()) {
    auto funcIface = dyn_cast_or_null<FunctionOpInterface>(
        arg.getParentBlock()->getParentOp());
    if (!funcIface || !arg.getOwner()->isEntryBlock()) {
      // TODO: we want a more advanced analysis based on MLIR interfaces here
      // For now, conservatively assume all block arguments are active
      return false;

      //
      // The code below is incomplete. We want to check all predecessors and,
      // additionally, if the owner block is listed as as successor of a given
      // predecessor more than once. Only if the value is constant in all cases
      // should it be deemed constant here. Some mixed evaluation using dataflow
      // analysis for constant propagation (similar to SCCP) and for activity
      // analysis may be more precise.
      //
      // for (Block *predecessor : arg.getOwner()->getPredecessors()) {
      //   if (auto branch =
      //           dyn_cast<BranchOpInterface>(predecessor->getTerminator())) {
      //     auto it = llvm::find(predecessor->getSuccessors(), arg.getOwner());
      //     unsigned successorNo = std::distance(predecessor->succ_begin(),
      //     it); SuccessorOperands successorOperands =
      //     branch.getSuccessorOperands(successorNo);
      //     // If the argument is forwarded, this will be non-empty.
      //     if (Value passedOperand = successorOperands[arg.getArgNumber()]) {
      //       // TODO: we must avoid infinite recursion here...
      //       return isConstantValue(TR, passedOperand);
      //     }
      //   }
      // }
    }

    // All arguments must be marked constant/nonconstant ahead of time
    if (!funcIface.getArgAttr(arg.getArgNumber(),
                              LLVM::LLVMDialect::getByValAttrName())) {
      llvm::errs() << funcIface << "\n";
      llvm::errs() << Val << "\n";
      assert(0 && "must've put arguments in constant/nonconstant");
    }
  }

  // This value is certainly an integer (and only and integer, not a pointer or
  // float). Therefore its value is constant
  // if (TR.intType(1, Val, /*errIfNotFound*/ false).isIntegral()) {
  //   if (EnzymePrintActivity)
  //     llvm::errs() << " Value const as integral " << (int)directions << " "
  //                  << *Val << " "
  //                  << TR.intType(1, Val, /*errIfNotFound*/ false).str() << "\n";
  //   InsertConstantValue(TR, Val);
  //   return true;
  // }

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

  // TODO: since in MLIR globals are operations like others, this should be
  // handled via interfaces.
  //
  // if (auto GI = dyn_cast<GlobalVariable>(Val)) {
  //   // If operating under the assumption globals are inactive unless
  //   // explicitly marked as active, this is inactive
  //   if (!hasMetadata(GI, "enzyme_shadow") && EnzymeNonmarkedGlobalsInactive)
  //   {
  //     InsertConstantValue(TR, Val);
  //     return true;
  //   }

  //   if (GI->getName().contains("enzyme_const") ||
  //       InactiveGlobals.count(GI->getName().str())) {
  //     InsertConstantValue(TR, Val);
  //     return true;
  //   }

  //   // If this global is unchanging and the internal constant data
  //   // is inactive, the global is inactive
  //   if (GI->isConstant() && GI->hasInitializer() &&
  //       isConstantValue(TR, GI->getInitializer())) {
  //     InsertConstantValue(TR, Val);
  //     if (EnzymePrintActivity)
  //       llvm::errs() << " VALUE const global " << *Val
  //                    << " init: " << *GI->getInitializer() << "\n";
  //     return true;
  //   }

  //   // If this global is a pointer to an integer, it is inactive
  //   // TODO note this may need updating to consider the size
  //   // of the global
  //   auto res = TR.query(GI).Data0();
  //   auto dt = res[{-1}];
  //   if (dt.isIntegral()) {
  //     if (EnzymePrintActivity)
  //       llvm::errs() << " VALUE const as global int pointer " << *Val
  //                    << " type - " << res.str() << "\n";
  //     InsertConstantValue(TR, Val);
  //     return true;
  //   }

  //   // If this is a global local to this translation unit with inactive
  //   // initializer and no active uses, it is definitionally inactive
  //   bool usedJustInThisModule =
  //       GI->hasInternalLinkage() || GI->hasPrivateLinkage();

  //   if (EnzymePrintActivity)
  //     llvm::errs() << "pre attempting(" << (int)directions
  //                  << ") just used in module for: " << *GI << " dir"
  //                  << (int)directions << " justusedin:" << usedJustInThisModule
  //                  << "\n";

  //   if (directions == 3 && usedJustInThisModule) {
  //     // TODO this assumes global initializer cannot refer to itself (lest
  //     // infinite loop)
  //     if (!GI->hasInitializer() || isConstantValue(TR, GI->getInitializer())) {

  //       if (EnzymePrintActivity)
  //         llvm::errs() << "attempting just used in module for: " << *GI << "\n";
  //       // Not looking at users to prove inactive (definition of down)
  //       // If all users are inactive, this is therefore inactive.
  //       // Since we won't look at origins to prove, we can inductively assume
  //       // this is inactive

  //       // As an optimization if we are going down already
  //       // and we won't use ourselves (done by PHI's), we
  //       // dont need to inductively assume we're true
  //       // and can instead use this object!
  //       // This pointer is inactive if it is either not actively stored to or
  //       // not actively loaded from
  //       // See alloca logic to explain why OnlyStores is insufficient here
  //       if (directions == DOWN) {
  //         if (isValueInactiveFromUsers(TR, Val, UseActivity::OnlyLoads)) {
  //           InsertConstantValue(TR, Val);
  //           return true;
  //         }
  //       } else {
  //         Instruction *LoadReval = nullptr;
  //         Instruction *StoreReval = nullptr;
  //         auto DownHypothesis = std::shared_ptr<ActivityAnalyzer>(
  //             new ActivityAnalyzer(*this, DOWN));
  //         DownHypothesis->ConstantValues.insert(Val);
  //         if (DownHypothesis->isValueInactiveFromUsers(
  //                 TR, Val, UseActivity::OnlyLoads, &LoadReval) ||
  //             (TR.query(GI)[{-1, -1}].isFloat() &&
  //              DownHypothesis->isValueInactiveFromUsers(
  //                  TR, Val, UseActivity::OnlyStores, &StoreReval))) {
  //           insertConstantsFrom(TR, *DownHypothesis);
  //           InsertConstantValue(TR, Val);
  //           return true;
  //         } else {
  //           if (LoadReval) {
  //             if (EnzymePrintActivity)
  //               llvm::errs() << " global activity of " << *Val
  //                            << " dependant on " << *LoadReval << "\n";
  //             ReEvaluateValueIfInactiveInst[LoadReval].insert(Val);
  //           }
  //           if (StoreReval)
  //             ReEvaluateValueIfInactiveInst[StoreReval].insert(Val);
  //         }
  //       }
  //     }
  //   }

  //   // Otherwise we have to assume this global is active since it can
  //   // be arbitrarily used in an active way
  //   // TODO we can be more aggressive here in the future
  //   if (EnzymePrintActivity)
  //     llvm::errs() << " VALUE nonconst unknown global " << *Val << " type - "
  //                  << res.str() << "\n";
  //   ActiveValues.insert(Val);
  //   return false;
  // }

  //
  // TODO: constants are just operations in MLIR, this should be handled via
  // interfaces.
  //
  // ConstantExpr's are inactive if their arguments are inactive
  // Note that since there can't be a recursive constant this shouldn't
  // infinite loop
  // if (auto ce = dyn_cast<ConstantExpr>(Val)) {
  //   if (ce->isCast()) {
  //     if (isConstantValue(TR, ce->getOperand(0))) {
  //       if (EnzymePrintActivity)
  //         llvm::errs() << " VALUE const cast from from operand " << *Val
  //                      << "\n";
  //       InsertConstantValue(TR, Val);
  //       return true;
  //     }
  //   }
  //   if (ce->getOpcode() == Instruction::GetElementPtr &&
  //       llvm::all_of(ce->operand_values(),
  //                    [&](Value *v) { return isConstantValue(TR, v); })) {
  //     if (isConstantValue(TR, ce->getOperand(0))) {
  //       if (EnzymePrintActivity)
  //         llvm::errs() << " VALUE const cast from gep operand " << *Val <<
  //         "\n";
  //       InsertConstantValue(TR, Val);
  //       return true;
  //     }
  //   }
  //   if (EnzymePrintActivity)
  //     llvm::errs() << " VALUE nonconst unknown expr " << *Val << "\n";
  //   ActiveValues.insert(Val);
  //   return false;
  // }

  if (auto CI = Val.getDefiningOp<CallOpInterface>()) {
    
    if (CI->hasAttr("enzyme_active")) {
      // if (EnzymePrintActivity)
      //   llvm::errs() << "forced active val " << *Val << "\n";
      ActiveValues.insert(Val);
      return false;
    }
    if (CI->hasAttr("enzyme_inactive")) {
      // if (EnzymePrintActivity)
      //   llvm::errs() << "forced inactive val " << *Val << "\n";
      InsertConstantValue(TR, Val);
      return true;
    }
    Operation *called = getFunctionFromCall(CI);
    if (called) {
      if (called->hasAttr("enzyme_active")) {
        // if (EnzymePrintActivity)
        //   llvm::errs() << "forced active val " << *Val << "\n";
        ActiveValues.insert(Val);
        return false;
      }
      if (called->hasAttr("enzyme_inactive")) {
        // if (EnzymePrintActivity)
        //   llvm::errs() << "forced inactive val " << *Val << "\n";
        InsertConstantValue(TR, Val);
        return true;
      }
    }
  }

  std::shared_ptr<ActivityAnalyzer> UpHypothesis;

  // Handle types that could contain pointers
  //  Consider all types except
  //   * floating point types (since those are assumed not pointers)
  //   * integers that we know are not pointers
  //
  // TODO: this needs to go through type interfaces.
  //
  bool containsPointer = true;
  Type vectorTypeOrSelf = LLVM::isCompatibleVectorType(Val.getType())
                              ? LLVM::getVectorElementType(Val.getType())
                              : Val.getType();
  if (LLVM::isCompatibleFloatingPointType(vectorTypeOrSelf))
    containsPointer = false;
  // if (!TR.intType(1, Val, /*errIfNotFound*/ false).isPossiblePointer())
  if (!isa<LLVM::LLVMPointerType, MemRefType>(Val.getType()))
    containsPointer = false;

  if (containsPointer && !isValuePotentiallyUsedAsPointer(Val)) {
    containsPointer = false;
  }

  //
  // TODO: support pointers; this will likely need type analysis, a better
  // aliasing analysis than what MLIR currently offers, and some generalization
  // across types to understand what a pointer in MLIR type system.
  //
#if 0
  if (containsPointer) {

    auto TmpOrig =
#if LLVM_VERSION_MAJOR >= 12
        getUnderlyingObject(Val, 100);
#else
        GetUnderlyingObject(Val, TR.getFunction()->getParent()->getDataLayout(),
                            100);
#endif

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
      } else if (auto op = dyn_cast<CallInst>(TmpOrig)) {
        if (op->hasFnAttr("enzyme_inactive")) {
          InsertConstantValue(TR, Val);
          insertConstantsFrom(TR, *UpHypothesis);
          return true;
        }
        Function *called = getFunctionFromCall(op);

        StringRef funcName = getFuncNameFromCall(op);

        if (called && called->hasFnAttribute("enzyme_inactive")) {
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

#if LLVM_VERSION_MAJOR >= 9
        auto dName = demangle(funcName.str());
        for (auto FuncName : DemangledKnownInactiveFunctionsStartingWith) {
          if (StringRef(dName).startswith(FuncName)) {
            InsertConstantValue(TR, Val);
            insertConstantsFrom(TR, *UpHypothesis);
            return true;
          }
        }
#endif

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

        if (KnownInactiveFunctions.count(funcName.str()) ||
            MPIInactiveCommAllocators.find(funcName.str()) !=
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
        // alloca OnlyStores is insufficient here since the loaded pointer can
        // have active memory stored into it [e.g. not just top level pointer
        // that matters]
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
            if (isInstructionInactiveFromOrigin(TR, inst)) {
              InsertConstantValue(TR, Val);
              return true;
            }
          } else {
            if (UpHypothesis->isInstructionInactiveFromOrigin(TR, inst)) {
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
    bool potentiallyActiveStore = false;
    bool potentialStore = false;
    bool potentiallyActiveLoad = false;

    // Assume the value (not instruction) is itself active
    // In spite of that can we show that there are either no active stores
    // or no active loads
    std::shared_ptr<ActivityAnalyzer> Hypothesis =
        std::shared_ptr<ActivityAnalyzer>(
            new ActivityAnalyzer(*this, directions));
    Hypothesis->ActiveValues.insert(Val);
    if (auto VI = dyn_cast<Instruction>(Val)) {
      for (auto V : DeducingPointers) {
        // UpHypothesis->InsertConstantValue(TR, V);
      }
      if (UpHypothesis->isInstructionInactiveFromOrigin(TR, VI)) {
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

      // If this is a malloc or free, this doesn't impact the activity
      if (auto CI = dyn_cast<CallInst>(I)) {
        if (CI->hasFnAttr("enzyme_inactive"))
          return false;

#if LLVM_VERSION_MAJOR >= 11
        if (auto iasm = dyn_cast<InlineAsm>(CI->getCalledOperand()))
#else
        if (auto iasm = dyn_cast<InlineAsm>(CI->getCalledValue()))
#endif
        {
          if (StringRef(iasm->getAsmString()).contains("exit") ||
              StringRef(iasm->getAsmString()).contains("cpuid"))
            return false;
        }

        Function *F = getFunctionFromCall(CI);
        StringRef funcName = getFuncNameFromCall(CI);

        if (F && F->hasFnAttribute("enzyme_inactive")) {
          return false;
        }
        if (isAllocationFunction(funcName, TLI) ||
            isDeallocationFunction(funcName, TLI)) {
          return false;
        }
        if (KnownInactiveFunctions.count(funcName.str()) ||
            MPIInactiveCommAllocators.find(funcName.str()) !=
                MPIInactiveCommAllocators.end()) {
          return false;
        }
        if (KnownInactiveFunctionInsts.count(funcName.str())) {
          return false;
        }
        if (isMemFreeLibMFunction(funcName) || funcName == "__fd_sincos_1") {
          return false;
        }
#if LLVM_VERSION_MAJOR >= 9
        auto dName = demangle(funcName.str());
        for (auto FuncName : DemangledKnownInactiveFunctionsStartingWith) {
          if (StringRef(dName).startswith(FuncName)) {
            return false;
          }
        }
#endif
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
            funcName == "__cxa_guard_abort" || funcName == "posix_memalign") {
          return false;
        }

        if (F) {
          switch (F->getIntrinsicID()) {
          case Intrinsic::nvvm_barrier0:
          case Intrinsic::nvvm_barrier0_popc:
          case Intrinsic::nvvm_barrier0_and:
          case Intrinsic::nvvm_barrier0_or:
          case Intrinsic::nvvm_membar_cta:
          case Intrinsic::nvvm_membar_gl:
          case Intrinsic::nvvm_membar_sys:
          case Intrinsic::amdgcn_s_barrier:
          case Intrinsic::assume:
          case Intrinsic::stacksave:
          case Intrinsic::stackrestore:
          case Intrinsic::lifetime_start:
          case Intrinsic::lifetime_end:
          case Intrinsic::dbg_addr:
          case Intrinsic::dbg_declare:
          case Intrinsic::dbg_value:
          case Intrinsic::invariant_start:
          case Intrinsic::invariant_end:
          case Intrinsic::var_annotation:
          case Intrinsic::ptr_annotation:
          case Intrinsic::annotation:
          case Intrinsic::codeview_annotation:
          case Intrinsic::expect:
          case Intrinsic::type_test:
          case Intrinsic::donothing:
          case Intrinsic::prefetch:
          case Intrinsic::trap:
#if LLVM_VERSION_MAJOR >= 8
          case Intrinsic::is_constant:
#endif
            return false;
          default:
            break;
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
#elif LLVM_VERSION_MAJOR >= 9
      auto AARes =
          AA.getModRefInfo(I, MemoryLocation(memval, LocationSize::unknown()));
#else
      auto AARes = AA.getModRefInfo(
          I, MemoryLocation(memval, MemoryLocation::UnknownSize));
#endif

      // Still having failed to replace the location used by AA, fall back to
      // getModref against any location.
      if (!memval->getType()->isPointerTy()) {
        if (auto CB = dyn_cast<CallInst>(I)) {
#if LLVM_VERSION_MAJOR >= 16
          AARes = AA.getModRefBehavior(CB).getModRef();
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
      }

      // TODO this aliasing information is too conservative, the question
      // isn't merely aliasing but whether there is a path for THIS value to
      // eventually be loaded by it not simply because there isnt aliasing

      // If we haven't already shown a potentially active load
      // check if this loads the given value and is active
      if (!potentiallyActiveLoad && isRefSet(AARes)) {
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
            potentiallyActiveLoad = true;
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
                          potentiallyActiveStore = true;
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
            potentiallyActiveLoad = true;
            if (TR.query(Val)[{-1, -1}].isPossiblePointer()) {
              if (EnzymePrintActivity)
                llvm::errs()
                    << "potential active store via pointer in memcpy: " << *I
                    << " of " << *Val << "\n";
              potentiallyActiveStore = true;
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
            potentiallyActiveLoad = true;
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
                potentiallyActiveStore = true;
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
            potentiallyActiveStore = true;
        } else if (auto MTI = dyn_cast<MemTransferInst>(I)) {
          bool cop = !Hypothesis->isConstantValue(TR, MTI->getArgOperand(1));
          potentialStore = true;
          if (cop)
            potentiallyActiveStore = true;
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
            potentiallyActiveStore = true;
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
      insertAllFrom(TR, *Hypothesis, Val);
      // TODO have insertall dependence on this
      if (TmpOrig != Val)
        ReEvaluateValueIfInactiveValue[TmpOrig].insert(Val);
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
      for (auto V : DeducingPointers) {
        // UpHypothesis->InsertConstantValue(TR, V);
      }
      assert(directions & UP);
      bool ActiveUp = !isa<Argument>(Val) &&
                      !UpHypothesis->isInstructionInactiveFromOrigin(TR, Val);

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
        insertAllFrom(TR, *Hypothesis, Val);
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
  #endif
  // End TODO support pointers

  // For all non-pointers, it is now sufficient to simply prove that
  // either activity does not flow in, or activity does not flow out
  // This alone cuts off the flow (being unable to flow through memory)

  // Not looking at uses to prove inactive (definition of up), if the creator of
  // this value is inactive, we are inactive Since we won't look at uses to
  // prove, we can inductively assume this is inactive
  if (directions & UP) {
    if (directions == UP && !Val.isa<BlockArgument>()) {
      if (isInstructionInactiveFromOrigin(TR, Val.getDefiningOp())) {
        InsertConstantValue(TR, Val);
        return true;
      } else if (Operation *op = Val.getDefiningOp()) {
        if (directions == (UP | DOWN)) {
          for (Value operand : op->getOperands()) {
            if (!UpHypothesis->isConstantValue(TR, operand)) {
              for (Value result : op->getResults()) {
                ReEvaluateValueIfInactiveValue[operand].insert(result);
              }
            }
          }
        }
      }
    } else {
      UpHypothesis =
          std::shared_ptr<ActivityAnalyzer>(new ActivityAnalyzer(*this, UP));
      UpHypothesis->ConstantValues.insert(Val);
      if (UpHypothesis->isInstructionInactiveFromOrigin(TR, Val.getDefiningOp())) {
        insertConstantsFrom(TR, *UpHypothesis);
        InsertConstantValue(TR, Val);
        return true;
      } else if (Operation *op = Val.getDefiningOp()) {
        if (directions == (UP | DOWN)) {
          for (Value operand : op->getOperands()) {
            if (!UpHypothesis->isConstantValue(TR, operand)) {
              for (Value result : op->getResults()) {
                ReEvaluateValueIfInactiveValue[operand].insert(result);
              }
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
    if (directions == DOWN && !Val.isa<BlockArgument>()) {
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

  // if (EnzymePrintActivity)
  //   llvm::errs() << " Value nonconstant (couldn't disprove)[" << (int)directions
  //                << "]" << *Val << "\n";
  ActiveValues.insert(Val);
  return false;
}

/// Is the instruction guaranteed to be inactive because of its operands
bool ActivityAnalyzer::isInstructionInactiveFromOrigin(TypeResults const &TR,
                                                       Operation *op /*llvm::Value *val*/) {
  // Must be an analyzer only searching up
  assert(directions == UP);

  // Not an instruction and thus not legal to search for activity via operands
  if (!op) {
    llvm::errs() << "unknown pointer source";
    assert(0 && "unknown pointer source");
    llvm_unreachable("unknown pointer source");
    return false;
  }

  // if (EnzymePrintActivity)
  //   llvm::errs() << " < UPSEARCH" << (int)directions << ">" << *inst << "\n";

  // cpuid is explicitly an inactive instruction
  if (auto iasm = dyn_cast<LLVM::InlineAsmOp>(op)) {
    if (iasm.getAsmString().contains("cpuid")) {
      // if (EnzymePrintActivity)
      //   llvm::errs() << " constant instruction from known cpuid instruction "
      //                << *inst << "\n";
      return true;
    }
  }

  if (auto store = dyn_cast<LLVM::StoreOp>(op)) {
    if (isConstantValue(TR, store.getValue()) ||
        isConstantValue(TR, store.getAddr())) {
      // if (EnzymePrintActivity)
      //   llvm::errs() << " constant instruction as store operand is inactive "
      //                << *inst << "\n";
      return true;
    }
  }

  if (isa<LLVM::MemcpyOp, LLVM::MemmoveOp>(op)) {
    // if either src or dst is inactive, there cannot be a transfer of active
    // values and thus the store is inactive
    if (isConstantValue(TR, op->getOperand(0)) ||
        isConstantValue(TR, op->getOperand(1))) {
      // if (EnzymePrintActivity)
      //   llvm::errs() << " constant instruction as memtransfer " << *inst
      //                << "\n";
      return true;
    }
  }

  if (auto call = dyn_cast<CallOpInterface>(op)) {
    if (op->hasAttr("enzyme_inactive")) {
      return true;
    }
    // Calls to print/assert/cxa guard are definitionally inactive
    Operation *called = getFunctionFromCall(call);
    StringRef funcName = called ? cast<SymbolOpInterface>(called).getName() : "";

    if (called && called->hasAttr("enzyme_inactive")) {
      return true;
    }
    if (funcName == "free" || funcName == "_ZdlPv" || funcName == "_ZdlPvm" ||
        funcName == "munmap") {
      return true;
    }

#if LLVM_VERSION_MAJOR >= 9
    auto dName = llvm::demangle(funcName.str());
    for (auto FuncName : DemangledKnownInactiveFunctionsStartingWith) {
      if (StringRef(dName).startswith(FuncName)) {
        return true;
      }
    }
#endif

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

    if (KnownInactiveFunctions.count(funcName.str()) ||
        MPIInactiveCommAllocators.find(funcName.str()) !=
            MPIInactiveCommAllocators.end()) {
      // if (EnzymePrintActivity)
      //   llvm::errs() << "constant(" << (int)directions
      //                << ") up-knowninactivecall " << *inst << "\n";
      return true;
    }

    //
    // TODO support this as a pass/configuration option.
    //
    // If requesting empty unknown functions to be considered inactive, abide
    // by those rules
    // if (called && EnzymeEmptyFnInactive && called->empty() &&
    //     !hasMetadata(called, "enzyme_gradient") &&
    //     !hasMetadata(called, "enzyme_derivative") &&
    //     !isAllocationFunction(funcName, TLI) &&
    //     !isDeallocationFunction(funcName, TLI) && !isa<IntrinsicInst>(op)) {
    //   if (EnzymePrintActivity)
    //     llvm::errs() << "constant(" << (int)directions << ") up-emptyconst "
    //                  << *inst << "\n";
    //   return true;
    // }
    Value callVal = call.getCallableForCallee().dyn_cast<Value>();
    if (isConstantValue(TR, callVal)) {
      // if (EnzymePrintActivity)
      //   llvm::errs() << "constant(" << (int)directions << ") up-constfn "
      //                << *inst << " - " << *callVal << "\n";
      return true;
    }
  }
  // Intrinsics known always to be inactive
  if (isa<NVVM::Barrier0Op, LLVM::AssumeOp, LLVM::StackSaveOp,
          LLVM::StackRestoreOp, LLVM::LifetimeStartOp, LLVM::LifetimeEndOp,
          LLVM::Prefetch, LLVM::MemsetOp>(op)) {
    // if (EnzymePrintActivity)
    //   llvm::errs() << "constant(" << (int)directions << ") up-intrinsic "
    //                << *inst << "\n";
    return true;
  }

  if (auto gep = dyn_cast<LLVM::GEPOp>(op)) {
    // A gep's only args that could make it active is the pointer operand
    if (isConstantValue(TR, gep.getBase())) {
      // if (EnzymePrintActivity)
      //   llvm::errs() << "constant(" << (int)directions << ") up-gep " << *inst
      //                << "\n";
      return true;
    }
    return false;
  }

  //
  // TODO: better support for known function calls. Ideally, they should become
  // operations, but we also need parity with LLVM-enzyme.
  //
  // if (auto ci = dyn_cast<CallInst>(inst)) {
  //   bool seenuse = false;

  //   propagateArgumentInformation(TLI, *ci, [&](Value *a) {
  //     if (!isConstantValue(TR, a)) {
  //       seenuse = true;
  //       if (EnzymePrintActivity)
  //         llvm::errs() << "nonconstant(" << (int)directions << ")  up-call "
  //                      << *inst << " op " << *a << "\n";
  //       return true;
  //     }
  //     return false;
  //   });
  //   if (EnzymeGlobalActivity) {
  //     if (!ci->onlyAccessesArgMemory() && !ci->doesNotAccessMemory()) {
  //       bool legalUse = false;

  //       StringRef funcName = getFuncNameFromCall(ci);

  //       if (funcName == "") {
  //       } else if (isMemFreeLibMFunction(funcName) ||
  //                  isDebugFunction(ci->getCalledFunction()) ||
  //                  isCertainPrint(funcName) ||
  //                  isAllocationFunction(funcName, TLI) ||
  //                  isDeallocationFunction(funcName, TLI)) {
  //         legalUse = true;
  //       }

  //       if (!legalUse) {
  //         if (EnzymePrintActivity)
  //           llvm::errs() << "nonconstant(" << (int)directions << ")  up-global "
  //                        << *inst << "\n";
  //         seenuse = true;
  //       }
  //     }
  //   }

  //   if (!seenuse) {
  //     if (EnzymePrintActivity)
  //       llvm::errs() << "constant(" << (int)directions << ")  up-call:" << *inst
  //                    << "\n";
  //     return true;
  //   }
  //   return !seenuse;
  // } 

  if (auto si = dyn_cast<LLVM::SelectOp>(op)) {

    if (isConstantValue(TR, si.getTrueValue()) &&
        isConstantValue(TR, si.getFalseValue())) {

      // if (EnzymePrintActivity)
      //   llvm::errs() << "constant(" << (int)directions << ") up-sel:" << *inst
      //                << "\n";
      return true;
    }
    return false;
  }
  
  if (isa<LLVM::SIToFPOp, LLVM::UIToFPOp, LLVM::FPToSIOp, LLVM::FPToUIOp>(op)) {
    // if (EnzymePrintActivity)
    //   llvm::errs() << "constant(" << (int)directions << ") up-fpcst:" << *inst
    //                << "\n";
    return true;
  } else if (op->getNumRegions() != 0) {
    //
    // TODO: MLIR ops have regions, which may result to activity flow into op
    // results, conservatively assume the op is active until we handle those
    // properly.
    //
    return false;
  } else {
    bool seenuse = false;
    //! TODO does not consider reading from global memory that is active and not
    //! an argument
    for (Value a : op->getOperands()) {
      bool hypval = isConstantValue(TR, a);
      if (!hypval) {
        // if (EnzymePrintActivity)
        //   llvm::errs() << "nonconstant(" << (int)directions << ")  up-inst "
        //                << *inst << " op " << *a << "\n";
        seenuse = true;
        break;
      }
    }

    if (!seenuse) {
      // if (EnzymePrintActivity)
      //   llvm::errs() << "constant(" << (int)directions << ")  up-inst:" << *inst
      //                << "\n";
      return true;
    }
    return false;
  }
}

/// Is the value free of any active uses
bool ActivityAnalyzer::isValueInactiveFromUsers(TypeResults const &TR,
                                                Value val,
                                                UseActivity PUA,
                                                Operation **FoundInst) {
  assert(directions & DOWN);
  // Must be an analyzer only searching down, unless used outside
  // assert(directions == DOWN);

  // To ensure we can call down

  // if (EnzymePrintActivity)
  //   llvm::errs() << " <Value USESEARCH" << (int)directions << ">" << *val
  //                << " UA=" << (int)PUA << "\n";

  bool seenuse = false;
  // user, predecessor
  std::deque<std::tuple<Operation *, Value, UseActivity>> todo;
  for (Operation *a : val.getUsers()) {
    todo.push_back(std::make_tuple(a, val, PUA));
  }
  std::set<std::tuple<Operation *, Value, UseActivity>> done = {};

  llvm::SmallPtrSet<Value, 1> AllocaSet;

  if (val.getDefiningOp<LLVM::AllocaOp>())
    AllocaSet.insert(val);

  // if (PUA == UseActivity::None && isAllocationCall(val, TLI))
  //   AllocaSet.insert(val);

  while (todo.size()) {
    Operation *a;
    Value parent;
    UseActivity UA;
    auto tuple = todo.front();
    todo.pop_front();
    if (done.count(tuple))
      continue;
    done.insert(tuple);
    std::tie(a, parent, UA) = tuple;

    if (auto LI = dyn_cast<LLVM::LoadOp>(a)) {
      if (UA == UseActivity::OnlyStores)
        continue;
      // if (UA == UseActivity::OnlyNonPointerStores ||
      //     UA == UseActivity::AllStores) {
      //   if (!TR.query(LI)[{-1}].isPossiblePointer())
      //     continue;
      // }
    }

    // if (EnzymePrintActivity)
    //   llvm::errs() << "      considering use of " << *val << " - " << *a
    //                << "\n";

    // Only ignore stores to the operand, not storing the operand
    // somewhere
    if (auto SI = dyn_cast<LLVM::StoreOp>(a)) {
      if (SI.getValue() != parent) {
        if (UA == UseActivity::OnlyLoads) {
          continue;
        }
        if (UA != UseActivity::AllStores &&
            (ConstantValues.count(SI.getValue()) ||
             // FIXME: this was a llvm::ConstantInt, should not be a hardcoded
             // assumption that ints are not active.
             (SI.getValue().getDefiningOp<LLVM::ConstantOp>() &&
              SI.getValue().getType().isa<IntegerType>())))
          continue;
        if (UA == UseActivity::None) {
          // If storing into itself, all potential uses are taken care of
          // elsewhere in the recursion.
          bool shouldContinue = true;
          SmallVector<Value, 1> vtodo = {SI.getValue()};
          llvm::SmallPtrSet<Value, 1> seen;
          llvm::SmallPtrSet<Value, 1> newAllocaSet;
          while (vtodo.size()) {
            auto TmpOrig = vtodo.back();
            vtodo.pop_back();
            if (seen.count(TmpOrig))
              continue;
            seen.insert(TmpOrig);
            if (AllocaSet.count(TmpOrig)) {
              continue;
            }
            if (TmpOrig.getDefiningOp<LLVM::AllocaOp>()) {
              newAllocaSet.insert(TmpOrig);
              continue;
            }
            // if (isAllocationCall(TmpOrig, TLI)) {
            //   newAllocaSet.insert(TmpOrig);
            //   continue;
            // }
            if (isa_and_nonnull<LLVM::UndefOp, LLVM::ConstantOp>(
                    TmpOrig.getDefiningOp())) {
              continue;
            }
            if (auto LI = TmpOrig.getDefiningOp<LLVM::LoadOp>()) {
              vtodo.push_back(LI.getAddr());
              continue;
            }
            //
            // TODO: handle globals in MLIResque way.
            //
            // if (auto CD = dyn_cast<ConstantDataSequential>(TmpOrig)) {
            //   for (size_t i = 0, len = CD->getNumElements(); i < len; i++)
            //     vtodo.push_back(CD->getElementAsConstant(i));
            //   continue;
            // }
            // if (auto CD = dyn_cast<ConstantAggregate>(TmpOrig)) {
            //   for (size_t i = 0, len = CD->getNumOperands(); i < len; i++)
            //     vtodo.push_back(CD->getOperand(i));
            //   continue;
            // }
            // if (auto GV = dyn_cast<GlobalVariable>(TmpOrig)) {
            //   // If operating under the assumption globals are inactive unless
            //   // explicitly marked as active, this is inactive
            //   if (!hasMetadata(GV, "enzyme_shadow") &&
            //       EnzymeNonmarkedGlobalsInactive) {
            //     continue;
            //   }
            //   if (GV->getName().contains("enzyme_const") ||
            //       InactiveGlobals.count(GV->getName().str())) {
            //     continue;
            //   }
            // }
            // auto TmpOrig_2 = getUnderlyingObject(TmpOrig, 100);
            // if (TmpOrig != TmpOrig_2) {
            //   vtodo.push_back(TmpOrig_2);
            //   continue;
            // }
            // if (EnzymePrintActivity)
            //   llvm::errs() << "      -- cannot continuing indirect store from "
            //                << *val << " due to " << *TmpOrig << "\n";
            shouldContinue = false;
            break;
          }
          if (shouldContinue) {
            // if (EnzymePrintActivity)
            //   llvm::errs() << "      -- continuing indirect store from " << *val
            //                << " into:\n";
            done.insert(std::make_tuple(SI.getOperation(), SI.getValue(), UA));
            for (Value TmpOrig : newAllocaSet) {

              for (Operation *a : TmpOrig.getUsers()) {
                todo.push_back(std::make_tuple(a, TmpOrig, UA));
                // if (EnzymePrintActivity)
                //   llvm::errs() << "         ** " << *a << "\n";
              }
              AllocaSet.insert(TmpOrig);
              shouldContinue = true;
            }
            continue;
          }
        }
      }
      if (SI.getAddr() != parent) {
        Value TmpOrig = SI.getAddr();
        // If storing into itself, all potential uses are taken care of
        // elsewhere in the recursion.
        bool shouldContinue = false;
        while (1) {
          if (AllocaSet.count(TmpOrig)) {
            shouldContinue = true;
            break;
          }
          if (TmpOrig.getDefiningOp<LLVM::AllocaOp>()) {
            done.insert(
                std::make_tuple(SI.getOperation(), SI.getAddr(), UA));
            for (const auto a : TmpOrig.getUsers()) {
              todo.push_back(std::make_tuple(a, TmpOrig, UA));
            }
            AllocaSet.insert(TmpOrig);
            shouldContinue = true;
            break;
          }
          if (PUA == UseActivity::None) {
            if (auto LI = TmpOrig.getDefiningOp<LLVM::LoadOp>()) {
              TmpOrig = LI.getAddr();
              continue;
            }
            // if (isAllocationCall(TmpOrig, TLI)) {
            //   done.insert(
            //       std::make_tuple(SI.getOperation(), SI.getAddr(), UA));
            //   for (Operation *a : TmpOrig.getUsers()) {
            //     todo.push_back(std::make_tuple(a, TmpOrig, UA));
            //   }
            //   AllocaSet.insert(TmpOrig);
            //   shouldContinue = true;
            //   break;
            // }
          }
//           auto TmpOrig_2 =
// #if LLVM_VERSION_MAJOR >= 12
//               getUnderlyingObject(TmpOrig, 100);
// #else
//               GetUnderlyingObject(
//                   TmpOrig, TR.getFunction()->getParent()->getDataLayout(), 100);
// #endif
//           if (TmpOrig != TmpOrig_2) {
//             TmpOrig = TmpOrig_2;
//             continue;
//           }
          break;
        }
        if (shouldContinue) {
          // if (EnzymePrintActivity)
          //   llvm::errs() << "      -- continuing indirect store2 from " << *val
          //                << " via " << *TmpOrig << "\n";
          continue;
        }
      }
      if (PUA == UseActivity::OnlyLoads) {
//         auto TmpOrig =
// #if LLVM_VERSION_MAJOR >= 12
//             getUnderlyingObject(SI->getPointerOperand(), 100);
// #else
//             GetUnderlyingObject(SI->getPointerOperand(),
//                                 TR.getFunction()->getParent()->getDataLayout(),
//                                 100);
// #endif
//         if (TmpOrig == val) {
//           continue;
//         }
      }
    }

    // if (!isa<Instruction>(a)) {
    //   if (auto CE = dyn_cast<ConstantExpr>(a)) {
    //     for (auto u : CE->users()) {
    //       todo.push_back(std::make_tuple(u, (Value *)CE, UA));
    //     }
    //     continue;
    //   }
    //   if (isa<ConstantData>(a)) {
    //     continue;
    //   }

    //   if (EnzymePrintActivity)
    //     llvm::errs() << "      unknown non instruction use of " << *val << " - "
    //                  << *a << "\n";
    //   return false;
    // }

    if (isa<LLVM::AllocaOp>(a)) {
      // if (EnzymePrintActivity)
      //   llvm::errs() << "found constant(" << (int)directions
      //                << ")  allocainst use:" << *val << " user " << *a << "\n";
      continue;
    }

    if (isa<LLVM::SIToFPOp, LLVM::UIToFPOp, LLVM::FPToSIOp, LLVM::FPToUIOp>(a)) {
      // if (EnzymePrintActivity)
      //   llvm::errs() << "found constant(" << (int)directions
      //                << ")  si-fp use:" << *val << " user " << *a << "\n";
      continue;
    }

    //
    // TODO: this should not happen in valid MLIR...
    //
    // if this instruction is in a different function, conservatively assume
    // it is active
    // Function *InstF = cast<Instruction>(a)->getParent()->getParent();
    // while (PPC.CloneOrigin.find(InstF) != PPC.CloneOrigin.end())
    //   InstF = PPC.CloneOrigin[InstF];

    // Function *F = TR.getFunction();
    // while (PPC.CloneOrigin.find(F) != PPC.CloneOrigin.end())
    //   F = PPC.CloneOrigin[F];

    // if (InstF != F) {
    //   if (EnzymePrintActivity)
    //     llvm::errs() << "found use in different function(" << (int)directions
    //                  << ")  val:" << *val << " user " << *a << " in "
    //                  << InstF->getName() << "@" << InstF
    //                  << " self: " << F->getName() << "@" << F << "\n";
    //   return false;
    // }
    // if (cast<Instruction>(a)->getParent()->getParent() != TR.getFunction())
    //   continue;

    // This use is only active if specified
    if (isa<LLVM::ReturnOp>(a)) {
      if (ActiveReturns == DIFFE_TYPE::CONSTANT &&
          UA != UseActivity::AllStores) {
        continue;
      } else {
        return false;
      }
    }

    if (isa<LLVM::MemcpyOp, LLVM::MemmoveOp>(a)) {
      //
      // TODO: see how this works with MLIR global constant operation approach
      //
      // copies of constant string data do not impact activity.
      // if (auto cexpr = dyn_cast<ConstantExpr>(call->getArgOperand(1))) {
      //   if (cexpr->getOpcode() == Instruction::GetElementPtr) {
      //     if (auto GV = dyn_cast<GlobalVariable>(cexpr->getOperand(0))) {
      //       if (GV->hasInitializer() && GV->isConstant()) {
      //         if (auto CDA =
      //                 dyn_cast<ConstantDataArray>(GV->getInitializer())) {
      //           if (CDA->getType()->getElementType()->isIntegerTy(8))
      //             continue;
      //         }
      //       }
      //     }
      //   }
      // }

      // Only need to care about loads from
      if (UA == UseActivity::OnlyLoads && a->getOperand(1) != parent)
        continue;

      // Only need to care about store from
      if (a->getOperand(0) != parent) {
        if (UA == UseActivity::OnlyStores)
          continue;
        else if (UA == UseActivity::OnlyNonPointerStores ||
                 UA == UseActivity::AllStores) {
          // todo can change this to query either -1 (all mem) or 0..size
          // (if size of copy is const)
          // if (!TR.query(call->getArgOperand(1))[{-1, -1}].isPossiblePointer())
          //   continue;
        }
      }

      bool shouldContinue = false;
      if (UA != UseActivity::AllStores)
        for (int arg = 0; arg < 2; arg++)
          if (a->getOperand(arg) != parent &&
              (arg == 0 || (PUA == UseActivity::None))) {
            Value TmpOrig = a->getOperand(arg);
            while (1) {
              if (AllocaSet.count(TmpOrig)) {
                shouldContinue = true;
                break;
              }
              if (TmpOrig.getDefiningOp<LLVM::AllocaOp>()) {
                done.insert(std::make_tuple(a, a->getOperand(arg), UA));
                for (Operation *a : TmpOrig.getUsers()) {
                  todo.push_back(std::make_tuple(a, TmpOrig, UA));
                }
                AllocaSet.insert(TmpOrig);
                shouldContinue = true;
                break;
              }
              if (PUA == UseActivity::None) {
                if (auto LI = TmpOrig.getDefiningOp<LLVM::LoadOp>()) {
                  TmpOrig = LI.getAddr();
                  continue;
                }
                // if (isAllocationCall(TmpOrig, TLI)) {
                //   done.insert(std::make_tuple((User *)call,
                //                               call->getArgOperand(arg), UA));
                //   for (const auto a : TmpOrig->users()) {
                //     todo.push_back(std::make_tuple(a, TmpOrig, UA));
                //   }
                //   AllocaSet.insert(TmpOrig);
                //   shouldContinue = true;
                //   break;
                // }
              }
              // auto TmpOrig_2 = getUnderlyingObject(TmpOrig, 100);
              // if (TmpOrig != TmpOrig_2) {
              //   TmpOrig = TmpOrig_2;
              //   continue;
              // }
              break;
            }
            if (shouldContinue)
              break;
          }

      if (shouldContinue)
        continue;
    }

    if (auto call = dyn_cast<CallOpInterface>(a)) {
      bool ConstantArg = isFunctionArgumentConstant(call, parent);
      if (ConstantArg && UA != UseActivity::AllStores) {
        // if (EnzymePrintActivity) {
        //   llvm::errs() << "Value found constant callinst use:" << *val
        //                << " user " << *call << "\n";
        // }
        continue;
      }

      if (Operation *F = getFunctionFromCall(call)) {
        if (UA == UseActivity::AllStores &&
            cast<SymbolOpInterface>(F).getName() == "julia.write_barrier")
          continue;
  
      } else if (PUA == UseActivity::None || PUA == UseActivity::OnlyStores) {
        // If calling a function derived from an alloca of this value,
        // the function is only active if the function stored into
        // the allocation is active (all functions not explicitly marked
        // inactive), or one of the args to the call is active
        Value operand = call.getCallableForCallee().dyn_cast<Value>();
        assert(operand);

        bool toContinue = false;
        if (operand.getDefiningOp<LLVM::LoadOp>()) {
          bool legal = true;

          for (unsigned i = 0; i < call.getArgOperands().size() + 1; ++i) {
            // FIXME: this is based on an assumption that the callee operand
            // precedes arg operands.
            Value a = call->getOperand(i);

            // FIXME: yet another ingrained assumption that integers cannot be
            // active.
            llvm::APInt intValue;
            if (matchPattern(a, m_ConstantInt(&intValue)))
              continue;

            Value ptr = a;
            bool subValue = false;
            // while (ptr) {
            //   auto TmpOrig2 = getUnderlyingObject(ptr, 100);
            //   if (AllocaSet.count(TmpOrig2)) {
            //     subValue = true;
            //     break;
            //   }
            //   if (isa<AllocaInst>(TmpOrig2)) {
            //     done.insert(std::make_tuple((User *)call, a, UA));
            //     for (const auto a : TmpOrig2->users()) {
            //       todo.push_back(std::make_tuple(a, TmpOrig2, UA));
            //     }
            //     AllocaSet.insert(TmpOrig2);
            //     subValue = true;
            //     break;
            //   }

            //   if (PUA == UseActivity::None) {
            //     if (isAllocationCall(TmpOrig2, TLI)) {
            //       done.insert(std::make_tuple((User *)call, a, UA));
            //       for (const auto a : TmpOrig2->users()) {
            //         todo.push_back(std::make_tuple(a, TmpOrig2, UA));
            //       }
            //       AllocaSet.insert(TmpOrig2);
            //       subValue = true;
            //       break;
            //     }
            //     if (auto L = dyn_cast<LoadInst>(TmpOrig2)) {
            //       ptr = L->getPointerOperand();
            //     } else
            //       ptr = nullptr;
            //   } else
            //     ptr = nullptr;
            // }
            if (subValue)
              continue;
            legal = false;
            break;
          }
          if (legal) {
            toContinue = true;
            break;
          }
        }
        if (toContinue)
          continue;
      }
    }

    // For an inbound gep, args which are not the pointer being offset
    // are not used in an active way by definition.
    if (auto gep = dyn_cast<LLVM::GEPOp>(a)) {
      // if (gep->isInBounds() && gep->getPointerOperand() != parent)
      //   continue;
    }

    //
    // TODO: in MLIR, a user is always an operation
    //
    // If this doesn't write to memory this can only be an active use
    // if its return is used in an active way, therefore add this to
    // the list of users to analyze
    if (Operation *I = a) {
      if (notForAnalysis.count(I->getBlock())) {
        // if (EnzymePrintActivity) {
        //   llvm::errs() << "Value found constant unreachable inst use:" << *val
        //                << " user " << *I << "\n";
        // }
        continue;
      }
      if (UA != UseActivity::AllStores && ConstantOperations.count(I)) {
        if (llvm::all_of(I->getResults(), [&](Value val) {
              return val.getType()
                         .isa<LLVM::LLVMVoidType, LLVM::LLVMTokenType>() ||
                     ConstantValues.count(val);
            })) {
          // if (EnzymePrintActivity) {
          //   llvm::errs() << "Value found constant inst use:" << *val << " user "
          //                << *I << "\n";
          // }
          continue;
        }
        UseActivity NU = UA;
        if (UA == UseActivity::OnlyLoads || UA == UseActivity::OnlyStores ||
            UA == UseActivity::OnlyNonPointerStores) {
          if (!isa<
            // clang-format off
            LLVM::GEPOp,
            // Integer binary ops.
            LLVM::AddOp,
            LLVM::SubOp,
            LLVM::MulOp,
            LLVM::UDivOp,
            LLVM::SDivOp,
            LLVM::URemOp,
            LLVM::SRemOp,
            LLVM::AndOp,
            LLVM::OrOp,
            LLVM::XOrOp,
            LLVM::ShlOp,
            LLVM::LShrOp,
            LLVM::AShrOp,
            // Float binary ops.
            LLVM::FAddOp,
            LLVM::FSubOp,
            LLVM::FMulOp,
            LLVM::FDivOp,
            LLVM::FRemOp,
            LLVM::FNegOp,
            // Cast op
            LLVM::BitcastOp,
            LLVM::AddrSpaceCastOp,
            LLVM::IntToPtrOp,
            LLVM::PtrToIntOp,
            LLVM::SExtOp,
            LLVM::ZExtOp,
            LLVM::TruncOp,
            LLVM::SIToFPOp,
            LLVM::UIToFPOp,
            LLVM::FPToSIOp,
            LLVM::FPToUIOp,
            LLVM::FPExtOp,
            LLVM::FPTruncOp
            // clang-format on
            >(I)) {
              NU = UseActivity::None;
          }
        }

        for (Value result : I->getResults()) {
          for (Operation *u : result.getUsers()) {
            todo.push_back(std::make_tuple(u, result, NU));
          }
        }
        continue;
      }
      // if (!I->mayWriteToMemory() || isa<LoadInst>(I)) {
      //   if (TR.query(I)[{-1}].isIntegral()) {
      //     continue;
      //   }
      //   UseActivity NU = UA;
      //   if (UA == UseActivity::OnlyLoads || UA == UseActivity::OnlyStores ||
      //       UA == UseActivity::OnlyNonPointerStores) {
      //     if (!isa<PHINode>(I) && !isa<CastInst>(I) &&
      //         !isa<GetElementPtrInst>(I) && !isa<BinaryOperator>(I))
      //       NU = UseActivity::None;
      //   }

      //   for (auto u : I->users()) {
      //     todo.push_back(std::make_tuple(u, (Value *)I, NU));
      //   }
      //   continue;
      // }

      if (FoundInst)
        *FoundInst = I;
    }

    // if (EnzymePrintActivity)
    //   llvm::errs() << "Value nonconstant inst (uses):" << *val << " user " << *a
    //                << "\n";
    seenuse = true;
    break;
  }

  // if (EnzymePrintActivity)
  //   llvm::errs() << " </Value USESEARCH" << (int)directions
  //                << " const=" << (!seenuse) << ">" << *val << "\n";
  return !seenuse;
}

/// Is the value potentially actively returned or stored
bool ActivityAnalyzer::isValueActivelyStoredOrReturned(TypeResults const &TR,
                                                       Value val,
                                                       bool outside) {
  // Must be an analyzer only searching down
  if (!outside)
    assert(directions == DOWN);

  bool ignoreStoresInto = true;
  auto key = std::make_pair(ignoreStoresInto, val);
  if (StoredOrReturnedCache.find(key) != StoredOrReturnedCache.end()) {
    return StoredOrReturnedCache[key];
  }

  // if (EnzymePrintActivity)
  //   llvm::errs() << " <ASOR" << (int)directions
  //                << " ignoreStoresinto=" << ignoreStoresInto << ">" << *val
  //                << "\n";

  StoredOrReturnedCache[key] = false;

  for (Operation *a : val.getUsers()) {
    if (isa<LLVM::AllocaOp>(a)) {
      continue;
    }
    // Loading a value prevents its pointer from being captured
    if (isa<LLVM::LoadOp>(a)) {
      continue;
    }

    if (isa<LLVM::ReturnOp>(a)) {
      if (ActiveReturns == DIFFE_TYPE::CONSTANT)
        continue;

      // if (EnzymePrintActivity)
      //   llvm::errs() << " </ASOR" << (int)directions
      //                << " ignoreStoresInto=" << ignoreStoresInto << ">"
      //                << " active from-ret>" << *val << "\n";
      StoredOrReturnedCache[key] = true;
      return true;
    }

    if (auto call = dyn_cast<CallOpInterface>(a)) {
      // if (!couldFunctionArgumentCapture(call, val)) {
      //   continue;
      // }
      bool ConstantArg = isFunctionArgumentConstant(call, val);
      if (ConstantArg) {
        continue;
      }
    }

    if (auto SI = dyn_cast<LLVM::StoreOp>(a)) {
      // If we are being stored into, not storing this value
      // this case can be skipped
      if (SI.getValue() != val) {
        if (!ignoreStoresInto) {
          // Storing into active value, return true
          if (!isConstantValue(TR, SI.getValue())) {
            StoredOrReturnedCache[key] = true;
            // if (EnzymePrintActivity)
            //   llvm::errs() << " </ASOR" << (int)directions
            //                << " ignoreStoresInto=" << ignoreStoresInto
            //                << " active from-store>" << *val
            //                << " store into=" << *SI << "\n";
            return true;
          }
        }
        continue;
      } else {
        // Storing into active memory, return true
        if (!isConstantValue(TR, SI.getAddr())) {
          StoredOrReturnedCache[key] = true;
          // if (EnzymePrintActivity)
          //   llvm::errs() << " </ASOR" << (int)directions
          //                << " ignoreStoresInto=" << ignoreStoresInto
          //                << " active from-store>" << *val << " store=" << *SI
          //                << "\n";
          return true;
        }
        continue;
      }
    }

    //
    // TODO: in MLIR, users are always operations
    //
    if (Operation *inst = a) {
      auto mayWriteToMemory = [](Operation *op) {
        auto iface = dyn_cast<MemoryEffectOpInterface>(op);
        if (!iface)
          return true;
        
        SmallVector<MemoryEffects::EffectInstance> effects;
        iface.getEffects(effects);
        return llvm::any_of(effects, [](const MemoryEffects::EffectInstance &effect) {
          return isa<MemoryEffects::Write>(effect.getEffect());
        });
      };

      if (!mayWriteToMemory(inst) /*||
          (isa<CallInst>(inst) && AA.onlyReadsMemory(cast<CallInst>(inst)))*/) {
        // // if not written to memory and returning a known constant, this
        // // cannot be actively returned/stored
        // if (inst->getParent()->getParent() == TR.getFunction() &&
        //     isConstantValue(TR, a)) {
        //   continue;
        // }

        // if not written to memory and returning a value itself
        // not actively stored or returned, this is not actively
        // stored or returned
        if (!llvm::all_of(a->getResults(), [&](Value val) {
              return isValueActivelyStoredOrReturned(TR, val, outside);
            })) {
          continue;
        }
      }
    }

    // if (isAllocationCall(a, TLI)) {
    //   // if not written to memory and returning a known constant, this
    //   // cannot be actively returned/stored
    //   if (isConstantValue(TR, a)) {
    //     continue;
    //   }
    //   // if not written to memory and returning a value itself
    //   // not actively stored or returned, this is not actively
    //   // stored or returned
    //   if (!isValueActivelyStoredOrReturned(TR, a, outside)) {
    //     continue;
    //   }
    // } else if (isDeallocationCall(a, TLI)) {
    //   // freeing memory never counts
    //   continue;
    // }
    // fallback and conservatively assume that if the value is written to
    // it is written to active memory
    // TODO handle more memory instructions above to be less conservative

    // if (EnzymePrintActivity)
    //   llvm::errs() << " </ASOR" << (int)directions
    //                << " ignoreStoresInto=" << ignoreStoresInto
    //                << " active from-unknown>" << *val << " - use=" << *a
    //                << "\n";
    return StoredOrReturnedCache[key] = true;
  }

  // if (EnzymePrintActivity)
  //   llvm::errs() << " </ASOR" << (int)directions
  //                << " ignoreStoresInto=" << ignoreStoresInto << " inactive>"
  //                << *val << "\n";
  return false;
}

void ActivityAnalyzer::InsertConstantOperation(TypeResults const &TR,
                                               Operation *I) {
  ConstantOperations.insert(I);
  auto found = ReEvaluateValueIfInactiveOp.find(I);
  if (found == ReEvaluateValueIfInactiveOp.end())
    return;
  auto set = std::move(ReEvaluateValueIfInactiveOp[I]);
  ReEvaluateValueIfInactiveOp.erase(I);
  for (Value toeval : set) {
    if (!ActiveValues.count(toeval))
      continue;
    ActiveValues.erase(toeval);
    // if (EnzymePrintActivity)
    //   llvm::errs() << " re-evaluating activity of val " << *toeval
    //                << " due to inst " << *I << "\n";
    isConstantValue(TR, toeval);
  }
}

void ActivityAnalyzer::InsertConstantValue(TypeResults const &TR,
                                           Value V) {
  ConstantValues.insert(V);
  auto found = ReEvaluateValueIfInactiveValue.find(V);
  if (found != ReEvaluateValueIfInactiveValue.end()) {
    auto set = std::move(ReEvaluateValueIfInactiveValue[V]);
    ReEvaluateValueIfInactiveValue.erase(V);
    for (auto toeval : set) {
      if (!ActiveValues.count(toeval))
        continue;
      ActiveValues.erase(toeval);
      // if (EnzymePrintActivity)
      //   llvm::errs() << " re-evaluating activity of val " << *toeval
      //                << " due to value " << *V << "\n";
      isConstantValue(TR, toeval);
    }
  }
  auto found2 = ReEvaluateOpIfInactiveValue.find(V);
  if (found2 != ReEvaluateOpIfInactiveValue.end()) {
    auto set = std::move(ReEvaluateOpIfInactiveValue[V]);
    ReEvaluateOpIfInactiveValue.erase(V);
    for (auto toeval : set) {
      if (!ActiveOperations.count(toeval))
        continue;
      ActiveOperations.erase(toeval);
      // if (EnzymePrintActivity)
      //   llvm::errs() << " re-evaluating activity of inst " << *toeval
      //                << " due to value " << *V << "\n";
      isConstantOperation(TR, toeval);
    }
  }
}
