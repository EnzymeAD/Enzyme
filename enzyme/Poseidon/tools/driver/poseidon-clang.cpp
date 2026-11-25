//===- poseidon-clang.cpp - the Poseidon compiler driver ------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Runs the clang the plugins were built against with Poseidon's plugin,
// Enzyme's to generate a profile, Poseidon's header directory and the runtimes
// the action links. A -poseidon-<flag> reaches the pass as
// -Xclang -mllvm -Xclang -poseidon-<flag>.
//
//===----------------------------------------------------------------------===//

#include "poseidon/poseidon.h"

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Driver/Action.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"
#include "llvm/TargetParser/Host.h"
#if LLVM_VERSION_MAJOR >= 22
#include "clang/Options/Options.h"
namespace clang_options = clang::options;
using clang::getDriverOptTable;
#else
#include "clang/Driver/Options.h"
namespace clang_options = clang::driver::options;
using clang::driver::getDriverOptTable;
#endif

#include <optional>
#include <string>
#include <system_error>

using namespace llvm;

/// Reports a mistake in what the driver is given as the pass reports one,
/// `<driver>: error: Poseidon: <message>`, and returns exit status 1.
static int userError(StringRef Prog, const Twine &Message) {
  errs() << Prog << ": error: Poseidon: " << Message << '\n';
  return 1;
}

/// The environment variable `Name`, or `Fallback` where it is unset or empty.
static std::string envOr(StringRef Name, StringRef Fallback) {
  std::optional<std::string> Value = sys::Process::GetEnv(Name);
  return Value && !Value->empty() ? *Value : Fallback.str();
}

/// `Rel` under the installation prefix `Prefix`.
static std::string underPrefix(StringRef Prefix, const Twine &Rel) {
  SmallString<256> P(Prefix);
  sys::path::append(P, Rel);
  return std::string(P);
}

/// The root of the GPU toolkit a link takes its runtime from: `Given`, else
/// `Built`, the one Poseidon was built against, which has to be on this
/// machine. A `Given` from the environment variable `GivenBy` has to be a
/// directory; clang checks a path on the command line itself.
static Expected<std::string> toolkitRoot(StringRef Toolkit, StringRef Given,
                                         StringRef GivenBy, StringRef Built,
                                         StringRef How) {
  if (!Given.empty() && !GivenBy.empty() && !sys::fs::is_directory(Given))
    return createStringError(GivenBy + "=" + Given +
                             " names no directory; set it to the " + Toolkit +
                             " installation or unset it");
  if (!Given.empty())
    return Given.str();
  if (Built.empty() || sys::fs::is_directory(Built))
    return Built.str();
  return createStringError("the " + Toolkit + " Poseidon was built against, " +
                           Built + ", is not on this machine; " + How);
}

/// Prints -poseidon-help.
static void printHelp(StringRef Prog) {
  outs()
      << Prog
      << ": clang with the Poseidon pass plugins loaded.\n\n"
         "  1. compile with -poseidon-profile-generate\n"
         "  2. run the program on a small problem; it writes its profile to\n"
         "     POSEIDON_PROFILE_DIR (default " POSEIDON_DEFAULT_PROFILE_DIR
         "). A program\n"
         "     that declares its result with poseidon_metric runs under\n"
         "     poseidon-probe -- <program>, which measures each site's\n"
         "     condition number against it in 1 + 3 x <sites> more runs\n"
         "  3. compile with -poseidon-profile-use, which prices candidates\n"
         "     from the cost model poseidon-calibrate measured for the\n"
         "     target device\n"
         "Profiling a host function needs glibc, whose __libc_malloc the\n"
         "profiler wraps to record heap extents.\n\n"
         "  -poseidon-profile-generate    instrument the annotated sites and\n"
         "                                link the profiler runtime\n"
         "  -poseidon-profile-use[=<dir>] rewrite from a profile\n"
         "                                "
         "(default " POSEIDON_DEFAULT_PROFILE_DIR ")\n"
         "  -poseidon-tau=<rel>           accuracy target of every site that\n"
         "                                does not carry its own\n"
         "  -poseidon-cost-budget=<share> share of its profiled cost a site\n"
         "                                with no target may add (negative:\n"
         "                                has to save)\n"
         "  -poseidon-cost-model=<csv>    cost model to price candidates "
         "with;\n"
         "                                without it the model for the target\n"
         "                                device is looked up "
         "(poseidon-calibrate)\n"
         "  -poseidon-kernels=<regex>     treat matching kernels as "
         "optimization\n"
         "                                sites ('all' matches every kernel)\n"
         "  -poseidon-min-cost-share=<f>  skip -poseidon-kernels sites below\n"
         "                                this share of the profiled FP cost\n"
         "  -poseidon-cache=<dir>         Herbie results, DP tables, "
         "descriptors\n"
         "  -poseidon-report-path=<dir>   write <func>.explain.json there\n"
         "  -poseidon-explain=<dir>       write <func>.explain.json (see\n"
         "                                poseidon-explain)\n"
         "  -poseidon-help                this message\n\n"
         "Any other -poseidon-<flag>[=<value>], with or without -mllvm before\n"
         "it, reaches the pass as -Xclang -mllvm -Xclang -poseidon-<flag>.\n"
         "POSEIDON_PLUGIN and ENZYME_PLUGIN override the plugin paths baked\n"
         "in at build time; POSEIDON_ECHO=1 prints the clang command line\n"
         "this driver runs. Every -poseidon-<flag> is listed by\n"
         "  "
      << Prog << " -mllvm -help-hidden -c -x c /dev/null\n";
}

/// The `<flag>[=<value>]` of -poseidon-<flag>[=<value>] or
/// --poseidon-<flag>[=<value>].
static std::optional<StringRef> poseidonFlag(StringRef A) {
  if (A.consume_front("-") &&
      (A.consume_front("-poseidon-") || A.consume_front("poseidon-")))
    return A;
  return std::nullopt;
}

/// Whether the argument is -poseidon-unit, bare or as the value of -mllvm=.
static bool namesUnit(StringRef A) {
  A.consume_front("-mllvm=");
  return A.consume_front("-") &&
         (A.consume_front("-poseidon-unit") ||
          A.consume_front("poseidon-unit")) &&
         (A.empty() || A.starts_with("="));
}

namespace {
/// What clang makes of a command line.
struct Invocation {
  /// Whether clang compiles or links an input, and whether it links an image.
  bool HasInput = false, Links = false;
  /// Whether it offloads to CUDA or HIP.
  bool CUDA = false, HIP = false;
  /// Whether an input is C++ (CUDA and HIP among them), and whether one is not
  /// an object.
  bool CXX = false, Source = false;
  /// The values of --cuda-path, --rocm-path, --hip-path and -stdlib.
  std::string CudaPath, RocmPath, HipPath, StdLib;
  /// The first compile-job argument that lets the code generator fuse or
  /// reorder floating-point operations across statements, or empty.
  std::string LooseFP;
};
} // namespace

/// Reads `Argv` the way clang's driver would, from the compilation it builds
/// without running a job. A command line clang rejects reads as one with no
/// input: clang reports it when the driver runs it.

static Invocation analyze(StringRef Clang, ArrayRef<const char *> Argv) {
  using namespace clang::driver;
  clang::DiagnosticOptions DiagOpts;
  clang::IgnoringDiagConsumer Ignore;
  clang::DiagnosticsEngine Diags(new clang::DiagnosticIDs(), DiagOpts, &Ignore,
                                 /*ShouldOwnClient=*/false);
  Driver D(Clang, sys::getDefaultTargetTriple(), Diags);
  // A missing input is clang's to report when it runs.
  D.setCheckInputsExist(false);
  std::unique_ptr<Compilation> C(D.BuildCompilation(Argv));
  Invocation I;
  if (!C)
    return I;
  const opt::ArgList &Args = C->getArgs();
  I.HasInput = Args.hasArg(clang_options::OPT_INPUT);
  I.Links = any_of(C->getActions(), [](const Action *A) {
    return A->getType() == types::TY_Image;
  });
  I.CUDA = C->hasOffloadToolChain<Action::OFK_Cuda>();
  I.HIP = C->hasOffloadToolChain<Action::OFK_HIP>();
  SmallVector<const Action *, 16> Work(C->getActions().begin(),
                                       C->getActions().end());
  while (!Work.empty()) {
    const Action *A = Work.pop_back_val();
    if (isa<InputAction>(A)) {
      I.CXX |= types::isCXX(A->getType());
      I.Source |= A->getType() != types::TY_Object;
    }
    Work.append(A->input_begin(), A->input_end());
  }
  I.CudaPath = Args.getLastArgValue(clang_options::OPT_cuda_path_EQ).str();
  I.RocmPath = Args.getLastArgValue(clang_options::OPT_rocm_path_EQ).str();
  I.HipPath = Args.getLastArgValue(clang_options::OPT_hip_path_EQ).str();
  I.StdLib = Args.getLastArgValue(clang_options::OPT_stdlib_EQ).str();
  for (const Command &Job : C->getJobs())
    for (StringRef A : Job.getArguments())
      if (I.LooseFP.empty() &&
          (A == "-ffp-contract=fast" ||
           A == "-ffp-contract=fast-honor-pragmas" || A == "-mreassociate" ||
           A == "-funsafe-math-optimizations"))
        I.LooseFP = A.str();
  return I;
}

namespace {
/// The user's arguments, with every Poseidon flag in the form clang hands to
/// the pass, and the action they ask for.
struct Request {
  SmallVector<std::string, 0> User;
  bool ProfileGenerate = false, ProfileUse = false;
};
} // namespace

/// `Args` with each Poseidon flag as -Xclang -mllvm -Xclang -poseidon-<flag>:
/// one clang's option table (`Parsed`) does not know, or the value of -mllvm.
/// A flag already written in that form is kept. -Xclang reaches every compile
/// job and no linker, so a line that only links is not warned about the flag.
static Request convertFlags(ArrayRef<const char *> Args,
                            const opt::InputArgList &Parsed) {
  Request R;
  auto Note = [&](StringRef Flag) {
    StringRef Name = Flag.split('=').first;
    R.ProfileGenerate |= Name == "profile-generate";
    R.ProfileUse |= Name == "profile-use";
  };
  // The converted flag that replaces Args[I], or the empty string for an
  // argument a converted flag consumes as well.
  SmallVector<std::optional<std::string>, 64> Converted(Args.size());
  StringRef PreviousXclang;
  for (const opt::Arg *A : Parsed) {
    // `Parsed` is of the arguments after the program name.
    unsigned At = A->getIndex() + 1;
    std::optional<StringRef> Flag;
    if (A->getOption().matches(clang_options::OPT_UNKNOWN))
      Flag = poseidonFlag(A->getSpelling());
    else if (A->getOption().matches(clang_options::OPT_mllvm))
      Flag = poseidonFlag(A->getValue());
    if (Flag) {
      Note(*Flag);
      Converted[At] = "-poseidon-" + Flag->str();
      if (A->getOption().matches(clang_options::OPT_mllvm) &&
          StringRef(Args[At]) == A->getSpelling())
        Converted[At + 1] = "";
    }
    if (A->getOption().matches(clang_options::OPT_Xclang)) {
      if (PreviousXclang == "-mllvm")
        if (std::optional<StringRef> Written = poseidonFlag(A->getValue()))
          Note(*Written);
      PreviousXclang = A->getValue();
    }
  }
  for (size_t I = 1; I < Args.size(); ++I) {
    if (!Converted[I])
      R.User.push_back(Args[I]);
    else if (!Converted[I]->empty())
      R.User.append({"-Xclang", "-mllvm", "-Xclang", *Converted[I]});
  }
  return R;
}

/// The -poseidon-unit of every compile job of this invocation, unique to it:
/// the host half of a compile reads the launch descriptors its device half
/// wrote under the same unit.
static std::string unitName(ArrayRef<const char *> Args) {
  // Without a working directory the arguments alone name the invocation.
  SmallString<256> Cwd;
  (void)sys::fs::current_path(Cwd);
  std::string Identity(Cwd);
  for (const char *A : Args) {
    Identity += '\0';
    Identity += A;
  }
  return "-poseidon-unit=" + utohexstr(xxh3_64bits(Identity),
                                       /*LowerCase=*/false, /*Width=*/16);
}

/// The arguments that link the runtimes `R` needs, Poseidon's from `LibDir`
/// and the GPU toolkit's, and for a C link (`CXX` false) the C++ library.
static Expected<SmallVector<std::string, 0>>
linkArguments(const Invocation &Inv, const Request &R, StringRef LibDir,
              bool CXX) {
  SmallVector<std::string, 0> Link;
  auto AsNeeded = [&](StringRef Lib) {
    Link.append({"-Wl,--push-state,--as-needed", Lib.str(), "-Wl,--pop-state"});
  };
  // A link of objects alone names no language, so it gets the runtimes of the
  // GPU this build has, as needed, from a toolkit that is installed.
  bool Objects = !Inv.Source;
  bool CUDA =
      Inv.CUDA || (Objects && !Inv.HIP && POSEIDON_DRIVER_PROFILER_CUDA);
  bool HIP = Inv.HIP || (Objects && !CUDA && POSEIDON_DRIVER_PROFILER_HIP);
  std::string CudaRoot, RocmRoot;
  if (CUDA && Objects) {
    if (sys::fs::is_directory(POSEIDON_DRIVER_CUDA_ROOT))
      CudaRoot = POSEIDON_DRIVER_CUDA_ROOT;
  } else if (CUDA) {
    Expected<std::string> Root =
        toolkitRoot("CUDA toolkit", Inv.CudaPath, "", POSEIDON_DRIVER_CUDA_ROOT,
                    "pass --cuda-path=<dir>");
    if (!Root)
      return Root.takeError();
    CudaRoot = *Root;
  }
  if (HIP && Objects) {
    if (sys::fs::is_directory(POSEIDON_DRIVER_ROCM_ROOT))
      RocmRoot = POSEIDON_DRIVER_ROCM_ROOT;
  } else if (HIP) {
    // Where clang takes the HIP runtime from: --hip-path, HIP_PATH,
    // --rocm-path, ROCM_PATH, else the ROCm Poseidon was built against.
    std::string Given = Inv.HipPath;
    StringRef GivenBy;
    if (Given.empty() && !(Given = envOr("HIP_PATH", "")).empty())
      GivenBy = "HIP_PATH";
    if (Given.empty())
      Given = Inv.RocmPath;
    if (Given.empty() && !(Given = envOr("ROCM_PATH", "")).empty())
      GivenBy = "ROCM_PATH";
    Expected<std::string> Root =
        toolkitRoot("ROCm", Given, GivenBy, POSEIDON_DRIVER_ROCM_ROOT,
                    "pass --rocm-path=<dir> or set ROCM_PATH");
    if (!Root)
      return Root.takeError();
    RocmRoot = *Root;
  }
  // clang links no CUDA runtime of its own.
  if (!CudaRoot.empty()) {
    Link.push_back("-L" + CudaRoot + "/lib64");
    if (Objects)
      AsNeeded("-lcudart");
    else
      Link.push_back("-lcudart");
  }
  if (!RocmRoot.empty()) {
    Link.push_back("-L" + RocmRoot + "/lib");
    if (Objects)
      AsNeeded("-lamdhip64");
    else
      Link.push_back("-lamdhip64");
  }
  Link.push_back(("-L" + LibDir).str());
  // The shadows of a POSEIDON_OPTIMIZE function on the host are sized from
  // the allocations this library records.
  if (R.ProfileGenerate)
    Link.push_back("-lposeidon_profile_host");
  // The host half of the GPU profiler. The CUDA and HIP ones export the same
  // names, so a link takes one.
  if (R.ProfileGenerate && HIP && POSEIDON_DRIVER_PROFILER_HIP)
    Link.push_back("-lposeidon_profile_hip");
  else if (R.ProfileGenerate && CUDA && POSEIDON_DRIVER_PROFILER_CUDA)
    Link.push_back("-lposeidon_profile_cuda");
  // poseidon_metric, and the profiler the instrumented code calls into.
  Link.push_back("-lposeidon_profile");
  if (R.ProfileUse && CUDA && !CudaRoot.empty()) {
    // A rewrite may dispatch a matrix product to any of the GEMM runtimes;
    // they export disjoint symbols and only the chosen one runs.
    Link.push_back("-lposeidon_rt");
    AsNeeded("-lcublas");
    // Libraries of the runtime outside the toolkit (cuMpSGEMM, which defines
    // cuBLAS's GEMM entry points, so after cuBLAS), found again where they
    // were built. -Xlinker, since after a -x cuda clang compiles a file input.
    SmallVector<StringRef, 2> RuntimeLibs;
    StringRef(POSEIDON_DRIVER_RT_LIBS).split(RuntimeLibs, ',', -1, false);
    for (StringRef Lib : RuntimeLibs)
      Link.append({"-Xlinker", Lib.str(), "-Xlinker", "-rpath", "-Xlinker",
                   sys::path::parent_path(Lib).str()});
  }
  if (R.ProfileUse && HIP && !RocmRoot.empty()) {
    // Ozaki-II and the direct products on hipBLAS; TCEC has no HIP build.
    Link.push_back("-lposeidon_rt_hip");
    AsNeeded("-lhipblas");
  }
  // The archives are C++; the C driver links no C++ library of its own.
  if (!CXX)
    Link.append({Inv.StdLib == "libc++" ? "-lc++" : "-lstdc++", "-lm"});
  return Link;
}

namespace {
/// The plugins, header directory and runtime directory a compile uses.
struct Installation {
  std::string Plugin, Enzyme, Include, LibDir;
};
} // namespace

/// Those of the build tree the driver `Exe` runs from or of its installation,
/// never some of each; Enzyme's plugin only to generate a profile.
static Expected<Installation> locate(StringRef Exe, bool ProfileGenerate) {
  StringRef BuiltEnzyme = POSEIDON_DRIVER_ENZYME;
  Installation In = {POSEIDON_DRIVER_PLUGIN, BuiltEnzyme.str(),
                     POSEIDON_DRIVER_INCLUDE, POSEIDON_DRIVER_LIBDIR};
  if (!sys::fs::equivalent(sys::path::parent_path(Exe),
                           POSEIDON_DRIVER_BINDIR)) {
    StringRef Prefix = sys::path::parent_path(sys::path::parent_path(Exe));
    In.Plugin = underPrefix(Prefix, POSEIDON_DRIVER_INSTALL_LIBDIR
                            "/" POSEIDON_DRIVER_PLUGIN_NAME);
    if (!BuiltEnzyme.empty())
      // Enzyme installs its plugins to lib/.
      In.Enzyme =
          underPrefix(Prefix, "lib/" + sys::path::filename(BuiltEnzyme));
    In.Include = underPrefix(Prefix, POSEIDON_DRIVER_INSTALL_INCLUDEDIR);
    In.LibDir = underPrefix(Prefix, POSEIDON_DRIVER_INSTALL_LIBDIR);
  }
  In.Plugin = envOr("POSEIDON_PLUGIN", In.Plugin);
  if (!sys::fs::exists(In.Plugin))
    return createStringError("Poseidon plugin not found at " + In.Plugin +
                             "; set POSEIDON_PLUGIN");
  // Enzyme differentiates the probes and nothing else; a program with
  // derivatives of its own loads it itself, as with clang alone.
  if (!ProfileGenerate) {
    In.Enzyme.clear();
  } else if (std::string Given = envOr("ENZYME_PLUGIN", ""); !Given.empty()) {
    In.Enzyme = Given;
  } else if (In.Enzyme.empty()) {
    return createStringError("-poseidon-profile-generate needs Enzyme to "
                             "differentiate the probes, and this build has no "
                             "ClangEnzyme plugin; set ENZYME_PLUGIN to one");
  } else if (!sys::fs::exists(In.Enzyme)) {
    return createStringError("Enzyme plugin not found at " + In.Enzyme +
                             "; install Enzyme to the same prefix or set "
                             "ENZYME_PLUGIN");
  }
  return In;
}

/// Runs `Cmd`, from a response file written as clang writes one when the
/// expanded command is longer than the system takes.
static int runClang(StringRef Prog, ArrayRef<std::string> Cmd) {
  SmallVector<StringRef, 128> Argv(Cmd.begin(), Cmd.end());
  SmallString<128> ResponseFile;
  FileRemover RemoveResponseFile;
  std::string ResponseArg;
  if (!sys::commandLineFitsWithinSystemLimits(Cmd.front(), Argv)) {
    std::string Contents;
    raw_string_ostream OS(Contents);
    for (StringRef A : ArrayRef<StringRef>(Argv).drop_front()) {
      sys::printArg(OS, A, /*Quote=*/true);
      OS << ' ';
    }
    if (std::error_code EC =
            sys::fs::createTemporaryFile("poseidon-clang", "rsp", ResponseFile))
      return userError(Prog, "cannot create a response file: " + EC.message());
    RemoveResponseFile.setFile(ResponseFile);
    if (std::error_code EC = sys::writeFileWithEncoding(ResponseFile, Contents))
      return userError(Prog,
                       "cannot write " + ResponseFile + ": " + EC.message());
    ResponseArg = ("@" + ResponseFile).str();
    Argv.truncate(1);
    Argv.push_back(ResponseArg);
  }
  std::string Err;
  int RC = sys::ExecuteAndWait(Cmd.front(), Argv, std::nullopt, {}, 0, 0, &Err);
  if (!Err.empty())
    userError(Prog, "cannot run " + Cmd.front() + ": " + Err);
  return RC;
}

int main(int argc, char **argv) {
  std::string Exe = sys::fs::getMainExecutable(argv[0], (void *)(intptr_t)main);
  // argv[0], not the resolved binary: poseidon-clang++ is a symlink to
  // poseidon-clang and the name it was invoked under chooses the clang driver.
  StringRef Prog = sys::path::filename(argv[0]);

  // Response files are expanded first, with the quoting clang reads them with,
  // so that a Poseidon flag in one is converted and the file is no input.
  BumpPtrAllocator Alloc;
  SmallVector<const char *, 64> Args(argv, argv + argc);
  if (Error E = cl::ExpansionContext(Alloc, cl::TokenizeGNUCommandLine)
                    .expandResponseFiles(Args))
    return userError(Prog, toString(std::move(E)));
  for (StringRef A : ArrayRef<const char *>(Args).drop_front())
    if (A == "-poseidon-help" || A == "--poseidon-help") {
      printHelp(Prog);
      return 0;
    }

  unsigned MissingIndex, MissingCount;
  opt::InputArgList Parsed = getDriverOptTable().ParseArgs(
      ArrayRef<const char *>(Args).drop_front(), MissingIndex, MissingCount,
      opt::Visibility(clang_options::ClangOption));
  Request R = convertFlags(Args, Parsed);
  if (R.ProfileGenerate && R.ProfileUse)
    return userError(Prog, "-poseidon-profile-generate and "
                           "-poseidon-profile-use are the two halves of the "
                           "workflow; pass one");

  // What clang will do with the rest, with the floating-point contract
  // Poseidon gives it ahead of the user's arguments.
  std::string ClangFor =
      Prog.contains("++") ? POSEIDON_DRIVER_CLANGXX : POSEIDON_DRIVER_CLANG;
  SmallVector<const char *, 64> Probe = {ClangFor.c_str(), "-ffp-contract=on"};
  for (const std::string &A : R.User)
    Probe.push_back(A.c_str());
  Invocation Inv = analyze(ClangFor, Probe);

  // Error-free transformations survive only if nothing is fused or reordered
  // across statements. Before LLVM 24 the code generator applies
  // -ffp-contract=fast and fast-math to every operation, the expansions'
  // included.
#if LLVM_VERSION_MAJOR < 24
  if (R.ProfileUse && !Inv.LooseFP.empty()) {
    const opt::Arg *Loose = Parsed.getLastArg(
        clang_options::OPT_ffast_math, clang_options::OPT_Ofast,
        clang_options::OPT_ffp_contract, clang_options::OPT_ffp_model_EQ,
        clang_options::OPT_fassociative_math,
        clang_options::OPT_funsafe_math_optimizations);
    return userError(
        Prog, (Loose ? Loose->getAsString(Parsed) : std::string("a flag")) +
                  " lets the compiler fuse or reorder floating-point "
                  "operations across statements (clang passes " +
                  Inv.LooseFP +
                  "), which collapses the floating-point expansions "
                  "Poseidon emits; drop it from the profile-use compile");
  }
#endif

  if (Inv.HasInput &&
      llvm::none_of(ArrayRef<const char *>(Args).drop_front(), namesUnit))
    R.User.insert(R.User.begin(),
                  {"-Xclang", "-mllvm", "-Xclang", unitName(Args)});

  // The clang the plugin is built against, wherever the driver runs. A C++
  // input is linked by the C++ driver.
  bool WantCXX = Prog.contains("++") || Inv.CXX;
  std::string Clang = WantCXX ? POSEIDON_DRIVER_CLANGXX : POSEIDON_DRIVER_CLANG;
  Expected<Installation> Found = locate(Exe, R.ProfileGenerate);
  if (!Found)
    return userError(Prog, toString(Found.takeError()));
  const Installation &In = *Found;

  SmallVector<std::string, 128> Cmd;
  Cmd.push_back(Clang);
  // Without an input clang compiles and links nothing (-v, --version,
  // -print-*), and gets the user's arguments alone.
  if (Inv.HasInput) {
    // Before LLVM 22 clang parses -mllvm before loading -fpass-plugin, so the
    // plugins' options are known only if they are also loaded with -load.
    auto AddPlugin = [&](const std::string &P) {
      Cmd.push_back("-fpass-plugin=" + P);
#if LLVM_VERSION_MAJOR < 22
      Cmd.append({"-Xclang", "-load", "-Xclang", P});
#endif
    };
    AddPlugin(In.Plugin);
    if (!In.Enzyme.empty())
      AddPlugin(In.Enzyme);
    // A HIP device compile is an LTO pre-link compile, where Enzyme leaves
    // derivatives to a device link that loads no plugin. Ahead of the user's
    // arguments, so that their -enzyme-lto-prelink wins.
    if (Inv.HIP && !In.Enzyme.empty())
      Cmd.append(
          {"-Xclang", "-mllvm", "-Xclang", POSEIDON_DRIVER_ENZYME_HIP_OPTION});
    // A system directory, as the CMake package gives it.
    Cmd.append({"-isystem", In.Include});
    // Error-free transformations survive only if the compiler contracts within
    // a statement and never across one.
    Cmd.push_back("-ffp-contract=on");
    // poseidon.h then keeps the device math a rewrite or a derivative calls.
    Cmd.push_back("-DPOSEIDON_DEVICE_MATH");
  }
  Cmd.append(R.User.begin(), R.User.end());
  if (Inv.Links) {
    Expected<SmallVector<std::string, 0>> Link =
        linkArguments(Inv, R, In.LibDir, WantCXX);
    if (!Link)
      return userError(Prog, toString(Link.takeError()));
    Cmd.append(Link->begin(), Link->end());
  }

  if (sys::Process::GetEnv("POSEIDON_ECHO")) {
    for (const std::string &A : Cmd)
      errs() << A << ' ';
    errs() << '\n';
  }
  return runClang(Prog, Cmd);
}
