//===- poseidon-calibrate.cpp - measure a device's cost model -------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Measures the cost model Poseidon prices candidates from, once per GPU or
// host CPU, where a later compile finds it by the module's target-cpu.
//
//===----------------------------------------------------------------------===//

#include "Model/CostTable.h"
#include "runtime/gemm/SchemeConstants.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"

#include <map>
#include <optional>
#include <string>

using namespace llvm;
using namespace poseidon;

static cl::opt<std::string>
    GpuId("gpu", cl::init(""), cl::value_desc("id"),
          cl::desc("Device to measure (default: the first of "
                   "CUDA_VISIBLE_DEVICES, or of "
                   "HIP_VISIBLE_DEVICES on an AMD GPU, else "
                   "0)"));
static cl::opt<std::string>
    Cpu("cpu", cl::init(""), cl::value_desc("target-cpu"),
        cl::desc("Measure this host for the target-cpu a compile uses: native, "
                 "or a name such as x86-64, as -march= takes it"));
static cl::opt<std::string>
    Out("out", cl::init(""), cl::value_desc("path"),
        cl::desc("Cost model to write; a path not ending in .csv is a "
                 "directory, which takes cm_<arch>_<device>.csv "
                 "(default: $XDG_CACHE_HOME/poseidon)"));
static cl::opt<bool>
    Check("check", cl::desc("Validate the resolved cost model against this "
                            "device instead of measuring"));
static cl::opt<bool>
    Force("force", cl::desc("Re-measure families that are already present"));
static cl::opt<std::string>
    Only("only", cl::init(""), cl::value_desc("arms"),
         cl::desc("Comma-separated subset of "
                  "ops,ozaki,ozaki-rect,tcec,direct,inkernel,"
                  "herbie-platform"));
static cl::opt<std::string>
    Python("python", cl::init(POSEIDON_CALIBRATE_PYTHON),
           cl::value_desc("path"),
           cl::desc("Python 3 the herbie-platform arm and the per-op arm of "
                    "--cpu run under (default: the one this build was "
                    "configured with, else python3 on PATH)"));
static cl::opt<unsigned> CpuIterations(
    "cpu-iterations", cl::Hidden, cl::value_desc("count"),
    cl::desc("Loop trips of each CPU microbenchmark run, in place of the "
             "count a trial run sizes"));
static cl::opt<bool> CpuChainOnly(
    "cpu-chain-only", cl::Hidden,
    cl::desc("Time every CPU operation in the chain through itself, which the "
             "microbenchmark refuses where the operation folds away"));
static cl::opt<std::string>
    Herbie("herbie", cl::init(POSEIDON_CALIBRATE_HERBIE),
           cl::value_desc("path"),
           cl::desc("Herbie binary the generated platform must load into "
                    "(default: the one this build was configured with)"));
static cl::opt<std::string> Product(
    "product", cl::init(""), cl::value_desc("MxNxK[:layout],..."),
    cl::desc("Matrix products of constant extents, M x N output and K "
             "reduction, whose rows the ozaki-rect arm (Ozaki-II and native "
             "DGEMM) and, given the layout of B along K (browmajor strided, "
             "bcolmajor contiguous), the inkernel arm measure at exactly "
             "those extents. A compile's warning names the products whose "
             "rows it lacks"));
static cl::opt<std::string> OzakiRect(
    "ozaki-rect", cl::init(""), cl::value_desc("MxK,..."),
    cl::desc("(M, K) of host GEMMs whose column count is a run-time extent, "
             "whose Ozaki-II rows the ozaki-rect arm measures at many "
             "columns. A compile's warning names the shapes whose rows it "
             "lacks"));
static cl::opt<unsigned>
    RefN("n", cl::init(2048), cl::value_desc("N"),
         cl::desc("Square reference shape the library arms and "
                  "the in-kernel class rows are timed at"));
static cl::opt<unsigned>
    Reps("reps", cl::init(30), cl::value_desc("count"),
         cl::desc("Timed repetitions of each library call and in-kernel "
                  "measurement"));

/// The directory of this executable.
static std::string ToolDir;

/// The rows this run was asked for and did not measure, which make it fail.
static SmallVector<std::string, 8> Unmeasured;

/// The tool `Name` installed next to this one.
static std::string sibling(StringRef Name) {
  SmallString<256> P(ToolDir);
  sys::path::append(P, Name);
  return std::string(P);
}

namespace {
/// The GPU calibrators, each built once per vendor.
enum class Calibrator {
  /// Per-op throughputs and the --device query.
  Ops,
  /// The library-call (library call) rows.
  Library
};
} // namespace

/// The installed name of `C` for an AMD GPU (`Hip`) or an NVIDIA one, as
/// tools/CMakeLists.txt names the executables.
static StringRef calibratorName(Calibrator C, bool Hip) {
  switch (C) {
  case Calibrator::Ops:
    return Hip ? "poseidon-calibrate-ops-hip" : "poseidon-calibrate-ops-cuda";
  case Calibrator::Library:
    return Hip ? "poseidon-calibrate-library-hip"
               : "poseidon-calibrate-library-cuda";
  }
  llvm_unreachable("every calibrator has a name");
}

/// `Rel` under share/poseidon of this tool's installation, else under the
/// source tree it was built from; the two layouts match below that point.
static std::string shipped(StringRef Rel) {
  SmallString<256> P(sys::path::parent_path(ToolDir));
  sys::path::append(P, "share/poseidon", Rel);
  if (sys::fs::exists(P))
    return std::string(P);
  SmallString<256> S(POSEIDON_CALIBRATE_SRCDIR);
  sys::path::append(S, Rel);
  return std::string(S);
}

/// `V` with six decimals, as a row holds it.
static std::string decimal(double V) { return formatv("{0:F6}", V).str(); }

/// Prints `poseidon-calibrate: <Message>` to stderr, after what was printed to
/// stdout.
static void printError(const Twine &Message) {
  outs().flush();
  errs() << "poseidon-calibrate: " << Message << '\n';
}

/// Prints `Message` as an error and exits with status 1.
[[noreturn]] static void fail(const Twine &Message) {
  printError(Message);
  std::exit(1);
}

/// Runs a program to its exit status, its stdout and stderr sent to the files
/// named, if any.
static int run(ArrayRef<StringRef> Argv, StringRef StdoutFile = "",
               StringRef StderrFile = "") {
  outs().flush();
  std::optional<StringRef> Redirects[3] = {std::nullopt, std::nullopt,
                                           std::nullopt};
  if (!StdoutFile.empty())
    Redirects[1] = StdoutFile;
  if (!StderrFile.empty())
    Redirects[2] = StderrFile;
  std::string Err;
  int RC =
      sys::ExecuteAndWait(Argv[0], Argv, std::nullopt, Redirects, 0, 0, &Err);
  if (!Err.empty())
    fail(Argv[0] + ": " + Err);
  return RC;
}

/// The contents of `Path`; a file that cannot be read is fatal.
static std::string readFile(StringRef Path) {
  auto Buf = MemoryBuffer::getFile(Path);
  if (!Buf)
    fail("cannot read " + Path + ": " + Buf.getError().message());
  return (*Buf)->getBuffer().str();
}

/// Writes `Path` whole or not at all, so that an interrupted run leaves the
/// model it started from.
static void writeFile(StringRef Path, StringRef Content) {
  std::string Part = (Path + ".part").str();
  {
    std::error_code EC;
    raw_fd_ostream OS(Part, EC);
    if (EC)
      fail("cannot write " + Part + ": " + EC.message());
    OS << Content;
  }
  if (std::error_code EC = sys::fs::rename(Part, Path))
    fail("cannot write " + Path + ": " + EC.message());
}

/// The stdout of a program that exits with status 0.
static std::optional<std::string> capture(ArrayRef<StringRef> Argv,
                                          StringRef StderrFile = "") {
  SmallString<128> Tmp;
  if (sys::fs::createTemporaryFile("poseidon-calibrate", "out", Tmp))
    fail("cannot create a temporary file");
  FileRemover RemoveTmp(Tmp);
  if (run(Argv, Tmp, StderrFile))
    return std::nullopt;
  return readFile(Tmp);
}

/// Whether `Csv` holds a row of `Family`.
static bool hasFamily(StringRef Csv, StringRef Family) {
  for (StringRef Line : llvm::split(Csv, '\n'))
    if (Line.starts_with(Family) &&
        Line.drop_front(Family.size()).starts_with(","))
      return true;
  return false;
}

/// The (family, key) a row is identified by.
static std::pair<StringRef, StringRef> rowKey(StringRef Line) {
  auto [Family, Rest] = Line.split(',');
  return {Family, Rest.split(',').first};
}

/// Replaces the rows of `Csv` whose (family, key) `Rows` carry, and keeps the
/// others.
static void mergeRows(std::string &Csv, StringRef Rows) {
  SmallVector<StringRef, 8> New;
  for (StringRef Line : llvm::split(Rows, '\n'))
    if (!Line.trim().empty())
      New.push_back(Line.trim());
  std::string Kept;
  for (StringRef Line : llvm::split(Csv, '\n')) {
    if (Line.empty() || llvm::any_of(New, [&](StringRef N) {
          return rowKey(N) == rowKey(Line);
        }))
      continue;
    Kept += Line;
    Kept += '\n';
  }
  for (StringRef Line : New) {
    Kept += Line;
    Kept += '\n';
  }
  Csv = Kept;
}

namespace {
/// The device a cost model is measured on.
struct Device {
  /// A GPU's id among the visible devices.
  std::string Id;
  /// The name in the model's file name, punctuation stripped.
  std::string Name;
  /// sm_<cc>, gfx<NNN> on an AMD GPU, a CPU's target-cpu.
  std::string Arch;
  /// An AMD GPU, which the -hip tools measure.
  bool Hip = false;
  /// This host's CPU (--cpu).
  bool Host = false;
  /// A host's model name, as the system reports it.
  std::string Model;
  /// The host's features, when Arch is its own CPU.
  std::string Features;
};
} // namespace

/// The exit status of a microbenchmark that finds no device of its vendor.
constexpr int kExitNoDevice = 2;

/// The GPU --gpu names, else the first the vendor's visible-devices variable
/// lists, as the microbenchmark of each vendor this build has names it, CUDA's
/// first. A microbenchmark that finds no device of its vendor passes the
/// question to the next; one that cannot initialize the device, or cannot be
/// started, ends the run with its reason.
static Device queryDevice() {
  // What each microbenchmark said when it found no device.
  std::string Reasons;
  for (bool Hip : {false, true}) {
    const StringRef Vendor = Hip ? "HIP" : "CUDA";
    std::string Exe = sibling(calibratorName(Calibrator::Ops, Hip));
    if (!sys::fs::exists(Exe))
      continue;
    const char *Visible = Hip ? "HIP_VISIBLE_DEVICES" : "CUDA_VISIBLE_DEVICES";
    Device D;
    D.Hip = Hip;
    D.Id = GpuId;
    if (D.Id.empty()) {
      if (const char *V = ::getenv(Visible))
        D.Id = StringRef(V).split(',').first.str();
      if (D.Id.empty())
        D.Id = "0";
    }
    // The probe, and every child after it, sees that device alone.
    ::setenv(Visible, D.Id.c_str(), 1);
    SmallString<128> Out, Err;
    if (sys::fs::createTemporaryFile("poseidon-calibrate", "out", Out) ||
        sys::fs::createTemporaryFile("poseidon-calibrate", "err", Err))
      fail("cannot create a temporary file");
    FileRemover RemoveOut(Out), RemoveErr(Err);
    outs().flush();
    std::optional<StringRef> Redirects[3] = {std::nullopt, StringRef(Out),
                                             StringRef(Err)};
    std::string Failure;
    StringRef Argv[] = {Exe, "--device"};
    int RC =
        sys::ExecuteAndWait(Exe, Argv, std::nullopt, Redirects, 0, 0, &Failure);
    std::string Why = StringRef(readFile(Err)).trim().str();
    if (RC == kExitNoDevice) {
      Reasons += ("\n  " + sys::path::filename(Exe) + ": " + Why).str();
      continue;
    }
    if (!Failure.empty() || (RC != 0 && RC != 1)) {
      if (Failure.empty())
        Failure = "exit status " + itostr(RC);
      fail("the " + Vendor + " microbenchmark " + sys::path::filename(Exe) +
           " could not be started (" + Failure + ")" +
           (Why.empty() ? "" : ": " + Why) +
           (Reasons.empty() ? ""
                            : "; the microbenchmarks before it found no "
                              "device:" +
                                  Reasons));
    }
    if (RC != 0)
      fail("the " + Vendor + " microbenchmark could not initialize device " +
           D.Id + ": " + Why);
    std::string Named = readFile(Out);
    auto [NameRef, Arch] = StringRef(Named).split('\n').first.split('\t');
    if (Arch.trim().empty())
      fail("the " + Vendor + " microbenchmark named device " + D.Id +
           " no processor: '" + StringRef(Named).trim() + "'");
    // A device ID several products share has a marketing name listing them all
    // ("AMD Instinct MI250X / MI250"); the first names the device. The vendor
    // prefixes say nothing about it.
    for (char C : NameRef.split('/').first.trim())
      if (isAlnum(C))
        D.Name += C;
    StringRef N(D.Name);
    for (StringRef Prefix :
         {"NVIDIA", "GeForce", "Tesla", "Quadro", "AMD", "Instinct", "Radeon"})
      while (N.consume_front(Prefix))
        ;
    D.Name = N.str();
    if (D.Name.empty())
      fail("the GPU runtime gives device " + D.Id + " no name");
    D.Arch = Arch.trim().str();
    return D;
  }
  fail("no GPU with id " + (GpuId.empty() ? Twine("0") : Twine(GpuId)) +
       " is visible to this build's microbenchmarks" +
       (Reasons.empty() ? "; measure this host with --cpu <target-cpu>"
                        : ":" + Reasons));
}

/// The model name the system reports for this host's CPU, or empty.
static std::string hostModelName() {
  auto Buf = MemoryBuffer::getFileAsStream("/proc/cpuinfo");
  if (!Buf)
    return std::string();
  for (StringRef Line : llvm::split((*Buf)->getBuffer(), '\n')) {
    auto [Key, Value] = Line.split(':');
    if (Key.trim() == "model name")
      return Value.trim().str();
  }
  return std::string();
}

/// This host, measured for the target-cpu --cpu names: `native` takes the CPU
/// and features LLVM detects, as -march=native does, and a named CPU that
/// CPU's own features. The device is named by the CPU's model name, or by the
/// detected CPU on a system that reports none.
static Device queryHost() {
  Device D;
  D.Host = true;
  D.Arch = Cpu;
  StringRef HostCpu = sys::getHostCPUName();
  if (D.Arch == "native") {
    if (HostCpu.empty() || HostCpu == "generic")
      fail("LLVM does not recognise this host's CPU, so --cpu native names "
           "no target-cpu; pass the -march= name the compile uses");
    D.Arch = HostCpu.str();
    for (const auto &F : sys::getHostCPUFeatures())
      D.Features += (D.Features.empty() ? "" : ",") +
                    ((F.second ? "+" : "-") + F.first()).str();
  }
  D.Model = hostModelName();
  if (D.Model.empty())
    D.Model = HostCpu.str();
  std::string Model = StringRef(D.Model).split('@').first.str();
  for (StringRef Mark : {"(R)", "(TM)", "(tm)"})
    for (size_t At; (At = Model.find(Mark)) != std::string::npos;)
      Model.erase(At, Mark.size());
  std::string Clean;
  for (char C : Model)
    if (isAlnum(C))
      Clean += C;
  StringRef N(Clean);
  for (StringRef Prefix : {"AMD", "Intel"})
    while (N.consume_front(Prefix))
      ;
  D.Name = N.str();
  if (D.Name.empty())
    fail("this host's CPU has no name to file its cost model under; pass "
         "--out <file>.csv");
  return D;
}

/// The model file of `D`: --out, a directory --out names, or the user cache.
static std::string resolveCsv(const Device &D) {
  std::string Path = Out;
  if (Path.empty())
    Path = userCacheDir();
  if (Path.empty())
    fail("neither XDG_CACHE_HOME nor HOME is set; pass --out");
  if (sys::fs::is_directory(Path) || StringRef(Path).ends_with("/") ||
      !StringRef(sys::path::filename(Path)).ends_with(".csv")) {
    if (std::error_code EC = sys::fs::create_directories(Path))
      fail("cannot create " + Path + ": " + EC.message());
    SmallString<256> P(Path);
    sys::path::append(P, "cm_" + D.Arch + "_" + D.Name + ".csv");
    return std::string(P);
  }
  sys::fs::create_directories(sys::path::parent_path(Path));
  return Path;
}

/// The arms --only takes.
static constexpr StringLiteral Arms[] = {
    "ops",  "herbie-platform", "ozaki",   "ozaki-rect",
    "tcec", "direct",          "inkernel"};

/// Whether `Arm` measures a host as well as a GPU.
static bool isHostArm(StringRef Arm) {
  return Arm == "ops" || Arm == "herbie-platform";
}

/// Whether this run measures `Arm`.
static bool wants(StringRef Arm) {
  if (Only.empty())
    return true;
  SmallVector<StringRef, 8> Asked;
  StringRef(Only).split(Asked, ',', -1, false);
  return llvm::is_contained(Asked, Arm);
}

/// The interpreter the Python arms run under.
static std::string python() {
  if (!Python.empty() &&
      (Python.getNumOccurrences() || sys::fs::exists(Python)))
    return Python;
  auto P = sys::findProgramByName("python3");
  if (!P)
    fail("no Python 3 interpreter: this build was configured without one and "
         "python3 is not on PATH; pass --python <path>");
  return *P;
}

/// The calibrator `C` that measures `D`, installed next to this tool.
static std::string tool(const Device &D, Calibrator C) {
  return sibling(calibratorName(C, D.Hip));
}

/// Whether `Arch` is a GPU's; anything else is a host's.
static bool isGpuArch(StringRef Arch) {
  return Arch.starts_with("sm_") || Arch.starts_with("gfx");
}

/// The Herbie platform of the model `Csv`, next to it and named after it, so
/// that two devices' models in one directory keep apart.
static std::string platformPathFor(StringRef Csv) {
  StringRef Stem = Csv.ends_with(".csv") ? Csv.drop_back(4) : Csv;
  return Stem.str() + ".herbie.rkt";
}

/// Writes the Herbie platform of a GPU model from `Csv` alone, so that Herbie
/// ranks rewrites by the prices the DP solves with.
static void armHerbiePlatform(StringRef Csv) {
  CostModelState CM;
  if (Error E = loadCostModel(CM, Csv))
    fail(toString(std::move(E)));
  const std::string &Arch = CM.NativeArch;
  if (Arch.empty())
    fail(Csv + " carries no '# native_arch=' header, so no Herbie platform "
               "can be generated from it");
  if (!isGpuArch(Arch)) {
    outs() << "poseidon-calibrate: herbie-platform: " << Arch
           << " is a host cost model; Herbie keeps its own default platform\n";
    return;
  }
  std::string Script = shipped("tools/calibrate/herbie_platform.py");
  if (!sys::fs::exists(Script))
    fail("the platform generator is missing: " + Script);
  std::string Py = python();
  // The CSV file name carries the device (cm_<arch>_<device>.csv); it is a
  // comment in the platform header, so a generated file says what it is.
  std::string Device;
  StringRef Name = sys::path::stem(Csv);
  if (Name.consume_front("cm_" + Arch + "_"))
    Device = Name.str();
  std::string Out = platformPathFor(Csv);
  SmallVector<StringRef, 10> Argv = {Py, Script, "--csv", Csv, "--output", Out};
  if (!Herbie.empty()) {
    Argv.push_back("--herbie-binary");
    Argv.push_back(Herbie);
  }
  if (!Device.empty()) {
    Argv.push_back("--device");
    Argv.push_back(Device);
  }
  if (run(Argv))
    fail("the Herbie platform generator failed on " + Csv);
  outs() << "poseidon-calibrate: wrote " << Out << '\n';
}

/// Measures the host `D` with ops_cpu.py into `Part`, and returns its exit
/// status.
static int measureHost(const Device &D, StringRef Part) {
  std::string Py = python();
  std::optional<std::string> Missing =
      capture({Py, "-c",
               "import importlib.util as u, sys; print(' '.join(n for n in "
               "sys.argv[1:] if u.find_spec(n) is None))",
               "llvmlite"});
  if (!Missing)
    fail("cannot run " + Py + "; pass --python <interpreter>");
  if (StringRef Package = StringRef(*Missing).trim(); !Package.empty())
    fail(Py + " cannot import the Python package " + Package +
         ", which the CPU microbenchmark needs; install it (pip install " +
         Package + ") or pass --python <interpreter>");
  std::string Script = shipped("tools/calibrate/ops_cpu.py");
  if (!sys::fs::exists(Script))
    fail("the CPU microbenchmark is missing: " + Script);
  std::string Iterations = utostr(CpuIterations);
  SmallVector<StringRef, 12> Argv = {Py,     Script,     "--cpu",
                                     D.Arch, "--device", D.Model};
  if (!D.Features.empty()) {
    Argv.push_back("--features");
    Argv.push_back(D.Features);
  }
  if (CpuIterations.getNumOccurrences()) {
    Argv.push_back("--iterations");
    Argv.push_back(Iterations);
  }
  if (CpuChainOnly)
    Argv.push_back("--chain-only");
  return run(Argv, Part);
}

// A GPU is measured kOpsRuns times and each row is the median of its runs:
// a run another process or the host stretched is outvoted, and a row is never
// the measurement of one disturbed run. A row is written when most runs wrote
// it (a tile row exists only when its GEMM ran faster than the scalar one).
constexpr unsigned kOpsRuns = 3;

static std::string medianOfRuns(ArrayRef<std::string> Runs) {
  std::string Out;
  SmallVector<std::string, 128> Order;
  std::map<std::string, SmallVector<double, 4>> Values;
  for (const std::string &Run : Runs) {
    bool First = &Run == &Runs.front();
    for (StringRef Line : llvm::split(Run, '\n')) {
      Line = Line.trim();
      if (Line.empty())
        continue;
      if (Line.starts_with("#")) {
        if (First)
          (Out += Line) += '\n';
        continue;
      }
      auto [Key, Value] = Line.rsplit(',');
      double V;
      if (Value.getAsDouble(V))
        fail("the microbenchmark printed the line '" + Line +
             "', which is not a row");
      auto [It, New] = Values.try_emplace(Key.str());
      if (New)
        Order.push_back(Key.str());
      It->second.push_back(V);
    }
  }
  for (const std::string &Key : Order) {
    SmallVector<double, 4> &V = Values[Key];
    if (2 * V.size() <= Runs.size())
      continue;
    llvm::sort(V);
    double Median = V.size() % 2 ? V[V.size() / 2]
                                 : (V[V.size() / 2 - 1] + V[V.size() / 2]) / 2;
    if (V.back() > 1.1 * V.front()) {
      outs() << "poseidon-calibrate: the runs differ by more than 10% on "
             << Key << ":";
      for (double X : V)
        outs() << ' ' << format("%g", X);
      outs() << "; the median is written\n";
    }
    raw_string_ostream(Out) << Key << ',' << format("%.9g", Median) << '\n';
  }
  return Out;
}

static bool isUsableModel(StringRef Rows) {
  return Rows.contains("\nfmul,double,") && Rows.contains("# native_arch=");
}

/// Measures the per-op rows into a new `Csv`.
static void armOps(StringRef Csv, const Device &D) {
  if (sys::fs::exists(Csv) && !Force) {
    outs() << "poseidon-calibrate: per-op throughputs already measured: " << Csv
           << '\n';
    return;
  }
  // The rows land in <csv>.part; the file takes the model's name once the
  // runs have written a usable model.
  std::string Part = (Csv + ".part").str();
  std::string Rows;
  if (D.Host) {
    outs() << "poseidon-calibrate: measuring per-op throughput\n";
    Rows = measureHost(D, Part) == 0 ? readFile(Part) : std::string();
  } else {
    SmallVector<std::string, kOpsRuns> Runs;
    for (unsigned I = 1; I <= kOpsRuns; ++I) {
      outs() << "poseidon-calibrate: measuring per-op throughput, run " << I
             << " of " << kOpsRuns << '\n';
      outs().flush();
      // The redirection does not truncate the file it writes.
      sys::fs::remove(Part);
      std::string Run = run({tool(D, Calibrator::Ops)}, Part) == 0
                            ? readFile(Part)
                            : std::string();
      if (!isUsableModel(Run))
        break;
      Runs.push_back(std::move(Run));
    }
    if (Runs.size() == kOpsRuns) {
      Rows = medianOfRuns(Runs);
      writeFile(Part, Rows);
    }
  }
  if (!isUsableModel(Rows)) {
    sys::fs::remove(Part);
    fail("the microbenchmark produced no usable cost model");
  }
  if (std::error_code EC = sys::fs::rename(Part, Csv))
    fail("cannot write " + Csv + ": " + EC.message());
  outs() << "poseidon-calibrate: wrote " << Csv << '\n';
}

/// Merges `Rows`, which an arm's run printed, into `Csv`; a failed run or one
/// with no row of `Family` is fatal.
static void writeRows(StringRef Csv, StringRef Arm, StringRef Family,
                      const std::optional<std::string> &Rows) {
  if (!Rows)
    fail("the " + Arm + " calibration failed");
  if (!hasFamily(*Rows, Family))
    fail(Arm + " calibration produced no " + Family + " row");
  std::string Content = readFile(Csv);
  mergeRows(Content, *Rows);
  writeFile(Csv, Content);
  for (StringRef Line : llvm::split(*Rows, '\n'))
    if (!Line.trim().empty())
      outs() << "  " << Line.trim() << '\n';
}

/// Whether `Csv` has `Family` and --force does not ask for it again.
static bool alreadyMeasured(StringRef Csv, StringRef Arm, StringRef Family) {
  if (Force || !hasFamily(readFile(Csv), Family))
    return false;
  outs() << "poseidon-calibrate: " << Arm << ": already measured\n";
  return true;
}

/// Measures the dispatch family of `Arm`, which names it to the dispatch
/// calibrator.
static void armLibrary(StringRef Csv, const Device &D, StringRef Arm,
                       StringRef Family) {
  if (alreadyMeasured(Csv, Arm, Family))
    return;
  outs() << "poseidon-calibrate: measuring " << Arm << " library call\n";
  writeRows(
      Csv, Arm, Family,
      capture({tool(D, Calibrator::Library), Arm, utostr(RefN), utostr(Reps)}));
}

/// The extents `Shape` spells as `Count` positive integers joined by 'x'.
static std::optional<SmallVector<unsigned, 3>> extentsOf(StringRef Shape,
                                                         unsigned Count) {
  SmallVector<StringRef, 3> Dims;
  Shape.split(Dims, 'x');
  if (Dims.size() != Count)
    return std::nullopt;
  SmallVector<unsigned, 3> Extents;
  for (StringRef Dim : Dims) {
    unsigned Extent;
    if (Dim.getAsInteger(10, Extent) || !Extent)
      return std::nullopt;
    Extents.push_back(Extent);
  }
  return Extents;
}

namespace {
/// A matrix product --product names.
struct ProductShape {
  unsigned M, N, K;
  /// The layout of B along K its in-kernel rows are measured at, one of
  /// BLayouts; empty when they are not measured.
  std::string Layout;
  /// "MxNxK".
  std::string extents() const { return formatv("{0}x{1}x{2}", M, N, K); }
  /// "m<M>n<N>k<K>", as row keys spell the extents.
  std::string key() const { return formatv("m{0}n{1}k{2}", M, N, K); }
};
} // namespace

/// The layouts of B along K, strided and contiguous, as the in-kernel rows of
/// a product name them.
static constexpr StringLiteral BLayouts[] = {"browmajor", "bcolmajor"};

/// The products --product names.
static SmallVector<ProductShape, 4> products() {
  SmallVector<StringRef, 4> Items;
  StringRef(Product).split(Items, ',', -1, false);
  SmallVector<ProductShape, 4> Out;
  for (StringRef Item : Items) {
    auto [Shape, Layout] = Item.trim().split(':');
    std::optional<SmallVector<unsigned, 3>> E = extentsOf(Shape, 3);
    if (!E || (Item.contains(':') && !llvm::is_contained(BLayouts, Layout)))
      fail("--product: '" + Item +
           "' is not a product MxNxK or MxNxK:<layout>, <layout> browmajor or "
           "bcolmajor");
    Out.push_back({(*E)[0], (*E)[1], (*E)[2], Layout.str()});
  }
  return Out;
}

/// Measures the Ozaki-II and native DGEMM rows of the --product extents and
/// the Ozaki-II rows of the --ozaki-rect (M, K). Rows of other shapes are
/// kept.
static void armOzakiRect(StringRef Csv, const Device &D) {
  // Each shape as the library calibrator takes it, and the key of its rows.
  SmallVector<std::pair<std::string, std::string>, 8> Shapes;
  for (const ProductShape &P : products())
    Shapes.push_back({P.extents(), P.key()});
  SmallVector<StringRef, 4> Items;
  StringRef(OzakiRect).split(Items, ',', -1, false);
  for (StringRef Item : Items) {
    std::optional<SmallVector<unsigned, 3>> E = extentsOf(Item.trim(), 2);
    if (!E)
      fail("--ozaki-rect: '" + Item +
           "' is not a shape MxK; the rows of a product of constant extents "
           "are measured with --product MxNxK");
    Shapes.push_back({formatv("{0}x{1}", (*E)[0], (*E)[1]).str(),
                      formatv("m{0}k{1}", (*E)[0], (*E)[1]).str()});
  }
  if (Shapes.empty()) {
    outs() << "poseidon-calibrate: ozaki-rect: no --product or --ozaki-rect "
              "shapes, "
              "nothing to measure\n";
    return;
  }
  std::string Content = readFile(Csv);
  SmallVector<std::string, 8> Argv = {tool(D, Calibrator::Library), "ozaki",
                                      utostr(RefN), utostr(Reps)};
  const size_t Fixed = Argv.size();
  for (const auto &[Arg, Key] : Shapes) {
    // A shape counts as measured when its first Ozaki-II row is in the model.
    if (!Force &&
        StringRef(Content).contains("nm" + utostr(kOzakiIIModuliCounts[0]) +
                                    "_" + Key + ",")) {
      outs() << "  = " << Arg << " already measured\n";
      continue;
    }
    if (!llvm::is_contained(Argv, Arg))
      Argv.push_back(Arg);
  }
  if (Argv.size() == Fixed)
    return;
  outs() << "poseidon-calibrate: measuring ozaki-rect library call\n";
  SmallVector<StringRef, 8> Args(Argv.begin(), Argv.end());
  writeRows(Csv, "ozaki-rect", "library_ozaki_rect_rel", capture(Args));
}

/// Measures the error-corrected library call, which only a build with cuMpSGEMM
/// has.
static void armTcec(StringRef Csv, const Device &D) {
  if (D.Hip) {
    outs() << "poseidon-calibrate: tcec: the error-corrected library call has "
              "no HIP "
              "runtime; its row stays absent\n";
    return;
  }
  if (!POSEIDON_CALIBRATE_TCEC) {
    outs() << "poseidon-calibrate: tcec: this build has no error-corrected "
              "library call, "
              "which needs cuMpSGEMM (configure with "
              "-DPOSEIDON_CUMPSGEMM_ROOT=<checkout>); its row stays absent\n";
    return;
  }
  armLibrary(Csv, D, "tcec", "library_tcec_rel");
}

/// The time `Exe` reports for --reps runs of its kernel, if it runs.
static std::optional<double> timeOf(StringRef Exe) {
  std::optional<std::string> Output = capture({Exe, utostr(Reps)});
  SmallVector<StringRef, 2> Match;
  double Ms;
  if (!Output || !Regex("time_ms = ([0-9.]+)").match(*Output, &Match) ||
      Match[1].getAsDouble(Ms) || Ms <= 0)
    return std::nullopt;
  return Ms;
}

/// The in-kernel classes the explanations a compile wrote into `Dir` propose,
/// each with the id of the candidate that applies it; none when it wrote none.
static std::map<std::string, std::string> inKernelClasses(StringRef Dir) {
  std::map<std::string, std::string> Classes;
  std::error_code EC;
  for (sys::fs::directory_iterator It(Dir, EC), End; It != End && !EC;
       It.increment(EC)) {
    if (!StringRef(It->path()).ends_with(".explain.json"))
      continue;
    Expected<json::Value> Doc = json::parse(readFile(It->path()));
    if (!Doc)
      fail("cannot read the explanation " + It->path() + ": " +
           toString(Doc.takeError()));
    const json::Object *Root = Doc->getAsObject();
    const json::Array *Units = Root ? Root->getArray("subgraphs") : nullptr;
    if (!Units)
      continue;
    for (const json::Value &Unit : *Units) {
      const json::Object *U = Unit.getAsObject();
      const json::Array *Alts = U ? U->getArray("alternatives") : nullptr;
      if (!Alts)
        continue;
      for (const json::Value &Alt : *Alts) {
        const json::Object *A = Alt.getAsObject();
        std::optional<StringRef> Class =
            A ? A->getString("inkernel_class") : std::nullopt;
        std::optional<StringRef> Id = A ? A->getString("id") : std::nullopt;
        if (Class && Id)
          Classes.emplace(Class->str(), Id->str());
      }
    }
  }
  if (EC && EC != std::errc::no_such_file_or_directory)
    fail("cannot list " + Dir + ": " + EC.message());
  return Classes;
}

/// The largest extent the in-kernel harness is profiled at: reverse-mode AD of
/// a long reduction does not survive at the timed size, and the profile
/// decides only which candidates exist, never what they measure.
constexpr unsigned kProfileExtent = 128;

/// Measures the in-kernel rows of the harness at `M` x `N` x `K`, B
/// column-major when `BColMajor`, each row keyed by its class and `Suffix`,
/// into `Csv`, and returns how many it wrote: the harness is profiled at small
/// extents, the classes the pass proposes from that profile are listed, and
/// the scalar-FP64 loop and each class applied are timed. The row of a class
/// that does not build or run is added to Unmeasured.
static unsigned measureInKernel(const std::string &Csv, const Device &D,
                                StringRef Work, unsigned M, unsigned N,
                                unsigned K, bool BColMajor, StringRef Suffix) {
  std::string Content = readFile(Csv);
  std::string Src = shipped("tools/calibrate/inkernel.cu");
  if (!sys::fs::exists(Src))
    fail("the in-kernel harness source is missing: " + Src);
  const std::string Shape = formatv("{0}x{1}x{2}", M, N, K);
  const std::string Dir =
      (Work + "/" + (Suffix.empty() ? "class" : Suffix.drop_front())).str();
  if (std::error_code EC = sys::fs::create_directories(Dir))
    fail("cannot create " + Dir + ": " + EC.message());
  outs() << "poseidon-calibrate: in-kernel raises at " << Shape << ", B "
         << (BColMajor ? "contiguous" : "strided") << " along K\n";

  std::string Clang = sibling("poseidon-clang++");
  auto defines = [&](unsigned EM, unsigned EN, unsigned EK) {
    return SmallVector<std::string, 16>{
        "-DCAL_M=" + utostr(EM), "-DCAL_N=" + utostr(EN),
        "-DCAL_K=" + utostr(EK), "-DCAL_B_COL_MAJOR=" + utostr(BColMajor)};
  };
  SmallVector<std::string, 16> Timed = defines(M, N, K);
  auto build = [&](ArrayRef<std::string> Flags, StringRef Exe, StringRef Log) {
    SmallVector<std::string, 32> Argv = {Clang,  "-x",
                                         "cuda", "--cuda-gpu-arch=" + D.Arch,
                                         "-O3",  "-Wno-unknown-cuda-version"};
    Argv.append(Flags.begin(), Flags.end());
    Argv.append({Src, "-o", Exe.str()});
    SmallVector<StringRef, 32> Args(Argv.begin(), Argv.end());
    return run(Args, Log, Log);
  };

  // The profile, at small extents with B in the same layout.
  std::string ProfDir = Dir + "/profile", ProfExe = Dir + "/profile.exe",
              ProfLog = Dir + "/profile.log";
  SmallVector<std::string, 16> ProfFlags =
      defines(std::min(M, kProfileExtent), std::min(N, kProfileExtent),
              std::min(K, kProfileExtent));
  ProfFlags.push_back("-poseidon-profile-generate=" + ProfDir);
  if (build(ProfFlags, ProfExe, ProfLog))
    fail("the profiling build of the in-kernel harness failed; see " + ProfLog);
  if (run({ProfExe, "--profiling"}, ProfLog, ProfLog))
    fail("the profiling run of the in-kernel harness failed; see " + ProfLog);

  // The scalar-FP64 baseline: the denominator every rel row shares. Compiled
  // by the same driver, with no action asked of the pass, so the only
  // difference from a candidate is the rewrite.
  std::string BaseExe = Dir + "/base.exe", BaseLog = Dir + "/build_base.log";
  if (build(Timed, BaseExe, BaseLog))
    fail("the scalar-FP64 baseline did not build; see " + BaseLog);
  std::optional<double> BaseMs = timeOf(BaseExe);
  if (!BaseMs)
    fail("the scalar-FP64 baseline produced no time");
  outs() << "poseidon-calibrate: scalar-FP64 baseline "
         << formatv("{0:F4}", *BaseMs) << " ms\n";

  // One compile explains the candidates; -poseidon-pin is mandatory under
  // -poseidon-matmul-calibration, so a placeholder price can never reach a DP
  // solve.
  SmallVector<std::string, 16> Common = Timed;
  Common.append({"-poseidon-profile-use=" + ProfDir,
                 "-poseidon-cache=" + Dir + "/cache",
                 "-poseidon-cost-model=" + Csv, "-poseidon-enable-algebraic=0",
                 "-poseidon-enable-precision=0", "-poseidon-enable-matmul",
                 "-poseidon-matmul-calibration"});
  auto compile = [&](ArrayRef<std::string> Apply, StringRef Exe,
                     StringRef Log) {
    SmallVector<std::string, 16> Flags = Common;
    Flags.append(Apply.begin(), Apply.end());
    return build(Flags, Exe, Log);
  };
  std::string EnumLog = Dir + "/enumerate.log", Explained = Dir + "/explain";
  sys::fs::remove_directories(Explained);
  compile({"-poseidon-pin=M0_0", "-poseidon-explain=" + Explained}, "/dev/null",
          EnumLog);
  std::map<std::string, std::string> Classes = inKernelClasses(Explained);
  if (Classes.empty())
    fail("no in-kernel class was proposed on this device; see " + EnumLog);
  outs() << "poseidon-calibrate: " << Classes.size()
         << " in-kernel class(es) proposed\n";

  std::string Rows;
  for (const auto &[Class, Id] : Classes) {
    const std::string Row = "inkernel_rel," + Class + Suffix.str();
    if (!Force && StringRef(Content).contains(Row + ",")) {
      outs() << "  = " << Class << Suffix << " already calibrated\n";
      continue;
    }
    std::string Exe = Dir + "/cand_" + Class + ".exe";
    std::string Log = Dir + "/build_" + Class + ".log";
    if (compile({"-poseidon-pin=" + Id}, Exe, Log)) {
      outs() << "  ! " << Class << Suffix << " did not build (" << Log << ")\n";
      Unmeasured.push_back(Row);
      continue;
    }
    std::optional<double> Ms = timeOf(Exe);
    if (!Ms) {
      outs() << "  ! " << Class << Suffix << " did not run\n";
      Unmeasured.push_back(Row);
      continue;
    }
    Rows += Row + "," + decimal(*Ms / *BaseMs) + "\n";
    outs() << "  + " << Class << Suffix << ' ' << formatv("{0:F4}", *Ms)
           << " ms  rel " << decimal(*Ms / *BaseMs) << '\n';
  }
  // Each shape's rows are written as soon as they are measured, so that a run
  // that stops at a later shape keeps them.
  if (Rows.empty())
    return 0;
  mergeRows(Content, Rows);
  writeFile(Csv, Content);
  return llvm::count(Rows, '\n');
}

/// Measures the in-kernel rows: of each class on the square harness at --n,
/// B strided along K, and of each --product that names a layout at its
/// extents and layout.
static void armInKernel(const std::string &Csv, const Device &D) {
  if (D.Hip) {
    outs() << "poseidon-calibrate: inkernel: tensor-core raises inside a "
              "kernel are "
              "NVPTX-only; their rows stay absent\n";
    return;
  }
  SmallString<256> WorkBuf;
  if (std::error_code EC =
          sys::fs::createUniqueDirectory("poseidon-inkernel", WorkBuf))
    fail("cannot create " + WorkBuf + ": " + EC.message());
  std::string Work(WorkBuf);
  outs() << "poseidon-calibrate: in-kernel workdir " << Work << '\n';

  unsigned Written = measureInKernel(Csv, D, Work, RefN, RefN, RefN,
                                     /*BColMajor=*/false, "");
  for (const ProductShape &P : products())
    if (!P.Layout.empty())
      Written +=
          measureInKernel(Csv, D, Work, P.M, P.N, P.K, P.Layout == BLayouts[1],
                          "_" + P.key() + "_" + P.Layout);
  if (!Written)
    outs() << "poseidon-calibrate: in-kernel: nothing new to write\n";
}

/// Resolves the model as a compile for `D` does, with --out in the place of
/// -poseidon-cost-model, and reports which row families it lacks.
static int checkModel(const Device &D) {
  CostModelState CM;
  Error E = requireCostModel(CM, Out.empty() ? std::string() : resolveCsv(D),
                             D.Arch, "this device", !D.Host);
  if (!CM.Loaded) {
    printError(toString(std::move(E)));
    return 1;
  }
  int Bad = 0;
  outs() << "cost model  " << costModelPath(CM) << '\n';
  outs() << "device      " << D.Name << " (" << D.Arch;
  if (!D.Host)
    outs() << ", id " << D.Id;
  outs() << ")\n";
  if (E) {
    printError(toString(std::move(E)));
    ++Bad;
  } else {
    outs() << "native_arch " << D.Arch << "  ok\n";
  }
  if (!D.Host) {
    std::string Platform = platformPathFor(costModelPath(CM));
    bool Have = sys::fs::exists(Platform);
    outs() << (Have ? "have        " : "MISSING     ") << Platform
           << "  (Herbie algebraic-search platform; regenerate with "
              "--only herbie-platform)\n";
    if (!Have)
      ++Bad;
  }
  enum Where { Everywhere, Gpu, Nvidia };
  struct {
    const char *Family;
    const char *What;
    Where On;
  } Families[] = {{"fmul", "per-op throughputs", Everywhere},
                  {"wmma_mma_m16n16k16", "tensor-core tile prices", Nvidia},
                  {"library_ozaki_rel", "Ozaki-II library call", Gpu},
                  {"library_tcec_rel", "error-corrected library call", Nvidia},
                  {"library_direct_rel", "reduced-precision library call", Gpu},
                  {"inkernel_rel", "in-kernel tensor-core raises", Nvidia}};
  for (const auto &F : Families) {
    if ((D.Host && F.On != Everywhere) || (D.Hip && F.On == Nvidia)) {
      outs() << "n/a         " << F.Family << "  (" << F.What << ": none on "
             << (D.Host ? "a host" : D.Arch) << ")\n";
      continue;
    }
    if (!POSEIDON_CALIBRATE_TCEC && StringRef(F.Family) == "library_tcec_rel") {
      outs() << "n/a         " << F.Family << "  (" << F.What
             << ": none in a build without cuMpSGEMM)\n";
      continue;
    }
    bool Have = llvm::any_of(getCostModel(CM), [&](const auto &Row) {
      return Row.first.first == F.Family;
    });
    outs() << (Have ? "have        " : "MISSING     ") << F.Family << "  ("
           << F.What << ")\n";
    if (!Have)
      ++Bad;
  }
  if (Bad)
    printError(Twine(Bad) + " problem(s); re-run poseidon-calibrate --force "
                            "on this device");
  return Bad ? 1 : 0;
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(
      argc, argv,
      "measure the Poseidon cost model of a GPU, or with --cpu of this host, "
      "or check an existing one\n\n"
      "Each arm writes its row families into the one CSV:\n"
      "  ops          <op>,<type> and "
      "fpext_/fptrunc_<from>_to_<to>,<from>,\n"
      "                   and on NVIDIA wmma_mma_<tile>,<in>_<acc>\n"
      "  herbie-platform  no rows: the Herbie platform <csv>.herbie.rkt\n"
      "  ozaki            library_ozaki_rel,nm<8..14> and ,dgemm\n"
      "  ozaki-rect       library_ozaki_rect_rel,nm<k>_m<M>n<N>k<K> and\n"
      "                   ,dgemm_m<M>n<N>k<K> (--product),\n"
      "                   ,nm<k>_m<M>k<K> (--ozaki-rect)\n"
      "  tcec             library_tcec_rel,fp16tcec\n"
      "  direct           library_direct_rel,<in>_<acc>\n"
      "  inkernel         inkernel_rel,<class> and\n"
      "                   ,<class>_m<M>n<N>k<K>_<layout> (--product)\n"
      "A host has the first two arms only.\n");
  ToolDir = std::string(sys::path::parent_path(
      sys::fs::getMainExecutable(argv[0], (void *)(intptr_t)main)));

  SmallVector<StringRef, 8> Asked;
  StringRef(Only).split(Asked, ',', -1, false);
  for (StringRef Arm : Asked) {
    if (!llvm::is_contained(Arms, Arm))
      fail("--only: there is no arm '" + Arm + "'");
    if (Cpu.getNumOccurrences() && !isHostArm(Arm))
      fail("the " + Arm + " arm measures a GPU, and --cpu a host");
  }
  if (Cpu.getNumOccurrences() && GpuId.getNumOccurrences())
    fail("--cpu measures this host and --gpu a GPU; pass one of them");
  if (Cpu.getNumOccurrences() && Cpu.empty())
    fail("--cpu needs a target-cpu: native, or a name -march= takes");
  // A malformed product is reported before any arm runs.
  products();

  // The platform arm reads a CSV and writes a file beside it. Asking for it
  // alone, against a named CSV, therefore needs no device present: that is how
  // a model measured on one machine gets its platform on another.
  if (Only == "herbie-platform" && StringRef(Out).ends_with(".csv")) {
    if (!sys::fs::exists(Out))
      fail("no cost model at " + Out);
    armHerbiePlatform(Out);
    return 0;
  }

  Device D = Cpu.getNumOccurrences() ? queryHost() : queryDevice();

  if (Check)
    return checkModel(D);
  std::string Csv = resolveCsv(D);

  outs() << "poseidon-calibrate: device " << D.Name << " (" << D.Arch;
  if (!D.Host)
    outs() << ", id " << D.Id;
  outs() << ")\n";
  if (wants("ops"))
    armOps(Csv, D);
  if (!sys::fs::exists(Csv))
    fail("no cost model at " + Csv +
         "; run the ops arm before the library arms");
  // Right after ops, because it translates exactly what ops wrote.
  if (wants("ops") || wants("herbie-platform"))
    armHerbiePlatform(Csv);
  if (!D.Host) {
    if (wants("ozaki"))
      armLibrary(Csv, D, "ozaki", "library_ozaki_rel");
    if (wants("ozaki-rect"))
      armOzakiRect(Csv, D);
    if (wants("tcec"))
      armTcec(Csv, D);
    if (wants("direct"))
      armLibrary(Csv, D, "direct", "library_direct_rel");
    if (wants("inkernel"))
      armInKernel(Csv, D);
  }
  outs() << "poseidon-calibrate: cost model: " << Csv << '\n';
  if (!Unmeasured.empty()) {
    printError(Twine(Unmeasured.size()) +
               " row(s) not measured, which the same command measures alone "
               "when run again: " +
               join(Unmeasured, ", "));
    return 1;
  }
  return 0;
}
