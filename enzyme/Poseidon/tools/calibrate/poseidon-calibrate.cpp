//===- poseidon-calibrate.cpp - measure a device's cost model -------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Poseidon prices every candidate from one CSV measured on the device the code
// will run on. This tool measures it, and is to Poseidon what llvm-profdata is
// to clang PGO: run it once per GPU or host CPU, and every later compile finds
// the model by the module's target-cpu.
//
//===----------------------------------------------------------------------===//

#include "Core/CostTable.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
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
                 "directory, which takes cm_<target-cpu>_<device>.csv "
                 "(default: $XDG_CACHE_HOME/poseidon)"));
static cl::opt<bool>
    Check("check", cl::desc("Validate the resolved cost model against this "
                            "device instead of measuring"));
static cl::opt<bool>
    Force("force", cl::desc("Re-measure families that are already present"));
static cl::opt<std::string>
    Only("only", cl::init(""), cl::value_desc("arms"),
         cl::desc("Comma-separated subset of "
                  "microbm,ozaki,ozaki-rect,tcec,direct,inkernel,"
                  "herbie-platform"));
static cl::opt<std::string> ProfileDir(
    "profile", cl::init(""), cl::value_desc("dir"),
    cl::desc("Profile of a GEMM site the in-kernel arm enumerates its "
             "candidates from (required by that arm)"));
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
static cl::opt<std::string> OzakiRect(
    "ozaki-rect", cl::init(""), cl::value_desc("MxK|MxNxK,..."),
    cl::desc("(M, K) of the M x Ncols x K host GEMMs whose Ozaki-II rows the "
             "ozaki-rect arm measures; MxNxK instead measures "
             "the native DGEMM row of a call with exactly those extents"));
static cl::opt<unsigned>
    RefN("n", cl::init(2048), cl::value_desc("N"),
         cl::desc("Square reference shape the dispatch arms and "
                  "the in-kernel arm are timed at"));
static cl::opt<unsigned>
    Reps("reps", cl::init(30), cl::value_desc("count"),
         cl::desc("Timed repetitions per dispatch measurement"));

static std::string ToolDir;

static std::string sibling(StringRef Name) {
  SmallString<256> P(ToolDir);
  sys::path::append(P, Name);
  return std::string(P);
}

// share/poseidon of the installation this tool was started from, and the
// source tree it was built from when it runs out of its build directory. The
// two layouts carry the same names below that point.
static std::string shipped(StringRef Rel) {
  SmallString<256> P(sys::path::parent_path(ToolDir));
  sys::path::append(P, "share/poseidon", Rel);
  if (sys::fs::exists(P))
    return std::string(P);
  SmallString<256> S(POSEIDON_CALIBRATE_SRCDIR);
  sys::path::append(S, Rel);
  return std::string(S);
}

static std::string num(double V) { return formatv("{0:F6}", V).str(); }

[[noreturn]] static void fail(const Twine &Msg) {
  errs() << "poseidon-calibrate: " << Msg << '\n';
  std::exit(1);
}

// Runs a program, optionally capturing its stdout into a file.
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

static std::string readFile(StringRef Path) {
  auto Buf = MemoryBuffer::getFile(Path);
  if (!Buf)
    fail("cannot read " + Path + ": " + Buf.getError().message());
  return (*Buf)->getBuffer().str();
}

// Writes `Path` whole or not at all: an interrupted run leaves the model it
// started from.
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

static std::string capture(ArrayRef<StringRef> Argv,
                           StringRef StderrFile = "") {
  SmallString<128> Tmp;
  if (sys::fs::createTemporaryFile("poseidon-calibrate", "out", Tmp))
    fail("cannot create a temporary file");
  int RC = run(Argv, Tmp, StderrFile);
  std::string S = RC == 0 ? readFile(Tmp) : std::string();
  sys::fs::remove(Tmp);
  return S;
}

static bool hasFamily(StringRef Csv, StringRef Family) {
  for (StringRef Line : llvm::split(Csv, '\n'))
    if (Line.starts_with(Family) &&
        Line.drop_front(Family.size()).starts_with(","))
      return true;
  return false;
}

// The (family, key) a row is identified by.
static std::pair<StringRef, StringRef> rowKey(StringRef Line) {
  auto [Family, Rest] = Line.split(',');
  return {Family, Rest.split(',').first};
}

// Replaces the rows whose (family, key) the new rows carry and keeps the
// others.
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
struct Device {
  std::string Id;
  std::string Name; // punctuation stripped, as it appears in the file name
  std::string Arch; // sm_<cc>, gfx<NNN> on an AMD GPU, a CPU's target-cpu
  bool Hip = false;
  bool Host = false;
  std::string Model;    // a host's model name, as the system reports it
  std::string Features; // the host's features, when Arch is its own CPU
};
} // namespace

// The GPU --gpu names, else the first one the vendor's visible-devices variable
// lists: the microbenchmark of each vendor this build has, CUDA's first, names
// it with --device.
static Device queryDevice() {
  // What each microbenchmark said when it named no device.
  std::string Reasons;
  for (bool Hip : {false, true}) {
    std::string Exe =
        sibling(Hip ? "poseidon-microbm-hip" : "poseidon-microbm");
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
    SmallString<128> Err;
    if (sys::fs::createTemporaryFile("poseidon-calibrate", "err", Err))
      fail("cannot create a temporary file");
    std::string O = capture({Exe, "--device"}, Err);
    std::string Why = readFile(Err);
    sys::fs::remove(Err);
    auto [NameRef, Arch] = StringRef(O).split('\n').first.split('\t');
    if (Arch.trim().empty()) {
      if (!StringRef(Why).trim().empty())
        Reasons +=
            ("\n  " + sys::path::filename(Exe) + ": " + StringRef(Why).trim())
                .str();
      continue;
    }
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

// The model name the system reports for this host's CPU, or empty.
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

// The host measured for the target-cpu --cpu names. A compile for `native`
// takes the CPU name and features LLVM detects here, so the microbenchmarks are
// compiled for them; a named CPU takes that CPU's own features, as -march=
// does. The model name distinguishes machines with the same target-cpu, and a
// system that reports none names the device by its detected CPU.
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

static constexpr StringLiteral Arms[] = {
    "microbm", "herbie-platform", "ozaki",   "ozaki-rect",
    "tcec",    "direct",          "inkernel"};

// The arms that take a host: the per-op microbenchmark, and the platform arm,
// which reads a CSV.
static bool isHostArm(StringRef Arm) {
  return Arm == "microbm" || Arm == "herbie-platform";
}

static bool wants(StringRef Arm) {
  if (Only.empty())
    return true;
  SmallVector<StringRef, 8> Asked;
  StringRef(Only).split(Asked, ',', -1, false);
  return llvm::is_contained(Asked, Arm);
}

// The interpreter the Python arms run under.
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

// The calibrator binary that measures this device: the HIP build of it on an
// AMD GPU.
static std::string tool(const Device &D, StringRef Name) {
  return sibling(D.Hip ? (Name + "-hip").str() : Name.str());
}

// The architectures of a GPU cost model; anything else is a host's.
static bool isGpuArch(StringRef Arch) {
  return Arch.starts_with("sm_") || Arch.starts_with("gfx");
}

// ---- the arms -------------------------------------------------------------

// The Herbie platform generated from a CSV lives next to it, named after it:
// one platform per cost model, so a directory holding two devices' models
// cannot mix them up.
static std::string platformPathFor(StringRef Csv) {
  StringRef Stem = Csv.ends_with(".csv") ? Csv.drop_back(4) : Csv;
  return Stem.str() + ".herbie.rkt";
}

// Herbie's algebraic search ranks candidates by a platform cost table. This
// arm writes that table from the same CSV the DP prices from, so a rewrite is
// proposed and priced on one device's numbers. No GPU is touched: the CSV is
// the only input.
static void armHerbiePlatform(StringRef Csv) {
  CostModelState CM(Csv.str());
  if (Error E = loadCostModel(CM))
    fail(toString(std::move(E)));
  const std::string &Arch = CM.NativeArch;
  if (Arch.empty())
    fail(Csv + " carries no '# native_arch=' header, so no Herbie platform "
               "can be generated from it");
  if (!isGpuArch(Arch)) {
    outs() << "[calibrate] herbie-platform: " << Arch
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
  outs() << "[calibrate] wrote " << Out << '\n';
}

// A host is measured by microbm_cpu.py, which JIT-compiles each
// microbenchmark with llvmlite for the CPU and features it is given. Its rows
// go to `Part`.
static int measureHost(const Device &D, StringRef Part) {
  std::string Py = python();
  std::string Missing =
      capture({Py, "-c",
               "import importlib.util as u, sys; print(' '.join(n for n in "
               "sys.argv[1:] if u.find_spec(n) is None))",
               "llvmlite"});
  if (!StringRef(Missing).trim().empty())
    fail(Py + " cannot import the Python package " + StringRef(Missing).trim() +
         ", which the CPU microbenchmark needs; install it (pip install " +
         StringRef(Missing).trim() + ") or pass --python <interpreter>");
  std::string Script = shipped("tools/calibrate/microbm_cpu.py");
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

static void armMicrobm(StringRef Csv, const Device &D) {
  if (sys::fs::exists(Csv) && !Force) {
    outs() << "[calibrate] per-op throughputs already measured: " << Csv
           << '\n';
    return;
  }
  outs() << "[calibrate] measuring per-op throughput (several minutes)\n";
  // The rows land in <csv>.part as they are measured; the file takes the
  // model's name once the run has written a usable model.
  std::string Part = (Csv + ".part").str();
  int RC =
      D.Host ? measureHost(D, Part) : run({tool(D, "poseidon-microbm")}, Part);
  std::string Rows = RC == 0 ? readFile(Part) : std::string();
  if (!StringRef(Rows).contains("\nfmul,double,") ||
      !StringRef(Rows).contains("# native_arch=")) {
    sys::fs::remove(Part);
    fail("the microbenchmark produced no usable cost model");
  }
  if (std::error_code EC = sys::fs::rename(Part, Csv))
    fail("cannot write " + Csv + ": " + EC.message());
  outs() << "[calibrate] wrote " << Csv << '\n';
}

static void writeRows(StringRef Csv, StringRef Arm, StringRef Family,
                      StringRef Rows) {
  if (!hasFamily(Rows, Family))
    fail(Arm + " calibration produced no " + Family + " row");
  std::string Content = readFile(Csv);
  mergeRows(Content, Rows);
  writeFile(Csv, Content);
  for (StringRef Line : llvm::split(Rows, '\n'))
    if (!Line.trim().empty())
      outs() << "  " << Line.trim() << '\n';
}

static bool alreadyMeasured(StringRef Csv, StringRef Arm, StringRef Family) {
  if (Force || !hasFamily(readFile(Csv), Family))
    return false;
  outs() << "[calibrate] " << Arm << ": already measured\n";
  return true;
}

// The dispatch calibrator measures the family of the arm it is named.
static void armDispatch(StringRef Csv, const Device &D, StringRef Arm,
                        StringRef Family) {
  if (alreadyMeasured(Csv, Arm, Family))
    return;
  outs() << "[calibrate] measuring " << Arm << " host dispatch\n";
  writeRows(Csv, Arm, Family,
            capture({tool(D, "poseidon-dispatch-calibrate"), Arm, utostr(RefN),
                     utostr(Reps)}));
}

// The ozaki-rect rows are keyed by the shape of the product they price, so
// the shapes come from the caller: MxK measures the Ozaki-II rows of (M, K)
// at the calibrator's large column count, MxNxK the native DGEMM row of a call
// with exactly those extents. Rows of shapes this run does not measure are
// kept.
static void armOzakiRect(StringRef Csv, const Device &D) {
  if (OzakiRect.empty()) {
    outs() << "[calibrate] ozaki-rect: no --ozaki-rect shapes, nothing to "
              "measure\n";
    return;
  }
  std::string Content = readFile(Csv);
  SmallVector<StringRef, 4> Shapes;
  StringRef(OzakiRect).split(Shapes, ',', -1, false);
  SmallVector<std::string, 8> Argv = {tool(D, "poseidon-dispatch-calibrate"),
                                      "ozaki", utostr(RefN), utostr(Reps)};
  SmallVector<std::string, 4> Keys;
  for (StringRef Shape : Shapes) {
    SmallVector<StringRef, 3> Dims;
    Shape.trim().split(Dims, 'x');
    SmallVector<std::string, 3> V;
    for (StringRef Dim : Dims) {
      unsigned X;
      if (Dim.getAsInteger(10, X) || !X)
        break;
      V.push_back(utostr(X));
    }
    if (V.size() != Dims.size() || (V.size() != 2 && V.size() != 3))
      fail("--ozaki-rect: '" + Shape + "' is not a shape MxK or MxNxK");
    std::string Arg = V[0] + "x" + V[1], Key = "_m" + V[0] + "k" + V[1] + ",";
    if (V.size() == 3) {
      Arg += "x" + V[2];
      Key = "_m" + V[0] + "n" + V[1] + "k" + V[2] + ",";
    }
    if (!Force && StringRef(Content).contains(Key)) {
      outs() << "  = " << Shape << " already measured\n";
      continue;
    }
    Argv.push_back(Arg);
    Keys.push_back(Key);
  }
  if (Keys.empty())
    return;
  outs() << "[calibrate] measuring ozaki-rect host dispatch\n";
  SmallVector<StringRef, 8> Args(Argv.begin(), Argv.end());
  writeRows(Csv, "ozaki-rect", "ozaki_dispatch_rect_rel", capture(Args));
}

// The error-corrected dispatch runs on cuMpSGEMM, so a build without it has no
// such candidate for the row to price.
static void armTcec(StringRef Csv, const Device &D) {
  if (D.Hip) {
    outs() << "[calibrate] tcec: the error-corrected dispatch has no HIP "
              "runtime; its row stays absent\n";
    return;
  }
  if (!POSEIDON_CALIBRATE_TCEC) {
    outs() << "[calibrate] tcec: this build has no error-corrected dispatch, "
              "which needs cuMpSGEMM (configure with "
              "-DPOSEIDON_TCEC_USE_CUMPSGEMM=ON "
              "-DPOSEIDON_CUMPSGEMM_ROOT=<checkout>); its row stays absent\n";
    return;
  }
  armDispatch(Csv, D, "tcec", "tcec_dispatch_rel");
}

// The in-kernel classes are the one family the compiler has to be asked about:
// each is a rewrite the pass proposes, so the harness compiles its kernel once
// per class through the plugin and times what a real solve would emit. A class
// the pass cannot propose or build here is reported and left absent.
static void armInKernel(const std::string &Csv, const Device &D) {
  if (D.Hip) {
    outs() << "[calibrate] inkernel: tensor-core raises inside a kernel are "
              "NVPTX-only; their rows stay absent\n";
    return;
  }
  std::string Content = readFile(Csv);
  std::string Src = shipped("tools/calibrate/inkernel_calibrate.cu");
  if (!sys::fs::exists(Src))
    fail("the in-kernel harness source is missing: " + Src);

  if (ProfileDir.empty()) {
    if (Only.empty()) {
      outs() << "[calibrate] inkernel: not measured; it needs --profile <dir> "
                "(see --help)\n";
      return;
    }
    fail("the in-kernel arm needs --profile <dir>: the profile of a GEMM "
         "site written by -poseidon-profile-generate");
  }
  SmallString<256> KernelProf(ProfileDir);
  sys::path::append(
      KernelProf, "preprocess__Z10matmul_optPdPKdS1__poseidon_body.fpprofile");
  if (!sys::fs::exists(KernelProf))
    fail("the in-kernel arm needs " + Twine(sys::path::filename(KernelProf)) +
         ": build tools/calibrate/inkernel_calibrate.cu at a small -DCAL_N "
         "with -poseidon-profile-generate, run it with --profiling, and pass "
         "its profile directory as --profile <dir>");

  SmallString<256> WorkBuf;
  if (std::error_code EC =
          sys::fs::createUniqueDirectory("poseidon-inkernel", WorkBuf))
    fail("cannot create " + WorkBuf + ": " + EC.message());
  std::string Work(WorkBuf);
  outs() << "[calibrate] in-kernel workdir " << Work << '\n';

  std::string Clang = sibling("poseidon-clang++");
  std::string Arch = "--cuda-gpu-arch=" + D.Arch;
  std::string CalN = "-DCAL_N=" + utostr(RefN);
  std::string ProfUse = "-poseidon-profile-use=" + ProfileDir;
  std::string CostModel = "-poseidon-cost-model=" + Csv;
  std::string Cache = "-poseidon-cache=" + Work + "/cache";
  std::string BaseExe = Work + "/base.exe";
  std::string BaseLog = Work + "/build_base.log";
  std::string EnumLog = Work + "/enumerate.log";

  // The scalar-FP64 baseline: the denominator every rel row shares. Compiled
  // by the same driver, with no action asked of the pass, so the only
  // difference from a candidate is the rewrite.
  if (run({Clang, "-x", "cuda", Arch, "-O3", "-Wno-unknown-cuda-version", CalN,
           Src, "-o", BaseExe},
          BaseLog, BaseLog))
    fail("the scalar-FP64 baseline did not build; see " + BaseLog);
  Regex TimeRe("time_ms = ([0-9.]+)");
  auto timeOf = [&](StringRef Exe) -> double {
    std::string O = capture({Exe, utostr(Reps)});
    SmallVector<StringRef, 2> M;
    if (!TimeRe.match(O, &M))
      return 0.0;
    double V = 0;
    M[1].getAsDouble(V);
    return V;
  };
  double BaseMs = timeOf(BaseExe);
  if (BaseMs <= 0)
    fail("the scalar-FP64 baseline produced no time");
  outs() << "[calibrate] scalar-FP64 baseline " << formatv("{0:F4}", BaseMs)
         << " ms\n";

  // One compile prints the candidate table; -poseidon-apply-rewrites is
  // mandatory under -poseidon-inkernel-calibration, so a placeholder price can
  // never reach a DP solve.
  SmallVector<std::string, 32> Common = {Clang,
                                         "-x",
                                         "cuda",
                                         Arch,
                                         "-O3",
                                         "-Wno-unknown-cuda-version",
                                         CalN,
                                         ProfUse,
                                         Cache,
                                         CostModel,
                                         "-poseidon-enable-herbie=0",
                                         "-poseidon-enable-pt=0",
                                         "-poseidon-loose-coverage",
                                         "-poseidon-raise-wmma",
                                         "-poseidon-print",
                                         "-poseidon-inkernel-calibration"};
  auto compile = [&](StringRef Apply, StringRef Exe, StringRef Log) {
    SmallVector<StringRef, 32> Argv(Common.begin(), Common.end());
    Argv.push_back(Apply);
    Argv.push_back(Src);
    Argv.push_back("-o");
    Argv.push_back(Exe);
    return run(Argv, Log, Log);
  };
  compile("-poseidon-apply-rewrites=M0_0", "/dev/null", EnumLog);

  // "[poseidon]   #8 tcec n=2 wmma m16n16k8 tf32/f32  compCost/MAC=..."
  Regex Cand("^\\[poseidon\\][ ]+#([0-9]+) (.*)  compCost/MAC");
  Regex Direct("^wmma m[0-9]+n[0-9]+k[0-9]+ ([a-z0-9]+)/([a-z0-9]+)$");
  Regex Tcec(
      "^tcec n=([0-9]+) wmma m[0-9]+n[0-9]+k[0-9]+ ([a-z0-9]+)/([a-z0-9]+)$");
  std::map<std::string, std::string> Classes; // class -> candidate index
  for (StringRef Line : llvm::split(readFile(EnumLog), '\n')) {
    SmallVector<StringRef, 3> M;
    if (!Cand.match(Line, &M))
      continue;
    StringRef Idx = M[1], Label = M[2].trim();
    SmallVector<StringRef, 4> P;
    std::string Cls;
    if (Direct.match(Label, &P))
      Cls = (P[1] + "_" + P[2]).str();
    else if (Tcec.match(Label, &P))
      Cls = ("tcec_n" + P[1] + "_" + P[2] + "_" + P[3]).str();
    else
      continue; // a host-dispatch family: priced by its own arm
    Classes.emplace(Cls, Idx.str());
  }
  if (Classes.empty())
    fail("no in-kernel class was proposed on this device; see " + EnumLog);
  outs() << "[calibrate] " << Classes.size()
         << " in-kernel class(es) proposed\n";

  std::string Rows;
  for (const auto &[Cls, Idx] : Classes) {
    if (!Force &&
        StringRef(Content).contains("wmma_inkernel_rel," + Cls + ",")) {
      outs() << "  = " << Cls << " already calibrated\n";
      continue;
    }
    std::string Exe = Work + "/cand_" + Cls + ".exe";
    std::string Log = Work + "/build_" + Cls + ".log";
    if (compile("-poseidon-apply-rewrites=M0_" + Idx, Exe, Log)) {
      outs() << "  ! " << Cls << " did not build; left uncalibrated (" << Log
             << ")\n";
      continue;
    }
    double Ms = timeOf(Exe);
    if (Ms <= 0) {
      outs() << "  ! " << Cls << " did not run; left uncalibrated\n";
      continue;
    }
    Rows += "wmma_inkernel_rel," + Cls + "," + num(Ms / BaseMs) + "\n";
    outs() << "  + " << Cls << ' ' << formatv("{0:F4}", Ms) << " ms  rel "
           << num(Ms / BaseMs) << '\n';
  }
  if (Rows.empty()) {
    outs() << "[calibrate] in-kernel: nothing new to write\n";
    return;
  }
  mergeRows(Content, Rows);
  writeFile(Csv, Content);
}

// Resolves the model as a compile for this device does, with --out in the
// place of -poseidon-cost-model, and reports what of it a full run writes.
static int checkModel(const Device &D) {
  CostModelState CM(Out.empty() ? std::string() : resolveCsv(D));
  Error E = requireCostModel(CM, D.Arch, "this device", !D.Host);
  if (!CM.Loaded) {
    errs() << "poseidon-calibrate: " << toString(std::move(E)) << '\n';
    return 1;
  }
  int Bad = 0;
  outs() << "cost model  " << costModelPath(CM) << '\n';
  outs() << "device      " << D.Name << " (" << D.Arch;
  if (!D.Host)
    outs() << ", id " << D.Id;
  outs() << ")\n";
  if (E) {
    errs() << "poseidon-calibrate: " << toString(std::move(E)) << '\n';
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
  } Families[] = {
      {"fmul", "per-op throughputs", Everywhere},
      {"wmma_mma_m16n16k16", "tensor-core tile prices", Nvidia},
      {"ozaki_dispatch_rel", "Ozaki-II host dispatch", Gpu},
      {"tcec_dispatch_rel", "error-corrected host dispatch", Nvidia},
      {"direct_dispatch_rel", "reduced-precision host dispatch", Gpu},
      {"wmma_inkernel_rel", "in-kernel tensor-core raises", Nvidia}};
  for (const auto &F : Families) {
    if ((D.Host && F.On != Everywhere) || (D.Hip && F.On == Nvidia)) {
      outs() << "n/a         " << F.Family << "  (" << F.What << ": none on "
             << (D.Host ? "a host" : D.Arch) << ")\n";
      continue;
    }
    if (!POSEIDON_CALIBRATE_TCEC &&
        StringRef(F.Family) == "tcec_dispatch_rel") {
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
    errs() << "poseidon-calibrate: " << Bad
           << " problem(s); re-run poseidon-calibrate --force on this device\n";
  return Bad ? 1 : 0;
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(
      argc, argv,
      "measure the Poseidon cost model of a GPU, or with --cpu of this host, "
      "or check an existing one\n\n"
      "Each arm writes its row families into the one CSV:\n"
      "  microbm          <op>,<type> and "
      "fpext_/fptrunc_<from>_to_<to>,<from>,\n"
      "                   and on NVIDIA wmma_mma_<tile>,<in>_<acc>\n"
      "  herbie-platform  no rows: the Herbie platform <csv>.herbie.rkt\n"
      "  ozaki            ozaki_dispatch_rel,nm<8..14> and ,dgemm\n"
      "  ozaki-rect       ozaki_dispatch_rect_rel,nm<k>_m<M>k<K> and\n"
      "                   ,dgemm_m<M>n<N>k<K>\n"
      "  tcec             tcec_dispatch_rel,fp16tcec\n"
      "  direct           direct_dispatch_rel,<in>_<acc>\n"
      "  inkernel         wmma_inkernel_rel,<class>\n"
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

  outs() << "[calibrate] device " << D.Name << " (" << D.Arch;
  if (!D.Host)
    outs() << ", id " << D.Id;
  outs() << ")\n";
  if (wants("microbm"))
    armMicrobm(Csv, D);
  if (!sys::fs::exists(Csv))
    fail("no cost model at " + Csv +
         "; run the microbm arm before the dispatch arms");
  // Right after microbm, because it translates exactly what microbm wrote.
  if (wants("microbm") || wants("herbie-platform"))
    armHerbiePlatform(Csv);
  if (!D.Host) {
    if (wants("ozaki"))
      armDispatch(Csv, D, "ozaki", "ozaki_dispatch_rel");
    if (wants("ozaki-rect"))
      armOzakiRect(Csv, D);
    if (wants("tcec"))
      armTcec(Csv, D);
    if (wants("direct"))
      armDispatch(Csv, D, "direct", "direct_dispatch_rel");
    if (wants("inkernel"))
      armInKernel(Csv, D);
  }
  outs() << "[calibrate] cost model: " << Csv << '\n';
  return 0;
}
