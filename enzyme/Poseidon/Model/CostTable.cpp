//===- CostTable.cpp - the measured cost table of one compile -------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CostTable.h"
#include "Targets/MathLibrary.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <cstdlib>
#include <system_error>

using namespace llvm;

namespace poseidon {

// Resolves the CSV the compile prices from to `Requested`, else
// POSEIDON_COST_MODEL; false when neither names one.
static bool resolveFromRequest(CostModelState &CM, StringRef Requested) {
  if (CM.IsResolved)
    return true;
  if (!Requested.empty()) {
    CM.Resolved = Requested.str();
    CM.IsResolved = true;
  } else if (const char *Env = std::getenv("POSEIDON_COST_MODEL")) {
    if (*Env) {
      CM.Resolved = Env;
      CM.IsResolved = true;
    }
  }
  return CM.IsResolved;
}

const std::string &costModelPath(const CostModelState &CM) {
  return CM.Resolved;
}

std::string userCacheDir() {
  SmallString<256> Dir;
  if (!sys::path::cache_directory(Dir))
    return "";
  sys::path::append(Dir, "poseidon");
  return std::string(Dir);
}

// One measured model per device: the file name carries the architecture it was
// measured on, so the module's target-cpu names it.
static Error resolveForTarget(CostModelState &CM, StringRef Requested,
                              StringRef TargetCpu, StringRef FnName,
                              bool OnDevice) {
  if (resolveFromRequest(CM, Requested))
    return Error::success();
  if (TargetCpu.empty())
    return createStringError(inconvertibleErrorCode(),
                             "optimizing " + Twine(FnName) +
                                 " needs a measured cost model, and the module "
                                 "names no target-cpu to look one up by; pass "
                                 "-poseidon-cost-model=<csv>");
  std::string Dir = userCacheDir();
  std::string Prefix = ("cm_" + TargetCpu + "_").str();
  SmallVector<std::string, 4> Found;
  std::error_code EC;
  for (sys::fs::directory_iterator It(Dir, EC), E; It != E && !EC;
       It.increment(EC)) {
    StringRef Name = sys::path::filename(It->path());
    if (Name.starts_with(Prefix) && Name.ends_with(".csv"))
      Found.push_back(It->path());
  }
  if (Found.empty())
    return createStringError(
        inconvertibleErrorCode(),
        "no cost model for target-cpu '" + Twine(TargetCpu) + "' in " +
            (Dir.empty()
                 ? "the user cache, which neither XDG_CACHE_HOME nor HOME "
                   "locates"
                 : Dir) +
            (OnDevice ? "; measure the GPU with 'poseidon-calibrate --gpu <id>'"
                      : "; measure this host with 'poseidon-calibrate --cpu " +
                            TargetCpu + "'") +
            ", or name a model with -poseidon-cost-model=<csv> or "
            "POSEIDON_COST_MODEL");
  if (Found.size() > 1) {
    std::string List;
    for (const std::string &P : Found)
      List += (List.empty() ? "" : ", ") + P;
    return createStringError(inconvertibleErrorCode(),
                             "several cost models for target-cpu '" +
                                 Twine(TargetCpu) + "': " + List +
                                 "; name the one to price from with "
                                 "-poseidon-cost-model=<csv>");
  }
  CM.Resolved = Found.front();
  CM.IsResolved = true;
  return Error::success();
}

// A price comes from the model or not at all: a part of the model that is
// missing or unreadable stops the compile, naming the model.
static Error modelError(CostModelState &CM, const Twine &What) {
  return createStringError(inconvertibleErrorCode(),
                           "cost model " + Twine(costModelPath(CM)) + " " +
                               What);
}

// Whether `T` is the cost-model type column of a number format. The column is
// read off NumberFormat.def because poseidon-calibrate links this file without
// Core/NumberFormat.cpp.
static bool isScalarTypeName(StringRef T) {
  static constexpr StringLiteral kCostTypes[] = {
#define NUMBER_FORMAT(Kind, Name, Dtype, CostType, ...) CostType,
#include "Core/NumberFormat.def"
  };
  return !T.empty() && is_contained(kCostTypes, T);
}

// Whether `Opcode,Type` is a row family poseidon-calibrate writes, read by a
// compile or not; must agree with the calibrator.
static bool isCostModelRow(StringRef Opcode, StringRef Type) {
  if (is_contained({"library_ozaki_rel", "library_ozaki_rect_rel",
                    "library_tcec_rel", "library_direct_rel", "inkernel_rel"},
                   Opcode))
    return !Type.empty();
  if (Opcode.consume_front("wmma_mma_")) {
    unsigned M, N, K;
    return Opcode.consume_front("m") && !Opcode.consumeInteger(10, M) &&
           Opcode.consume_front("n") && !Opcode.consumeInteger(10, N) &&
           Opcode.consume_front("k") && !Opcode.consumeInteger(10, K) &&
           Opcode.empty() && !Type.empty();
  }
  for (StringRef Cast : {"fpext_", "fptrunc_"})
    if (Opcode.consume_front(Cast)) {
      auto [From, To] = Opcode.split("_to_");
      return isScalarTypeName(From) && isScalarTypeName(To) && Type == From;
    }
  if (!isScalarTypeName(Type))
    return false;
  return is_contained({"fneg", "fadd", "fsub", "fmul", "fdiv", "fcmp",
                       "fmuladd", "maxnum", "minnum"},
                      Opcode) ||
         any_of(mathFunctions(),
                [&](const MathFunction &Fn) { return Fn.Name == Opcode; });
}

Error loadCostModel(CostModelState &CM, StringRef Requested) {
  if (CM.Loaded)
    return Error::success();
  resolveFromRequest(CM, Requested);
  ErrorOr<std::unique_ptr<MemoryBuffer>> File =
      MemoryBuffer::getFile(costModelPath(CM));
  if (!File)
    return modelError(CM, "cannot be opened");
  SmallSet<StringRef, 2> SeenHeaders;
  for (line_iterator It(**File); !It.is_at_end(); ++It) {
    StringRef Line = *It;
    if (Line[0] == '#') {
      auto [Key, Value] = Line.drop_front(1).split('=');
      Key = Key.trim();
      Value = Value.trim();
      if (Key != "native_arch" && Key != "scalar_types")
        continue;
      if (!SeenHeaders.insert(Key).second)
        return modelError(CM, "line " + Twine(It.line_number()) +
                                  " repeats the '# " + Key + "=' header");
      if (Key == "native_arch") {
        CM.NativeArch = Value.str();
        continue;
      }
      SmallVector<StringRef, 8> Types;
      Value.split(Types, ',', -1, false);
      for (StringRef Type : Types) {
        Type = Type.trim();
        if (!isScalarTypeName(Type))
          return modelError(CM, "line " + Twine(It.line_number()) +
                                    " lists the scalar type '" + Type +
                                    "', which is none of bf16, half, float, "
                                    "double, fp80 and fp128");
        CM.ScalarTypes.insert(Type.str());
      }
      continue;
    }
    SmallVector<StringRef, 3> Fields;
    Line.split(Fields, ',');
    double Cost;
    if (Fields.size() != 3 || Fields[2].trim().getAsDouble(Cost))
      return modelError(CM, "line " + Twine(It.line_number()) +
                                " is not '<opcode>,<type>,<cost>': " + Line);
    if (!std::isfinite(Cost) || Cost < 0)
      return modelError(CM, "line " + Twine(It.line_number()) +
                                " prices the row '" + Fields[0] + "," +
                                Fields[1] + "' at " + Fields[2].trim() +
                                ", which is not a finite non-negative cost");
    if (!isCostModelRow(Fields[0], Fields[1]))
      return modelError(CM, "line " + Twine(It.line_number()) +
                                " has the row '" + Fields[0] + "," + Fields[1] +
                                "', of no row family Poseidon reads "
                                "(poseidon-calibrate --help lists them); "
                                "remove the line, or compile with the "
                                "Poseidon whose poseidon-calibrate wrote it");
    if (!CM.Model.try_emplace({Fields[0].str(), Fields[1].str()}, Cost).second)
      return modelError(CM, "has the row '" + Fields[0] + "," + Fields[1] +
                                "' twice");
  }
  CM.Loaded = true;
  return Error::success();
}

const std::map<std::pair<std::string, std::string>, double> &
getCostModel(CostModelState &CM) {
  assert(CM.Loaded && "the cost model is read before a site required it");
  return CM.Model;
}

std::string costModelRows(CostModelState &CM) {
  std::string Rows;
  if (costModelPath(CM).empty())
    return Rows;
  raw_string_ostream OS(Rows);
  for (const auto &[Row, Cost] : getCostModel(CM))
    OS << Row.first << ',' << Row.second << ',' << format("%a", Cost) << '\n';
  OS << "native_arch=" << CM.NativeArch << '\n';
  return Rows;
}

Error requireCostModel(CostModelState &CM, StringRef Requested,
                       StringRef targetCpu, StringRef function, bool OnDevice) {
  if (Error E = resolveForTarget(CM, Requested, targetCpu, function, OnDevice))
    return E;
  if (Error E = loadCostModel(CM, Requested))
    return E;
  if (CM.NativeArch.empty())
    return modelError(CM, "carries no '# native_arch=' header, so it cannot "
                          "be matched against the compile target");
  if (targetCpu != CM.NativeArch)
    return modelError(CM,
                      "was measured on native_arch '" + CM.NativeArch +
                          "' but " + function + " compiles for target-cpu '" +
                          (targetCpu.empty() ? "<unset>" : targetCpu) + "'");
  return Error::success();
}

const std::string &getCostModelNativeArch(CostModelState &CM) {
  if (!costModelPath(CM).empty())
    getCostModel(CM);
  return CM.NativeArch;
}

std::optional<double> lookupCost(CostModelState &CM, StringRef OpcodeName,
                                 StringRef TypeName) {
  const auto &CostModel = getCostModel(CM);
  auto It = CostModel.find({OpcodeName.str(), TypeName.str()});
  if (It == CostModel.end())
    return std::nullopt;
  return It->second;
}

} // namespace poseidon
