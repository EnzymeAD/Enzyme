#include "MathFunctions.h"

#include "llvm/ADT/STLExtras.h"

#include <vector>

using namespace llvm;

namespace poseidon {

namespace {

// Name, Libm, Optimizable, Device, FloatSuffixInC. libmFuncs is built from
// the Libm rows in this order.
constexpr MathFunction kMathFunctions[] = {
    {"sin", true, true, true, true},
    {"cos", true, true, true, true},
    {"tan", true, true, true, true},
    {"asin", true, true, true, true},
    {"acos", true, true, true, true},
    {"atan", true, true, true, true},
    {"atan2", true, true, true, true},
    {"sinh", true, true, true, true},
    {"cosh", true, true, true, true},
    {"tanh", true, true, true, true},
    {"asinh", true, true, true, true},
    {"acosh", true, true, true, true},
    {"atanh", true, true, true, true},
    {"exp", true, true, true, true},
    {"log", true, true, true, true},
    {"sqrt", true, true, true, true},
    {"cbrt", true, true, true, true},
    {"pow", true, true, true, true},
    {"powi", true, false, true, false},
    {"fabs", true, true, true, true},
    {"fma", true, true, true, true},
    {"hypot", true, true, true, true},
    {"expm1", true, true, true, true},
    {"log1p", true, true, true, true},
    {"ceil", true, true, true, true},
    {"floor", true, true, true, true},
    {"erf", true, true, true, true},
    {"exp2", true, true, true, true},
    {"lgamma", true, true, true, true},
    {"log10", true, true, true, true},
    {"log2", true, true, true, true},
    {"rint", true, true, true, true},
    {"round", true, true, true, true},
    {"tgamma", true, true, true, true},
    {"trunc", true, true, true, true},
    {"copysign", true, true, true, true},
    {"fdim", true, true, true, true},
    {"fmod", true, true, true, true},
    {"remainder", true, true, true, true},
    {"fmin", false, false, true, true},
    {"fmax", false, false, true, true},
    {"erfc", false, false, false, true},
    {"nearbyint", false, false, false, true},
};

} // namespace

ArrayRef<MathFunction> mathFunctions() { return kMathFunctions; }

const std::unordered_set<std::string> &libmFuncs() {
  // Built from the rows in their order, as the initializer list it replaces:
  // getPTFuncs walks it and names the first function the cost model lacks.
  static const std::unordered_set<std::string> LibmFuncs = [] {
    std::vector<std::string> Names;
    for (const MathFunction &F : kMathFunctions)
      if (F.Libm)
        Names.push_back(F.Name.str());
    return std::unordered_set<std::string>(Names.begin(), Names.end());
  }();
  return LibmFuncs;
}

bool isOptimizableLibmCall(StringRef Name) {
  auto Optimizable = [](StringRef N) {
    return any_of(kMathFunctions, [&](const MathFunction &F) {
      return F.Optimizable && F.Name == N;
    });
  };
  return Optimizable(Name) ||
         (Name.ends_with("f") && Optimizable(Name.drop_back()));
}

bool hasFloatSuffixInC(StringRef Name) {
  return any_of(kMathFunctions, [&](const MathFunction &F) {
    return F.FloatSuffixInC && F.Name == Name;
  });
}

} // namespace poseidon
