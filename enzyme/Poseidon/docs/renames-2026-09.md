# Renames, 2026-09

In the Enzyme monorepo (2026-10) the standalone tree's `lib/` is flattened into
`enzyme/Poseidon/` (so `lib/Optimize.cpp` below is `enzyme/Poseidon/Optimize.cpp`),
the public header is `enzyme/include/poseidon/poseidon.h`, the lit suite is
`enzyme/test/Poseidon/` (integration tests under `Integration/`), and the artifact
is `enzyme/Poseidon/artifacts/cgo2027/`. Nothing else moved.

The standalone `poseidon` repo dropped the `Poseidon`/`FPOpt` prefixes it
inherited from the Enzyme fork it was extracted from. Everything user-facing is
unchanged: the 59 `-poseidon-*` flag spellings, the `POSEIDON_*` environment
variables, the binaries, `Poseidon-<N>.so`, `libposeidon_{profile,rt}.a`, the
public header `include/poseidon/poseidon.h`, the CMake package `Poseidon`, every
`[poseidon]` / `Poseidon:` diagnostic, the `Expansion2/3/4` labels, the runtime
ABI symbols, and every on-disk profile, cache and report name. Header guards keep
the `POSEIDON_` prefix, which a macro in the preprocessor's flat namespace earns,
and follow the new file name: `Expansion.h` guards `POSEIDON_EXPANSION_H`.

These tables map the old names to the new ones so a citation in an older
document can still be resolved.

## Files

| Old | New |
|---|---|
| `lib/Poseidon.cpp` | `lib/Optimize.cpp` |
| `lib/Poseidon.h` | `lib/Optimize.h` |
| `lib/PoseidonCanonicalize.{cpp,h}` | `lib/Canonicalize.{cpp,h}` |
| `lib/PoseidonCostModel.{cpp,h}` | `lib/CostModel.{cpp,h}` |
| `lib/PoseidonDriver.{cpp,h}` | `lib/Driver.{cpp,h}` |
| `lib/PoseidonEvaluators.{cpp,h}` | `lib/Evaluators.{cpp,h}` |
| `lib/PoseidonFlags.{cpp,h}` | `lib/Flags.{cpp,h}` |
| `lib/PoseidonHerbieUtils.{cpp,h}` | `lib/Herbie.{cpp,h}` |
| `lib/PoseidonHostDispatch.{cpp,h}` | `lib/HostDispatch.{cpp,h}` |
| `lib/PoseidonInstrument.{cpp,h}` | `lib/Instrument.{cpp,h}` |
| `lib/PoseidonLaunchDescriptors.{cpp,h}` | `lib/LaunchDescriptors.{cpp,h}` |
| `lib/PoseidonMatmul.h` | `lib/matmul/Matmul.h` |
| `lib/PoseidonMultiFloat.{cpp,h}` | `lib/Expansion.{cpp,h}` |
| `lib/PoseidonOzaki.{cpp,h}` | `lib/InKernelRaise.{cpp,h}` |
| `lib/PoseidonOzakiIIConsts.h` | `lib/OzakiII.h` |
| `lib/PoseidonPrecUtils.{cpp,h}` | `lib/Precision.{cpp,h}` |
| `lib/PoseidonProfUtils.{cpp,h}` | `lib/ProfileRead.{cpp,h}` |
| `lib/PoseidonRaiseWMMA.{cpp,h}` | `lib/RaiseWMMA.{cpp,h}` |
| `lib/PoseidonSampling.{cpp,h}` | `lib/Sampling.{cpp,h}` |
| `lib/PoseidonSolvers.{cpp,h}` | `lib/Solvers.{cpp,h}` |
| `lib/PoseidonStageParam.{cpp,h}` | `lib/StageParam.{cpp,h}` |
| `lib/PoseidonStaging.{cpp,h}` | `lib/Staging.{cpp,h}` |
| `lib/PoseidonTypes.{cpp,h}` | `lib/Types.{cpp,h}` |
| `lib/PoseidonUtils.{cpp,h}` | `lib/Utils.{cpp,h}` |
| `lib/PoseidonWmmaUtils.{cpp,h}` | `lib/WmmaUtils.{cpp,h}` |
| `runtime/direct/poseidon_direct_rt.cu` | `runtime/direct/direct_rt.cu` |
| `runtime/ozaki/poseidon_ozaki_rt.cu` | `runtime/ozaki/ozaki_rt.cu` |
| `runtime/stage/poseidon_stage_rt.cu` | `runtime/stage/stage_rt.cu` |
| `runtime/tcec/poseidon_tcec_rt.cu` | `runtime/tcec/tcec_rt.cu` |

`lib/Plugin.cpp`, `lib/matmul/Matmul{Accuracy,Apply,Candidates,Pricing,Profile}.cpp`,
`lib/matmul/MatmulInternal.h`, `tools/driver/poseidon-clang.cpp`,
`tools/calibrate/poseidon-calibrate.cpp`, `include/poseidon/poseidon.h`,
`cmake/PoseidonConfig.cmake.in`, `runtime/fpprofiler/*` and `test/Poseidon/` did
not move.

## Types and namespaces

| Old | New |
|---|---|
| `FPOpt (legacy-PM FunctionPass)` | deleted |
| `FPOptNewPM` | deleted |
| `PoseidonPass` | `poseidon::OptimizePass` |
| `PoseidonFinalizePass` | `poseidon::FinalizePass` |
| `PoseidonHostStubPass` | `poseidon::HostStubPass` |
| `PoseidonNoteMap<Note>` | `poseidon::NoteMap<Note>` |
| `MFValue` | `ExpansionValue` |
| `MFN` | `ExpansionN` |
| `MFB` | `ExpansionBuilder` |
| `MFNUnaryFn / MFNBinaryFn` | `ExpansionNUnaryFn / ExpansionNBinaryFn` |
| `DeferredMFPhiIn` | `DeferredExpansionPhiIn` |
| `PrecisionChangeType::MultiFloat` | `PrecisionChangeType::Expansion2` |
| `namespace enzyme_fpprofile` | `namespace poseidon` |

`lib/` is now one `namespace poseidon`. `lib/OzakiII.h` stays at global scope:
`runtime/ozaki/ozaki_rt.cu` includes it by relative path and is compiled by the
application's own clang. `lib/Plugin.cpp` stays at global scope for the
`extern "C"` plugin entry point.

## Functions and constants

| Old | New |
|---|---|
| `Poseidonable` | `poseidon::isOptimizable` |
| `setPoseidonMetadata` | `poseidon::setSlotMetadata` |
| `preprocessForPoseidon` | `poseidon::preprocess` |
| `runJointPoseidon` | `poseidon::solveJointly` |
| `notePoseidonSiteOrigin` | `poseidon::noteSiteOrigin` |
| `notePoseidonSiteArgs` | `poseidon::noteSiteArgs` |
| `poseidonSiteArg` | `poseidon::siteArg` |
| `redirectPoseidonNoopSite` | `poseidon::redirectNoopSite` |
| `runPoseidonFunctionSimplify` | `poseidon::simplifyFunction` |
| `registerPoseidon` | `poseidon::registerPasses` |
| `poseidonApplyFlagDefaults` | `poseidon::applyFlagDefaults` |
| `hasPoseidonSite` | `poseidon::hasSite` |
| `emitPoseidonReport` | `poseidon::emitReport` |
| `poseidonCanonicalize` | `poseidon::canonicalize` |
| `poseidonCanonicalFormHash` | `poseidon::canonicalFormHash` |
| `poseidonSampleError` | `poseidon::sampleError` |
| `poseidonLibmFuncs` | `poseidon::libmFuncs` |
| `poseidonDeviceMathName` | `poseidon::deviceMathName` |
| `poseidonPercentile` | `poseidon::percentile` |
| `poseidonMangledSuffix` | `poseidon::mangledSuffix` |
| `poseidonApplyGradientFloor` | `poseidon::applyGradientFloor` |
| `poseidonDPCachePath` | `poseidon::dpCachePath` |
| `poseidonDPCacheHasFunction` | `poseidon::dpCacheHasFunction` |
| `poseidonWriteDescriptor` | `poseidon::writeDescriptor` |
| `poseidonReadDescriptors` | `poseidon::readDescriptors` |
| `poseidonRemoveDescriptor` | `poseidon::removeDescriptor` |
| `poseidonIsLaunchStubFor` | `poseidon::isLaunchStubFor` |
| `poseidonInKernelRaiseSharedBytes` | `poseidon::inKernelRaiseSharedBytes` |
| `poseidonEnclosingKernelSharedBytes` | `poseidon::enclosingKernelSharedBytes` |
| `poseidonIdx (local)` | `slotIdx` |
| `kPoseidonDefaultProfileDir` | `poseidon::kDefaultProfileDir` |
| `kPoseidonDefaultCacheDir` | `poseidon::kDefaultCacheDir` |
| `kPoseidonStaticShmemCap` | `poseidon::kStaticShmemCap` |
| `kPoseidonAnnotation` | `poseidon::kAnnotation` |
| `applyMultiFloat` | `applyExpansion` |
| `useMultiFloat` | `useExpansion2` |
| `emitMF{Add,Sub,Mul,Div,Neg,Sqrt,ToFP,ForInstruction}` | `emitExpansion{Add,Sub,Mul,Div,Neg,Sqrt,ToFP,ForInstruction}` |
| `emitToMF` | `emitToExpansion` |
| `operandMF / getOrSplitOperandMF` | `operandExpansion / getOrSplitOperandExpansion` |
| `splitConstantFPMF` | `splitConstantFPExpansion` |
| `inMF / ptHasMF / recordMF` | `inExpansion / ptHasExpansion / recordExpansion` |
| `liveMFRoots` | `liveExpansionRoots` |
| `deferredMFPhiIns` | `deferredExpansionPhiIns` |
| `setMFEliminateCarriedPhis / g_mfEliminateCarriedPhis` | `setExpansionEliminateCarriedPhis / g_expansionEliminateCarriedPhis` |
| `takeLastMFLimbs / g_lastMFLimbs` | `takeLastExpansionLimbs / g_lastExpansionLimbs` |
| `mf<Op> (mfTwoSum, mfAddRaw3, mfNormalize4, ...)` | `expansion<Op> (expansionTwoSum, expansionAddRaw3, expansionNormalize4, ...)` |
| `mfn<Op> (mfnAdd, mfnMul, mfnSqrt, ...)` | `expansionN<Op> (expansionNAdd, expansionNMul, expansionNSqrt, ...)` |

## Command-line flag variables

Every variable moved into `poseidon::flags` and is now named for its flag: the
flag string after `-poseidon-`, CamelCased, with `PT`, `DP`, `MPFR`, `WMMA` and
`InKernel` keeping their case. **No flag string changed.**

| Old variable | New variable | Flag (unchanged) |
|---|---|---|
| `FPProfileGenerate` | `poseidon::flags::ProfileGenerate` | `-poseidon-profile-generate` |
| `FPProfileUse` | `poseidon::flags::ProfileUse` | `-poseidon-profile-use` |
| `PoseidonKernels` | `poseidon::flags::Kernels` | `-poseidon-kernels` |
| `PoseidonMinCostShare` | `poseidon::flags::MinCostShare` | `-poseidon-min-cost-share` |
| `FPOptLooseCoverage` | `poseidon::flags::LooseCoverage` | `-poseidon-loose-coverage` |
| `FPOptEnableHerbie` | `poseidon::flags::EnableHerbie` | `-poseidon-enable-herbie` |
| `FPOptEnablePT` | `poseidon::flags::EnablePT` | `-poseidon-enable-pt` |
| `FPOptEnableMultiFloat` | `poseidon::flags::EnableMultifloat` | `-poseidon-enable-multifloat` |
| `FPOptExpansionComponents` | `poseidon::flags::ExpansionComponents` | `-poseidon-expansion-components` |
| `FPOptEnableThreeTier` | `poseidon::flags::EnableThreeTier` | `-poseidon-enable-three-tier` |
| `FPOptTwoTierStep` | `poseidon::flags::TwoTierStep` | `-poseidon-two-tier-step` |
| `FPOptRaiseWMMA` | `poseidon::flags::RaiseWMMA` | `-poseidon-raise-wmma` |
| `FPOptMaxExprLength` | `poseidon::flags::MaxExprLength` | `-poseidon-max-expr-length` |
| `FPOptMinUsesForSplit` | `poseidon::flags::MinUsesSplit` | `-poseidon-min-uses-split` |
| `FPOptMinOpsForSplit` | `poseidon::flags::MinOpsSplit` | `-poseidon-min-ops-split` |
| `FPOptComputationCostBudget` | `poseidon::flags::CompCostBudget` | `-poseidon-comp-cost-budget` |
| `FPOptErrorBudget` | `poseidon::flags::Tau` | `-poseidon-tau` |
| `FPOptTauCheapest` | `poseidon::flags::TauCheapest` | `-poseidon-tau-cheapest` |
| `FPOptEarlyPrune` | `poseidon::flags::EarlyPrune` | `-poseidon-early-prune` |
| `FPOptJointDP` | `poseidon::flags::JointDP` | `-poseidon-joint-dp` |
| `FPOptApplyRewrites` | `poseidon::flags::ApplyRewrites` | `-poseidon-apply-rewrites` |
| `FPOptCostModelPath` | `poseidon::flags::CostModel` | `-poseidon-cost-model` |
| `FPOptNumSamples` | `poseidon::flags::NumSamples` | `-poseidon-num-samples` |
| `FPOptRandomSeed` | `poseidon::flags::RandomSeed` | `-poseidon-random-seed` |
| `FPOptSampleLogBits` | `poseidon::flags::SampleLogBits` | `-poseidon-sample-log-bits` |
| `FPOptStrictMode` | `poseidon::flags::StrictMode` | `-poseidon-strict-mode` |
| `FPOptExponentPenalty` | `poseidon::flags::ExponentPenalty` | `-poseidon-exponent-penalty` |
| `FPOptCachePath` | `poseidon::flags::Cache` | `-poseidon-cache` |
| `HerbieNumThreads` | `poseidon::flags::HerbieNumThreads` | `-poseidon-herbie-num-threads` |
| `HerbieTimeout` | `poseidon::flags::HerbieTimeout` | `-poseidon-herbie-timeout` |
| `HerbieNumPoints` | `poseidon::flags::HerbieNumPts` | `-poseidon-herbie-num-pts` |
| `HerbieNumIters` | `poseidon::flags::HerbieNumIters` | `-poseidon-herbie-num-iters` |
| `HerbieNumEnodes` | `poseidon::flags::HerbieNumEnodes` | `-poseidon-herbie-num-enodes` |
| `FPOptHerbieSubgraphTimeout` | `poseidon::flags::HerbieSubgraphTimeout` | `-poseidon-herbie-subgraph-timeout` |
| `FPOptHerbieBinary` | `poseidon::flags::HerbieBinary` | `-poseidon-herbie-binary` |
| `FPOptHerbiePlatform` | `poseidon::flags::HerbiePlatform` | `-poseidon-herbie-platform` |
| `FPOptOzakiHostDispatch` | `poseidon::flags::OzakiHostDispatch` | `-poseidon-ozaki-host-dispatch` |
| `FPOptInKernelCalibration` | `poseidon::flags::InKernelCalibration` | `-poseidon-inkernel-calibration` |
| `FPOptPrint` | `poseidon::flags::Print` | `-poseidon-print` |
| `FPOptShowTable` | `poseidon::flags::ShowTable` | `-poseidon-show-table` |
| `FPOptReportPath` | `poseidon::flags::ReportPath` | `-poseidon-report-path` |

## IR strings

| Old | New |
|---|---|
| `!enzyme_fpprofile_idx (metadata kind)` | `!poseidon.prof.idx` |
| `enzyme_err_tol (marker keyword)` | `poseidon_tau` |
| `"fpopt.{fpcast,const.fpcast,promote,demote}" (IR value names)` | `"poseidon.{fpcast,const.fpcast,promote,demote}"` |

`!poseidon.prof.idx` and `poseidon_tau` were verified absent from the pinned
Enzyme before the change; they were never part of Enzyme's contract.
`__enzyme_autodiff_poseidon` and every `enzyme_*` activity keyword are Enzyme's
and did not change.

