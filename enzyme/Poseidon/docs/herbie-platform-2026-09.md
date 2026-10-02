# Per-device Herbie platform, and platform identity in the Herbie cache key

2026-09-09. Working tree of `poseidon/` (standalone, branch `main`).

Herbie's algebraic search does not just rewrite an expression, it RANKS the
rewrites it finds, and it ranks them by a platform: a Racket module that gives
every operation and representation a cost. Poseidon's DP then re-prices the
surviving candidates from the measured CSV. Until this change the two halves
could disagree about which device they were describing, and nothing said so.

Two defects, both silent:

1. **One platform existed.** The build generated `cuda-sm120` from the RTX 5090
   CSV and patched it into the Herbie source tree before compiling the binary
   (`cmake/Herbie.cmake`, `tools/herbie/csv_to_herbie_platform.py --herbie-src`).
   The pass asked for `cuda-<native_arch>` (`PoseidonHerbieUtils.cpp`), so on a
   GH200 it asked for `cuda-sm90`, which had never been generated.
2. **The cache key did not name the platform.** Entries were
   `cachedHerbieOutput_<site hash>_<subgraph>_<baseargs>.txt`. The site hash is
   a property of the IR, not of the device, so every GH200 GENGA solve replayed
   the candidate set Herbie had proposed for the RTX 5090 and Poseidon then
   priced it with GH200 numbers.

## Changes

Compiler:

- `lib/PoseidonHerbieUtils.cpp`: new `getHerbiePlatform()` resolves one
  `{Name, Path, Digest}` per compile. `Name` is `cuda-<arch without
  underscores>` for a CUDA cost model and `default` for a host one, or the
  explicit `-poseidon-herbie-platform=<name>`. `Path` is the explicit flag when
  it looks like a file, else `<resolved cost model>.herbie.rkt`; a CUDA cost
  model with no such file aborts the compile naming the CSV and the
  `poseidon-calibrate --only herbie-platform` command that writes it. There is
  no fallback to another architecture's platform. `Digest` is xxh3 of that
  file.
- `lib/PoseidonHerbieUtils.cpp`: the platform is passed to Herbie by path
  (`--platform <file>`), and `Name` is now the first component of every cache
  entry: `cachedHerbieOutput_<platform>_<site hash>_<subgraph>_<baseargs>.txt`,
  and likewise for the digest-qualified collision sibling and the
  `cachedHerbieTimeout_` marker. An entry without the component is never read.
- `lib/PoseidonHerbieUtils.cpp`: a `<entry>.platform` stamp records
  `<name> <digest>` when an entry is written. On the read path a stamped digest
  that differs from this compile's platform prints a `[poseidon] WARNING` and
  the entry is still used: the architecture is the same and only the measured
  prices moved, which is a recalibration, not another device. A missing stamp
  (every shipped entry) is noted under `-poseidon-print` only.
- `lib/PoseidonHerbieUtils.cpp`: a Herbie run that exits without writing
  `results.json` now aborts the compile instead of continuing with no algebraic
  candidates. That is exactly how a platform Herbie cannot load presents
  itself: exit status 1, a Racket message on stderr, no output directory.
- `lib/PoseidonFlags.cpp`: `-poseidon-herbie-platform` documents that it takes
  a path or a name.

Platform generator, `tools/herbie/csv_to_herbie_platform.py`:

- Emits a `(module <platform-name> <language> ...)` form instead of
  `#lang s-exp "../syntax/platform-language.rkt"`, and drops `--herbie-src`,
  `--no-patch` and `patch_load_platform`. **This is the change that makes a
  path-passed platform work at all.** Herbie's `activate-platform!` falls back
  to `(dynamic-require (string->path name) 'platform)` for a name it does not
  know, but the Herbie this project builds and ships is a `raco exe` /
  `raco distribute` executable and carries no Racket collection directories, so
  reading a `#lang s-exp` module fails with `standard-module-name-resolver:
  collection not found for module path: s-exp/lang/reader`. A plain `(module
  ...)` form needs no reader, and its language is the name `raco exe` gave the
  embedded `src/syntax/platform-language.rkt`,
  `'|#%embedded:syntax/platform-language:|` (`flsingle` likewise comes from
  `'|#%embedded:math/flonum:|`). `--platform-language` and `--flonum-module`
  override both for a Herbie that is not a `raco exe` binary.
- `--output` is the only output mode besides stdout.

Calibration tool, `tools/calibrate/poseidon-calibrate.cpp`:

- New arm `herbie-platform`: reads the CSV's `# native_arch=` header and writes
  `<csv without .csv>.herbie.rkt` beside it through the generator. It measures
  nothing, so `--only herbie-platform --out <csv>` short-circuits the device
  query in `main()` and runs on a machine with no GPU at all, which is how a
  model measured elsewhere gets its platform here. It runs after `microbm` and
  in the default full run. A host cost model gets no file and says so.
- `--check` reports the platform file for a CUDA model and counts it as a
  problem when it is absent.
- `CMakeLists.txt` passes `POSEIDON_CALIBRATE_PYTHON=${Python3_EXECUTABLE}`;
  the tool falls back to `python3` on `PATH` and fails with a clear message if
  neither exists.

Build:

- `cmake/Herbie.cmake` no longer runs the generator against the Herbie source
  tree, so the ExternalProject build no longer mutates it and no platform is
  compiled into the binary. `HERBIE_PLATFORM_ARCH` and `HERBIE_PLATFORM_DEVICE`
  are gone; `HERBIE_PLATFORM_CSV` stays because `test/CMakeLists.txt` uses it
  as the default CUDA test cost model. A build configured with
  `-DPOSEIDON_HERBIE_BINARY=<prebuilt>` (the local one) is unaffected.
- `cost_models/cm_sm_120_RTX5090.herbie.rkt` is the paper GPU's platform,
  generated from the shipped CSV by the new arm and installed alongside it by
  the existing `install(DIRECTORY cost_models ...)` rule. Its body is
  byte-identical to the file that was compiled into the pinned Herbie binary
  (verified below), so the paper pipeline now passes a path like any other
  device and searches under exactly the costs it always did.

Artifact:

- `artifacts/cgo2027/benchmarks/genga_nbody/scripts/common.sh` generates the
  platform beside the derived `cm_native_x1e6.csv` it hands the compiler.
  Scaling every price by 1e6 leaves every Herbie ranking unchanged, so this is
  the device's platform in the derived file's units.
- `artifacts/cgo2027/scripts/bootstrap_cost_model.sh`: the full calibrate run
  already writes the platform; the paper-GPU early exit now writes the shipped
  CSV's platform before `--check` so that path cannot report it missing.
- The shipped caches were moved with plain `mv` into the platform-keyed names
  (all were produced under `cuda-sm120`):
  `benchmarks/genga_nbody/cache_kep_rolled/cachedHerbieOutput_cuda-sm120_5aec0310314f345c_<0..7>_0.txt`
  and `cache_unified/cachedHerbieOutput_cuda-sm120_c24841e596af4397_<0..2>_0.txt`.
  No `.platform` stamp was written for them: the digest they were produced
  under is not recorded anywhere, and inventing one would assert a measurement
  that was never made. The read path treats a missing stamp as unverified and
  the platform NAME in the key is what protects them.
- `expected_results.md`, the `Makefile` clean note, `run_kepler.sh` (comment and
  the `#f` count, now a single integer) and `.gitignore` follow the new names.

Tests:

- `test/Poseidon/herbie_platform_key_nvptx.ll` (new): one seeded entry keyed
  `cuda-sm120`, four runs. The matching architecture reads it; an sm_90 cost
  model does not and runs Herbie instead; a CUDA cost model with no platform
  file beside it aborts naming the file and the command; the same platform name
  with a different cost digest is reused and reported. No arm needs a Herbie
  binary.
- `test/Poseidon/Inputs/cm_gpu_fixture_sm90_x1e6.csv` (new): the existing GPU
  fixture with its `native_arch` header changed, so two architectures exist in
  the fixtures. Its header says the prices are the RTX 5090's and are not a
  measurement of any sm_90 part.
- `test/Poseidon/Inputs/cm_gpu_fixture_x1e6.herbie.rkt` and
  `cm_gpu_fixture_sm90_x1e6.herbie.rkt` (new): their platforms.
- `test/Poseidon/Inputs/fma_opt_cache/cachedHerbieOutput_default_*` : renamed;
  a host cost model's platform is `default`. `fma_opt_herbie_cached.ll` follows.

## Verification (CPU only; no benchmark binary was executed)

Everything below ran on the workstation with the GPUs left alone. The binaries
the frontier arms produce were compiled and never launched.

### C.1 Build and lit

`cmake -S . -B build && ninja -C build` clean; `ninja -C build check-poseidon`
19/19 pass, including the new `herbie_platform_key_nvptx.ll`.

### C.2 The local Herbie loads a generated platform, and the platform changes the search

`herbie report` on two FPCores, `(- 1 (cos x))` and `(- (sin x) x)`, seed 239,
256 points, 3 iterations, under three platforms
(`herbie_platform_20260909/c2/`, binary
`poseidon-clean/enzyme/build/Enzyme/herbie/install/herbie/bin/herbie`):

| `--platform` | exit | core `(- 1 (cos x))`: initial cost / best cost / alternatives |
| --- | --- | --- |
| `cuda-sm120` (compiled into the binary) | 0 | 0.128 / 0.269 / 9 |
| `cost_models/cm_sm_120_RTX5090.herbie.rkt` (path) | 0 | 0.128 / 0.269 / 9 |
| the GH200 platform (path) | 0 | 0.008 / 0.012 / 4 |

The first two `results.json` agree on every cost, every alternative and both
chosen expressions: passing the shipped platform by path reproduces the
compiled-in one exactly. Its body is also byte-identical to
`herbie-cuda/src/platforms/cuda-sm120.rkt`, the file that was generated into
the Herbie tree when this binary was built.

The GH200 platform loads and gives a different search: four alternatives
instead of nine, and a different chosen expression. The same file under the old
`#lang s-exp` form fails on this binary with `standard-module-name-resolver:
collection not found for module path: s-exp/lang/reader`, which is what made
the `(module ...)` form necessary.

### C.3 GENGA Kepler drift on GH200, live Herbie under the GH200 platform

`herbie_platform_20260909/c3/solve_gh200.sh`. The shipped profile
`fpprofile_kep_rolled`, `--cuda-gpu-arch=sm_90`, the GH200 CSV
(`crossgpu_figdata/gh200_fixed/cost_models/cm_sm_90_GH200120GB.csv`) scaled by
1e6 the way `common.sh` does, its generated platform
`cm_gh200_x1e6.herbie.rkt`, a fresh cache, the shipped `HERBIE_CAPS` and
`SPLIT`. Herbie ran live for all 8 subgraphs, none timed out, and the whole
solve took 2 min 59 s at 6.5 GB peak RSS, well inside the caps.

A control arm isolates the platform from every other difference between this
compiler and phase 3's: `herbie_platform_20260909/c3ctl/solve_replay.sh` is the
same solve with the shipped `cuda-sm120` entries copied under `cuda-sm90`
names, which is exactly what the platform-blind cache key did on GH200 (8 cache
hits, 0 live invocations). Same compiler, same profile, same GH200 prices; only
the candidate set differs.

**Candidate sets.** 26 Herbie cores, 11 of them differ (`c3/candidate_diff.txt`):
2 in subgraph 6 (the Newton residual) and 9 in subgraph 7 (the f-and-g
cancellations). Subgraphs 0 to 5, the six dot-product FMA reassociations, are
identical. Herbie's alternatives carry no expression string in this build, so a
core's candidate set is its chosen `output`, and that is what differs.

**Applied steps per budget.** `dot` is one of the six FMA reassociations of
`rsq`/`vsq`/`u`; `ia` is `2*ir - vsq/mu`; `newton` is the Newton update
`dEj -= f0/f1` with its `1 - (ec cos dEj - es sin dEj)` cancellation; `en` is
`sqrt(mu*ia2*ia)`. Full tables in `c3/applied_live_sm90.tsv`,
`c3/applied_control_sm120cache.tsv`, `c3/applied_phase3_sm120cache.tsv`.

| control (cuda-sm120 candidates) | applied | live (cuda-sm90 candidates) | applied |
| --- | --- | --- | --- |
| -21011024 | dot x6 + 2 PT | -21011024 | dot x6 + 2 PT |
| -18874180 | dot x6 + 2 PT | -18874180 | dot x6 + 2 PT |
| -18548830 | dot x6 + 2 PT | -18548830 | dot x6 + 2 PT |
| -18057421 | dot x6 + 2 PT | -18057421 | dot x6 + 2 PT |
| -18050437 | dot x6 + 2 PT | -18050437 | dot x6 + 2 PT |
| -17714746 | dot x6 + PT | -17714746 | dot x6 + PT |
| -14884546 | dot x6, **ia** + PT | -14817702 | dot x6, **ia** + PT |
| -6565311 | dot x6 + PT | **-7148661** | dot x6, **newton** |
| -3735111 | dot x6, **ia** + PT | **-4251617** | dot x6, **ia**, **newton** |
| -2634035 | dot x6 + PT | -2634035 | dot x6 + PT |
| -4386 | dot x6 | -4386 | dot x6 |
| 2825814 | dot x6, **ia** | 2892658 | dot x6, **ia** |
| 14730452 | dot x6 + Expansion2 PT | 14730452 | dot x6 + Expansion2 PT |
| 17560652 | dot x6, **ia** + Expansion2 PT | 17627496 | dot x6, **ia** + Expansion2 PT |

**Verdict: yes.** The GH200-platform search produces an algebraic pick the
replayed `cuda-sm120` cache never offered: the subgraph-6 Newton-residual
rewrite, selected at budgets -7148661 and -4251617, replacing two mid-frontier
points that the sm120 candidate set filled with a plain precision step. Under
the GH200 platform Herbie's FP64 rewrite of that subgraph reaches an end error
of 0.0916 bits against the sm120 rewrite's 0.1704, so it is not a tie broken
differently: it is a rewrite the other platform did not find. Neither platform
selects the standalone `1 - cos dEj` or `sin dEj - dEj` rewrites at any budget;
those two cores are proposed and priced out in both.

The FP64-grade top of the frontier (budgets -4386 and above) applies the same
candidate content in both arms, so the row Table 1 quotes has not changed
composition. Its budget value moves (2825814 to 2892658) because the DP table
changed, and two mid-frontier rows change composition outright.

**Consequence: the GH200 GENGA drift frontier must be re-measured on Daint
before the paper's GH200 GENGA drift cell is final.** Speedup and energy drift
of the two changed budgets cannot be read off any existing run, and the whole
budget axis shifted.

For the record, phase 3's own numbers came from a different compiler as well:
`c3/applied_phase3_sm120cache.tsv` has 16 budgets and applies the subgraph-7
`en` rewrite at four of them, which neither arm here does. That is why the
control arm exists; the comparison above is the platform alone.

### C.4 Paper-GPU regression, cache replay only

`herbie_platform_20260909/c4/run.sh` runs step [2/4] of `run_kepler.sh` through
the artifact's own `scripts/common.sh` with `GPU_ARCH=sm_120` pinned so nothing
queries a device, and executes no binary:

```
hits:  8
live:  0
budgets: IDENTICAL to the shipped file
```

All 8 hits are the migrated
`cachedHerbieOutput_cuda-sm120_5aec0310314f345c_<0..7>_0.txt`, and
`cache_kep_rolled/budgets.txt` came back byte-identical to the copy taken
before migration (`herbie_platform_20260909/budgets_shipped_kep_rolled.txt`,
35 budgets from -803128083 to 131743624).

## Where the evidence is

The campaign directory `herbie_platform_20260909/` (outside the repository, as
run outputs are). Its `README.md` lists the four arms; the two analysis scripts
`summarize_cache.py` and `applied_table.py` rebuild the candidate diff and the
per-budget tables from the cache directories and compile logs.

## Two things a reader should know

- The `#%embedded:` module names the generated platform uses are `raco exe`'s
  naming of the modules it embedded. That ties a generated platform to a
  `raco exe` Herbie built from a tree whose platform language sits at
  `src/syntax/platform-language.rkt`, which is what
  `cmake/Herbie.cmake` builds and what the pinned binary is. A Herbie that is
  not such a binary needs `--platform-language` / `--flonum-module`, and a
  Herbie that cannot load the platform exits 1 with a Racket message and the
  compile now aborts on it rather than silently proposing nothing.
- `cost_models/cm_sm_120_RTX5090.herbie.rkt`, the two lit fixture platforms and
  the renamed cache entries are new files in the working tree and are not
  staged: git was read-only for this work.
