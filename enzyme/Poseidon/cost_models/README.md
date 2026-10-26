# The cost model

Poseidon prices every candidate from one CSV measured on the device the code
will run on. `poseidon-calibrate` produces it, and a compile that is not told
which model to use looks one up by the module's target-cpu:

    -poseidon-cost-model=<csv>              what the compile was told
    POSEIDON_COST_MODEL=<csv>               what the environment says
    ~/.cache/poseidon/cm_<arch>_*.csv       what this user has measured
    <prefix>/share/poseidon/cost_models/cm_<arch>_*.csv    what is installed

Zero matches, or more than one, aborts the compile naming `poseidon-calibrate`
and the files it found. A model whose `# native_arch=` header does not name the
compile target aborts as well: another device's CSV is that device's prices, and
would reproduce that device's picks.

`cm_sm_120_RTX5090.csv` is the model the paper was solved with, measured on an
RTX 5090 (Blackwell GB202). `cm_sm_120_RTX5090.herbie.rkt` beside it is the
Herbie platform generated from it: Herbie ranks the rewrites its algebraic
search finds by a platform cost table, and a compile passes the one that sits
next to the CSV it prices from, so the search and the DP describe one device.
Both are installed to `share/poseidon/cost_models/`.

## Measuring a device

    poseidon-calibrate --gpu 1                 # writes ~/.cache/poseidon/cm_<arch>_<gpu>.csv
    poseidon-calibrate --gpu 1 --out <dir>     # or somewhere else
    poseidon-calibrate --gpu 1 --check         # is the resolved model this device's?

Five arms write disjoint row families into the same file; `--only` runs a
subset, `--force` re-measures a family that is already there. A sixth,
`herbie-platform`, measures nothing: it translates the CSV into
`<csv>.herbie.rkt` beside it. It runs after `microbm` and in a full run, and
because it needs no device, `--only herbie-platform --out <csv>` also runs on a
machine that does not have the GPU the CSV was measured on.

| Arm | Rows | What is measured |
|---|---|---|
| microbm | `<op>,<type>,<cost>`, `wmma_mma_<shape>,<in>_<acc>,<cost>` | Saturated reciprocal throughput per (op, type), and per tensor-core tile, from dependent op chains on every SM. Also writes the `# native_arch=`, `# scalar_types=` and `# matrix_types=` headers. |
| ozaki | `ozaki_dispatch_rel,nm<8..14>`, `ozaki_dispatch_rel,dgemm` | Host-dispatched Ozaki-II GEMM per modulus count, and the native cuBLAS DGEMM. |
| tcec | `tcec_dispatch_rel,fp16tcec` | Host-dispatched error-corrected GEMM. `--cumpsgemm <dir>` measures it against that cuMpSGEMM build, which is what the paper measured; without it the runtime's built-in cuBLAS backend is measured, and that is a different and slower product. |
| direct | `direct_dispatch_rel,<in>_<acc>` | Host-dispatched reduced-precision cuBLAS GEMM. |
| inkernel | `wmma_inkernel_rel,<class>` | Tensor-core raises inside a kernel, one class at a time: the harness is compiled through `poseidon-clang++` with `-poseidon-apply-rewrites`, so what is timed is the code a real solve emits. |
| herbie-platform | none (writes `<csv>.herbie.rkt`) | Nothing. The per-op rows are translated into the Herbie platform the algebraic search ranks candidates by, so Herbie and the DP price the same device. A compile against a CUDA CSV with no platform beside it aborts rather than search under another architecture's costs, and Herbie result caches are keyed by the platform name so one device's rewrites are never replayed for another. |

Every `*_rel` row is one number in one unit: the candidate's wall clock over the
scalar-FP64 GEMM's, at the same shape on the same device, so the DP compares an
in-kernel raise against a library dispatch directly. The in-kernel arm needs a
profile of the kernel it raises to know which candidates the pass proposes
(`--profile <dir>`); the profile decides which candidates exist, never what they
measure.

A class with no measured row is not proposed. The pass refuses to compose a
price for it out of per-MMA rows: that composition is an arithmetic ceiling that
ran 3.5x to 64x optimistic, because a materialized raise also pays a scratch
fill, shared-memory staging and accumulator traffic that no per-MMA row
contains.

## Format

One row per line, `key,name,value`, plus `#`-prefixed headers. `native_arch` is
the architecture the file was measured on; `scalar_types` and `matrix_types`
list the precisions the solver may propose, and override the built-in defaults.
The DP cache is keyed on a hash of the parsed rows, so changing a price
invalidates a cached table while reordering rows or editing comments does not.

Scalar costs are the precise, IEEE lowering of each operation (`div.rn.f32`,
libdevice `__nv_*` for the transcendentals). Hardware `.approx` variants are 5x
to 50x faster and lose accuracy; a kernel that opts into them would need a
parallel row family, which does not exist. `fneg` and `fabs` measure at about
one cycle because NVIDIA folds them into the consuming instruction's operand
modifiers.
