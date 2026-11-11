# The cost model

Poseidon prices every candidate from one CSV measured on the device the code
will run on. No model ships with Poseidon: `poseidon-calibrate` produces one,
and a compile that is not told which model to use looks one up by the module's
target-cpu:

    -poseidon-cost-model=<csv>              what the compile was told
    POSEIDON_COST_MODEL=<csv>               what the environment says
    $XDG_CACHE_HOME/poseidon/cm_<arch>_*.csv  what this user has measured
                                            (default ~/.cache/poseidon)

No match stops the compile naming `poseidon-calibrate`; more than one stops it
naming the files it found. A model whose `# native_arch=` header does not name
the compile target aborts as well: another device's CSV is that device's
prices, and would reproduce that device's picks. Results solved with a measured
model are reproduced from that model, so it is kept with them and named with
`-poseidon-cost-model`.

Herbie ranks the rewrites its algebraic search finds by a platform cost table.
A compile passes the platform that sits next to the CSV it prices from
(`<csv>.herbie.rkt`), so the search and the DP describe one device.

## Measuring a device

    poseidon-calibrate --gpu 1                 # writes ~/.cache/poseidon/cm_<arch>_<gpu>.csv
    poseidon-calibrate --gpu 1 --out <dir>     # or somewhere else
    poseidon-calibrate --gpu 1 --check         # is the resolved model this device's?
    poseidon-calibrate --gpu 1 --ozaki-rect 343x1536   # plus the rectangular rows of an M x Ncols x K GEMM
    poseidon-calibrate --gpu 1 --ozaki-rect 112x112x2048  # the native DGEMM row of an M x Ncols x K call

Six arms write disjoint row families into the same file; `--only` runs a
subset, `--force` re-measures a family that is already there. A seventh,
`herbie-platform`, measures nothing: it translates the CSV into
`<csv>.herbie.rkt` beside it. It runs after `microbm` and in a full run, and
because it needs no device, `--only herbie-platform --out <csv>` also runs on a
machine that does not have the GPU the CSV was measured on.

| Arm | Rows | What is measured |
|---|---|---|
| microbm | `<op>,<type>,<cost>`, `wmma_mma_<shape>,<in>_<acc>,<cost>` | Saturated reciprocal throughput per (op, type), and per tensor-core tile, from dependent op chains on every SM. Also writes the `# native_arch=`, `# scalar_types=` and `# matrix_types=` headers. |
| ozaki | `ozaki_dispatch_rel,nm<8..14>`, `ozaki_dispatch_rel,dgemm` | Host-dispatched Ozaki-II GEMM per modulus count, and the native cuBLAS DGEMM. |
| ozaki-rect | `ozaki_dispatch_rect_rel,nm<8..14>_m<M>k<K>`, `ozaki_dispatch_rect_rel,dgemm_m<M>n<N>k<K>` | The rectangular arm of the Ozaki-II dispatch at the `--ozaki-rect` shapes `MxK`, per MAC. A host GEMM whose square padding would waste work runs that arm, whose cost is linear in the column count, so the row is keyed by (M, K) and replaces the square row times the padding ratio. A shape `MxNxK` measures the native DGEMM of a call with exactly those extents (`dgemm_m<M>n<N>k<K>`): a DGEMM with few columns runs far below the square rate, and a call whose three extents are constants of its compile is priced from this row when the model has it, from `ozaki_dispatch_rel,dgemm` otherwise. Rows of shapes a run does not name are kept. |
| tcec | `tcec_dispatch_rel,fp16tcec` | Host-dispatched error-corrected GEMM. `--cumpsgemm <dir>` measures it against that cuMpSGEMM build; without it the runtime's built-in cuBLAS backend is measured, and that is a different and slower product. |
| direct | `direct_dispatch_rel,<in>_<acc>` | Host-dispatched reduced-precision cuBLAS GEMM. |
| inkernel | `wmma_inkernel_rel,<class>` | Tensor-core raises inside a kernel, one class at a time: the harness is compiled through `poseidon-clang++` with `-poseidon-apply-rewrites`, so what is timed is the code a real solve emits. |
| herbie-platform | none (writes `<csv>.herbie.rkt`) | Nothing. The per-op rows are translated into the Herbie platform the algebraic search ranks candidates by, so Herbie and the DP price the same device. A compile against a GPU CSV with no platform beside it aborts rather than search under another architecture's costs, and Herbie result caches are keyed by the platform name so one device's rewrites are never replayed for another. |

An AMD GPU is measured by the HIP builds of the same tools
(`poseidon-microbm-hip`, `poseidon-ozaki-calibrate-hip`,
`poseidon-direct-calibrate-hip`, built when the configuration names
`POSEIDON_ROCM_PATH`, for the processor `POSEIDON_AMDGPU_ARCH` names, default
`gfx942`), which `poseidon-calibrate` runs where there is no `nvidia-smi`; the
device is the one `HIP_VISIBLE_DEVICES` (or `--gpu`) selects.
They write `cm_gfx<NNN>_<device>.csv` with the same row names in the same unit,
and a compile for that processor finds it by its triple. Rows the device has no
runtime or instruction for are absent, so the solver does not propose those
classes: the WMMA tile rows and the in-kernel raises (NVPTX-only), the
error-corrected dispatch (no HIP runtime) and `direct_dispatch_rel,tf32_f32`
(hipBLAS has no TF32 compute type). Its Herbie platform is `hip-gfx<NNN>`.

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

One row per line, `opcode,type,cost`, plus `#`-prefixed headers. `native_arch`
is the architecture the file was measured on; `scalar_types` lists the
precisions the solver may propose on a GPU. A row the compile needs, a header it
reads, a line that is not a row, a row given twice and a row of no family above
each stop the compile, naming the model: nothing is substituted for a part of
it, and a misspelt row a candidate class depends on does not silently withdraw
the class. A new family is added to the pass (`Model/CostModel.cpp`) with the
arm that writes it. The DP cache is keyed
on a hash of the parsed rows, so changing a price invalidates a cached table
while reordering rows or editing comments does not.

Scalar costs are the precise, IEEE lowering of each operation (`div.rn.f32`,
libdevice `__nv_*` for the transcendentals; on AMD the IEEE `fdiv` lowering and
ROCm's ocml `__ocml_*`). Hardware `.approx` variants are 5x to 50x faster and
lose accuracy; a kernel that opts into them would need a parallel row family,
which does not exist. `fneg` and `fabs` measure at about one cycle because
NVIDIA folds them into the consuming instruction's operand modifiers; on AMD
they are the 32-bit sign-bit operation the backend emits where it cannot fold
them.
