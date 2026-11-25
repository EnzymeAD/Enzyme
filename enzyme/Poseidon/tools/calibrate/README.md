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

    poseidon-calibrate --gpu 0                 # writes ~/.cache/poseidon/cm_<arch>_<gpu>.csv
    poseidon-calibrate --gpu 0 --out <dir>     # or somewhere else
    poseidon-calibrate --gpu 0 --check         # is the resolved model this device's?
    poseidon-calibrate --gpu 0 --ozaki-rect MxK      # plus the rectangular rows of an M x Ncols x K GEMM
    poseidon-calibrate --gpu 0 --ozaki-rect MxNxK    # the native DGEMM row of an M x Ncols x K call
    poseidon-calibrate --cpu native            # this host, for a compile with -march=native
    poseidon-calibrate --cpu x86-64            # this host, for a compile with -march=x86-64, the default on x86-64
    poseidon-calibrate --cpu x86-64 --check

A host CPU is measured for the target-cpu its compiles use: `--cpu` takes the
name `-march=` takes, and `native` stands for the CPU LLVM detects on this
machine, whose features the measurement takes as well. The model is
`cm_<target-cpu>_<cpu>.csv`, in the same place as a GPU's. Its rows are timed
by `microbm_cpu.py`, which JIT-compiles each microbenchmark with llvmlite for
that CPU and stops if llvmlite's LLVM does not know it. A trial run sizes each
loop to run for a quarter of a second, and a row is the time of one operation
less that of the same loop without it, in the unit of the GPU rows, clock
cycles times two, so a model takes a few minutes and does not depend on the
trip count. A host model holds rows of two kinds, which mean different things:

- An arithmetic operation, comparison, conversion, fused multiply-add, square
  root, minimum, maximum, rounding or sign operation, or `powi` row is the
  operation's latency: a dependent chain through the operation, as the GPU
  microbenchmark chains it, or, where the compiler would fold that chain, the
  operation on fixed inputs that a restoring step makes depend on the
  previous result.
- A math-library row is the function's reciprocal throughput: independent
  calls on fixed inputs. A dependent chain would price a call whose result
  follows a predicted branch (an overflowing `exp`, a saturated `tanh`) at
  nothing. The inputs lie inside the function's domain, on the ordinary path
  of its routine in the C library, in the intervals `microbm_cpu.py` lists
  beside each function.

The GPU rows are reciprocal throughputs throughout, so a GPU model and a host
model are not comparable row for row. Before a loop is timed, the script
checks its machine code for an instruction of every operation it times, and
stops naming the row if the compiler has folded one away. The script runs under
the Python 3 the build was configured with (`Python3_EXECUTABLE`) or the one
`--python` names, which needs the `llvmlite` package. A host model holds the
`float` and `double` per-op rows, the formats a precision change offers a host
function, and no row of the arms that measure a GPU: `--cpu` refuses those
arms and `--gpu`.

Six arms write disjoint row families into the same file; `--only` runs a
subset, `--force` re-measures a family that is already there. A seventh,
`herbie-platform`, measures nothing: it translates the CSV into
`<csv>.herbie.rkt` beside it. It runs after `microbm` and in a full run, and
because it needs no device, `--only herbie-platform --out <csv>` also runs on a
machine that does not have the GPU the CSV was measured on.

| Arm | Rows | What is measured |
|---|---|---|
| microbm | `<op>,<type>,<cost>`, `wmma_mma_<shape>,<in>_<acc>,<rel>` | Saturated reciprocal throughput per (op, type), from dependent op chains on every SM, or on a host from JIT-compiled loops. On NVIDIA a tensor-core tile row exists when a real tiled WMMA GEMM of that scheme beats a scalar-FP64 GEMM of the same shape, and holds the ratio of their times; a compile reads only whether the row exists. Also writes the `# native_arch=` and `# scalar_types=` headers. |
| ozaki | `ozaki_dispatch_rel,nm<8..14>`, `ozaki_dispatch_rel,dgemm` | Host-dispatched Ozaki-II GEMM per modulus count, and the native cuBLAS DGEMM. |
| ozaki-rect | `ozaki_dispatch_rect_rel,nm<8..14>_m<M>k<K>`, `ozaki_dispatch_rect_rel,dgemm_m<M>n<N>k<K>` | The Ozaki-II dispatch at the `--ozaki-rect` shapes `MxK`, per MAC. Its cost is linear in the column count, so the row is keyed by (M, K). A shape `MxNxK` measures the native DGEMM of a call with exactly those extents (`dgemm_m<M>n<N>k<K>`): a DGEMM with few columns runs far below the square rate. The shapes are the ones the calls run at: a call whose three extents are constants of its compile and not a square is priced only from the rows of its own shape, and a compile without them warns with the `--ozaki-rect` argument that measures them; a square call is priced from the square rows (a native DGEMM one from its own row when the model has it). The shape of a call whose extents are run-time values is not known to the compile: its Ozaki-II candidates are priced from the rows of its profiled (M, K), its native DGEMM from the square row. Rows of shapes a run does not name are kept. |
| tcec | `tcec_dispatch_rel,fp16tcec` | Host-dispatched error-corrected GEMM, on the cuMpSGEMM the compiler is built with. A compiler built without cuMpSGEMM has no such dispatch, and the arm says so and writes no row. |
| direct | `direct_dispatch_rel,<in>_<acc>` | Host-dispatched reduced-precision cuBLAS GEMM. |
| inkernel | `wmma_inkernel_rel,<class>` | Tensor-core raises inside a kernel, one class at a time: the harness is compiled through `poseidon-clang++` with `-poseidon-apply-rewrites`, so what is timed is the code a real solve emits. |
| herbie-platform | none (writes `<csv>.herbie.rkt`) | Nothing. The per-op rows are translated into the Herbie platform the algebraic search ranks candidates by, so Herbie and the DP price the same device. A compile against a GPU CSV with no platform beside it aborts rather than search under another architecture's costs, and Herbie result caches are keyed by the platform name so one device's rewrites are never replayed for another. |

An AMD GPU is measured by the HIP builds of the same tools
(`poseidon-microbm-hip` and `poseidon-dispatch-calibrate-hip`, built when the
configuration names `POSEIDON_ROCM_PATH`, for the processor
`POSEIDON_AMDGPU_ARCH` names, default `gfx942`), which `poseidon-calibrate`
runs when the CUDA build finds no device with that id; the device is the one
`HIP_VISIBLE_DEVICES` (or `--gpu`) selects.
They write `cm_gfx<NNN>_<device>.csv` with the same row names in the same unit,
and a compile for that processor finds it by its triple. Rows the device has no
runtime or instruction for are absent, so the solver does not propose those
classes: the WMMA tile rows and the in-kernel raises (NVPTX-only), the
error-corrected dispatch (no HIP runtime) and `direct_dispatch_rel,tf32_f32`
(hipBLAS has no TF32 compute type). Its Herbie platform is `hip-gfx<NNN>`.

Every `*_rel` row is one number in one unit: the candidate's wall clock over the
scalar-FP64 GEMM's, at the same shape on the same device, so the DP compares an
in-kernel raise against a library dispatch directly. The in-kernel arm needs a
profile of the kernel it raises to know which candidates the pass proposes; the
profile decides which candidates exist, never what they measure. It is recorded
from the harness `inkernel_calibrate.cu` (installed under
`share/poseidon/tools/calibrate`) at a small size:

    poseidon-clang++ -poseidon-profile-generate -DCAL_N=256 inkernel_calibrate.cu -o harness
    POSEIDON_PROFILE_DIR=<dir> ./harness --profiling
    poseidon-calibrate --gpu 0 --only inkernel --profile <dir>

A class with no measured row is not proposed. The pass does not compose a price
for it out of per-MMA rows: a materialized raise also pays a scratch fill,
shared-memory staging and accumulator traffic that no per-MMA row contains.

## Format

One row per line, `opcode,type,cost`, plus `#`-prefixed headers. `native_arch`
is the architecture the file was measured on, a GPU's or the target-cpu of a
host; `scalar_types` lists the precisions the solver may propose on a GPU. A
row the compile needs, a header it reads, a line that is not a row, a row given
twice and a row of no family above each stop the compile, naming the model:
nothing is substituted for a part of it, and a misspelt row a candidate class
depends on does not silently withdraw the class. A new family is added to the
pass (`Core/CostTable.cpp`) with the arm that writes it. The DP cache is keyed
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
