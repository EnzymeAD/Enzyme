/*
 * enzyme/checkpoint.h - checkpointed time loops for Enzyme's reverse mode.
 *
 * A time loop
 *
 *     for (int64_t i = start; i < start + n; i++) step(i, args...);
 *
 * written as
 *
 *     __enzyme_checkpoint_for((void *)step, start, n,
 *                             enzyme_scheme, &scheme, &config,
 *                             [enzyme_checkpoint_region, ptr, bytes,]...
 *                             args...);
 *
 * runs the same loop in the primal. Differentiated in reverse mode, Enzyme
 * does not keep the tape of every step. It asks the scheme which steps to
 * take snapshots before, which to recompute, and when to turn around, and
 * keeps the tape of one step at a time.
 *
 * Enzyme decides what a snapshot holds: the regions marked with
 * enzyme_checkpoint_region, plus every global the step writes, or reads while
 * later code may write it. The scheme decides when to snapshot and where the
 * snapshot is kept. A scheme that knows its state better than Enzyme can set
 * save_state/load_state and copy it itself.
 *
 * The action protocol is that of Revolve (Griewank and Walther, ACM TOMS
 * Alg. 799), and the flag values and the layout of EnzymeCkptAction are those
 * of Checkpointing.jl's ActionFlag and Action, so that its schemes can drive
 * this interface unchanged. Steps are counted from 0; step j is called with
 * i = start + j.
 *
 *   STORE      (iteration = c, cpnum = k): the state is the one before step c.
 *              Save it into slot k.
 *   FORWARD    (startiteration = a, iteration = b): run steps a, ..., b-1
 *              without taping.
 *   FIRSTUTURN (iteration = b): the state is the one before step b-1, the last
 *              step, whose adjoint is the first of the reverse sweep. Only in
 *              the forward sweep.
 *   UTURN      (iteration = b): the state is the one before step b-1. Run it
 *              with taping, then its adjoint.
 *   RESTORE    (iteration = c, cpnum = k): load slot k, the state before
 *              step c.
 *   DONE       the schedule is complete. With n = 0 it is the first action.
 *
 * Besides the slots the actions name, the driver uses two of its own, which
 * store and restore (or save_state and load_state) must accept:
 *
 *   slot -2    the state before the last step. At FIRSTUTURN the forward sweep
 *              stores it and runs the last step without taping; the reverse
 *              sweep starts by restoring it and differentiating that step.
 *   slot -1    the state the reverse sweep starts from, restored when the
 *              schedule is done, so that the primal state is left as the
 *              forward pass left it.
 *
 * Enzyme differentiates one step at a time, its forward and reverse passes
 * together, so no tape outlives a step.
 *
 * This header also has three reference schemes, used as
 * `enzyme_scheme, &EnzymeCkptRevolve, &config`:
 *
 *   EnzymeCkptRevolve   binomial checkpointing, config.snapshots slots
 *   EnzymeCkptBinomial  Enzyme-MLIR's binomial schedule, config.snapshots
 *                       slots
 *   EnzymeCkptPeriodic  config.snapshots segments; the steps of the segment
 *                       being reversed are all stored
 *   EnzymeCkptStoreAll  a snapshot before every step (periodic with one
 *                       segment); also for while loops
 *
 * They keep snapshots in memory, or past config.mem_budget bytes in files
 * under config.spill_dir.
 */
#ifndef ENZYME_CHECKPOINT_H
#define ENZYME_CHECKPOINT_H

#include "checkpoint_schedule.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ENZYME_CKPT_ABI_VERSION 1

/* The action flags are in checkpoint_schedule.h. */

/* Same layout as Checkpointing.jl's Action. */
typedef struct EnzymeCkptAction {
  int32_t flag;
  int64_t iteration;
  int64_t startiteration;
  int64_t cpnum;
} EnzymeCkptAction;

struct EnzymeCkptCallbacks;

/* A region with this flag is state a callback snapshots, rather than memory
 * copied: ptr and shadow are the primal and shadow of its root, bytes is 0. */
#define ENZYME_CKPT_REGION_CALLBACK 1u

/* One piece of memory a snapshot holds. shadow is its shadow in a
 * derivative's pass when it has one, else NULL. */
typedef struct EnzymeCkptRegion {
  void *ptr;
  uint64_t bytes;
  uint32_t addrspace;
  uint32_t flags;
  void *shadow;
  const struct EnzymeCkptCallbacks *callbacks;
} EnzymeCkptRegion;

/* The state of a callback region, a garbage-collected object graph say,
 * which the frontend snapshots itself. Its shadow holds references that
 * must point to the shadows of what the primal references: the forward
 * sweep runs steps without their derivatives, and a restore puts back the
 * primal's references only, so the drivers call sync before each step's
 * derivative runs and when the forward sweep is done. enter is called when
 * the forward sweep starts, leave when the reverse sweep is done. */
typedef struct EnzymeCkptCallbacks {
  void (*enter)(const struct EnzymeCkptCallbacks *cb, void *primal,
                void *shadow);
  void (*save)(const struct EnzymeCkptCallbacks *cb, void *primal,
               void *shadow, int64_t slot);
  void (*restore)(const struct EnzymeCkptCallbacks *cb, void *primal,
                  void *shadow, int64_t slot);
  void (*sync)(const struct EnzymeCkptCallbacks *cb, void *primal,
               void *shadow);
  void (*leave)(const struct EnzymeCkptCallbacks *cb, void *primal,
                void *shadow);
  void *data;
} EnzymeCkptCallbacks;

typedef struct EnzymeCheckpointScheme {
  uint32_t version; /* ENZYME_CKPT_ABI_VERSION */
  /* nsteps is -1 for a while loop. snapshot_bytes is the size of all regions. */
  void *(*init)(void *data, int64_t nsteps, uint64_t snapshot_bytes);
  void (*next_action)(void *state, EnzymeCkptAction *out);
  /* May be NULL when save_state/load_state are set. */
  void (*store)(void *state, int64_t slot, int64_t step,
                const EnzymeCkptRegion *regions, uint64_t nregions);
  void (*restore)(void *state, int64_t slot, int64_t step,
                  const EnzymeCkptRegion *regions, uint64_t nregions);
  /* While loops: the loop ran nsteps steps. May be NULL. */
  void (*set_nsteps)(void *state, int64_t nsteps);
  void (*finalize)(void *state);
  /* Optional: the scheme copies the step's arguments itself. env points at
   * the arguments after i, each followed by its shadow if it has one. */
  void (*save_state)(void *state, int64_t slot, int64_t step, void *env);
  void (*load_state)(void *state, int64_t slot, int64_t step, void *env);
  /* Optional, called after init: what the step accesses through the object
   * its first argument after i points to, so save_state/load_state can copy
   * just that. paths holds, for each access, the number n of pointer fields
   * followed from that object, their byte offsets, the byte offset of the
   * access in the object reached (-1: the whole object, or unknown), and
   * flags: 1 read, 2 written. An access that is not listed is not made. */
  void (*set_paths)(void *state, const int64_t *paths, uint64_t len);
} EnzymeCheckpointScheme;

extern int enzyme_scheme;
extern int enzyme_checkpoint_region;

void __enzyme_checkpoint_for(void *step, int64_t start, int64_t n, ...);

/* The loop
 *
 *     int64_t i = 0;
 *     while (step(i++, args...));
 *
 * with the same markers. Its number of steps is only known when step returns
 * false: init gets nsteps = -1, and set_nsteps the number of steps once the
 * loop has ended, after which the schedule continues from the final state to
 * its first turn. */
void __enzyme_checkpoint_while(void *step, ...);

/* ---------------------------------------------------------------------- */
/* Reference schemes                                                       */
/* ---------------------------------------------------------------------- */

typedef struct EnzymeCkptStats {
  int64_t forward_steps; /* steps run without taping */
  int64_t taped_steps;   /* steps run with taping (one per step) */
  int64_t stores;
  int64_t restores;
  int64_t max_slots; /* largest number of slots holding a snapshot */
  uint64_t max_bytes;
} EnzymeCkptStats;

typedef struct EnzymeCkptConfig {
  /* Revolve, Binomial: number of snapshot slots. Periodic: number of
   * segments. 0 or less: the default (see checkpoint_schedule.h). */
  int64_t snapshots;
  /* 1 prints a summary, 2 also prints every action to stderr. */
  int verbose;
  /* NULL keeps every snapshot in memory. */
  const char *spill_dir;
  /* With spill_dir: slots whose start is past this many bytes go to files. */
  uint64_t mem_budget;
  /* If not NULL, filled in when the schedule completes. */
  EnzymeCkptStats *stats;
} EnzymeCkptConfig;

typedef struct enzyme_ckpt_store {
  uint64_t bytes;
  int64_t nslots;
  /* Per slot, in memory: the host regions, and the device regions in a
   * buffer on the device. */
  void **slots;
  void **dslots;
  const EnzymeCkptConfig *config;
  int64_t used;
  EnzymeCkptStats stats;
} enzyme_ckpt_store;

static inline void enzyme_ckpt_fail(const char *msg) {
  fprintf(stderr, "enzyme checkpoint: %s\n", msg);
  abort();
}

/* The driver's slots -2 and -1 are kept at indices 0 and 1. */
static inline int enzyme_ckpt_on_disk(enzyme_ckpt_store *st, int64_t idx) {
  return st->config->spill_dir &&
         (uint64_t)idx * st->bytes >= st->config->mem_budget;
}

static inline void enzyme_ckpt_path(enzyme_ckpt_store *st, int64_t slot,
                                    char *buf, size_t len) {
  snprintf(buf, len, "%s/enzyme_ckpt_%p_%lld.bin", st->config->spill_dir,
           (void *)st, (long long)slot);
}

static inline void enzyme_ckpt_store_init(enzyme_ckpt_store *st,
                                          const EnzymeCkptConfig *config,
                                          uint64_t bytes) {
  memset(st, 0, sizeof(*st));
  st->bytes = bytes;
  st->config = config;
}

/* A region in an address space other than 0 is device memory. A slot in
 * memory keeps it on the device, in a buffer allocated at the slot's first
 * store and copied into and out of device to device; a slot spilled to disk
 * goes through host memory. With ENZYME_CKPT_CUDA defined where the schemes
 * are compiled, these are cudaMalloc, cudaFree and cudaMemcpy. */
#ifdef ENZYME_CKPT_CUDA
#include <cuda_runtime_api.h>
static inline void enzyme_ckpt_device_copy(void *dst, const void *src,
                                           uint64_t bytes) {
  if (cudaMemcpy(dst, src, bytes, cudaMemcpyDefault) != cudaSuccess)
    enzyme_ckpt_fail("cudaMemcpy of a snapshot failed");
}
static inline void *enzyme_ckpt_device_alloc(uint64_t bytes) {
  void *p = NULL;
  if (cudaMalloc(&p, bytes ? bytes : 1) != cudaSuccess)
    enzyme_ckpt_fail("cudaMalloc of a snapshot slot failed");
  return p;
}
static inline void enzyme_ckpt_device_free(void *p) { cudaFree(p); }
#else
static inline void enzyme_ckpt_device_copy(void *dst, const void *src,
                                           uint64_t bytes) {
  (void)dst;
  (void)src;
  (void)bytes;
  enzyme_ckpt_fail("snapshots of device memory need the schemes compiled "
                   "with ENZYME_CKPT_CUDA");
}
static inline void *enzyme_ckpt_device_alloc(uint64_t bytes) {
  (void)bytes;
  enzyme_ckpt_fail("snapshots of device memory need the schemes compiled "
                   "with ENZYME_CKPT_CUDA");
  return NULL;
}
static inline void enzyme_ckpt_device_free(void *p) { (void)p; }
#endif

static inline void enzyme_ckpt_copy_out(void *dst,
                                        const EnzymeCkptRegion *region) {
  if (region->addrspace)
    enzyme_ckpt_device_copy(dst, region->ptr, region->bytes);
  else
    memcpy(dst, region->ptr, region->bytes);
}

static inline void enzyme_ckpt_copy_in(const EnzymeCkptRegion *region,
                                       const void *src) {
  if (region->addrspace)
    enzyme_ckpt_device_copy(region->ptr, src, region->bytes);
  else
    memcpy(region->ptr, src, region->bytes);
}

static inline void enzyme_ckpt_store_put(enzyme_ckpt_store *st, int64_t slot,
                                         const EnzymeCkptRegion *regions,
                                         uint64_t nregions) {
  uint64_t r, off = 0;
  int64_t idx = slot + 2;
  int fresh;
  if (slot < -2)
    enzyme_ckpt_fail("negative slot");
  if (idx >= st->nslots) {
    int64_t n = st->nslots ? st->nslots : 4, i;
    while (n <= idx)
      n *= 2;
    st->slots = (void **)realloc(st->slots, n * sizeof(void *));
    st->dslots = (void **)realloc(st->dslots, n * sizeof(void *));
    for (i = st->nslots; i < n; i++)
      st->slots[i] = st->dslots[i] = NULL;
    st->nslots = n;
  }
  fresh = st->slots[idx] == NULL;
  if (enzyme_ckpt_on_disk(st, idx)) {
    char path[4096];
    FILE *f;
    enzyme_ckpt_path(st, idx, path, sizeof(path));
    f = fopen(path, "wb");
    if (!f)
      enzyme_ckpt_fail("cannot open spill file");
    for (r = 0; r < nregions; r++) {
      void *buf = regions[r].ptr;
      if (regions[r].addrspace) {
        buf = malloc(regions[r].bytes ? regions[r].bytes : 1);
        enzyme_ckpt_copy_out(buf, &regions[r]);
      }
      if (fwrite(buf, 1, regions[r].bytes, f) != regions[r].bytes)
        enzyme_ckpt_fail("short write to spill file");
      if (buf != regions[r].ptr)
        free(buf);
    }
    fclose(f);
    st->slots[idx] = (void *)1;
  } else {
    uint64_t hbytes = 0, dbytes = 0, doff = 0;
    for (r = 0; r < nregions; r++)
      *(regions[r].addrspace ? &dbytes : &hbytes) += regions[r].bytes;
    if (!st->slots[idx])
      st->slots[idx] = malloc(hbytes ? hbytes : 1);
    if (dbytes && !st->dslots[idx])
      st->dslots[idx] = enzyme_ckpt_device_alloc(dbytes);
    for (r = 0; r < nregions; r++) {
      if (regions[r].addrspace) {
        enzyme_ckpt_device_copy((char *)st->dslots[idx] + doff, regions[r].ptr,
                                regions[r].bytes);
        doff += regions[r].bytes;
      } else {
        memcpy((char *)st->slots[idx] + off, regions[r].ptr, regions[r].bytes);
        off += regions[r].bytes;
      }
    }
  }
  for (r = 0; r < nregions; r++)
    if (regions[r].flags & ENZYME_CKPT_REGION_CALLBACK)
      regions[r].callbacks->save(regions[r].callbacks, regions[r].ptr,
                                 regions[r].shadow, slot);
  if (slot < 0)
    return;
  st->stats.stores++;
  // Slots are kept until the end: those that hold a snapshot, not the highest
  // slot number, are what the schedule costs.
  if (fresh)
    st->used++;
  if (st->used > st->stats.max_slots) {
    st->stats.max_slots = st->used;
    st->stats.max_bytes = (uint64_t)st->used * st->bytes;
  }
}

static inline void enzyme_ckpt_store_get(enzyme_ckpt_store *st, int64_t slot,
                                         const EnzymeCkptRegion *regions,
                                         uint64_t nregions) {
  uint64_t r, off = 0;
  int64_t idx = slot + 2;
  if (slot < -2 || idx >= st->nslots || !st->slots[idx])
    enzyme_ckpt_fail("restore of a slot that was never stored");
  if (enzyme_ckpt_on_disk(st, idx)) {
    char path[4096];
    FILE *f;
    enzyme_ckpt_path(st, idx, path, sizeof(path));
    f = fopen(path, "rb");
    if (!f)
      enzyme_ckpt_fail("cannot open spill file");
    for (r = 0; r < nregions; r++) {
      void *buf = regions[r].addrspace
                      ? malloc(regions[r].bytes ? regions[r].bytes : 1)
                      : regions[r].ptr;
      if (fread(buf, 1, regions[r].bytes, f) != regions[r].bytes)
        enzyme_ckpt_fail("short read from spill file");
      if (buf != regions[r].ptr) {
        enzyme_ckpt_copy_in(&regions[r], buf);
        free(buf);
      }
    }
    fclose(f);
  } else {
    uint64_t doff = 0;
    for (r = 0; r < nregions; r++) {
      if (regions[r].addrspace) {
        enzyme_ckpt_device_copy(regions[r].ptr, (char *)st->dslots[idx] + doff,
                                regions[r].bytes);
        doff += regions[r].bytes;
      } else {
        memcpy(regions[r].ptr, (char *)st->slots[idx] + off, regions[r].bytes);
        off += regions[r].bytes;
      }
    }
  }
  for (r = 0; r < nregions; r++)
    if (regions[r].flags & ENZYME_CKPT_REGION_CALLBACK)
      regions[r].callbacks->restore(regions[r].callbacks, regions[r].ptr,
                                    regions[r].shadow, slot);
  if (slot >= 0)
    st->stats.restores++;
}

static inline void enzyme_ckpt_store_free(enzyme_ckpt_store *st) {
  int64_t i;
  for (i = 0; i < st->nslots; i++) {
    if (!st->slots[i])
      continue;
    if (enzyme_ckpt_on_disk(st, i)) {
      char path[4096];
      enzyme_ckpt_path(st, i, path, sizeof(path));
      remove(path);
    } else {
      free(st->slots[i]);
      if (st->dslots[i])
        enzyme_ckpt_device_free(st->dslots[i]);
    }
  }
  free(st->slots);
  free(st->dslots);
  st->slots = st->dslots = NULL;
  st->nslots = 0;
}

static inline const char *enzyme_ckpt_flag_name(int32_t flag) {
  switch (flag) {
  case ENZYME_CKPT_STORE:
    return "store";
  case ENZYME_CKPT_RESTORE:
    return "restore";
  case ENZYME_CKPT_FORWARD:
    return "forward";
  case ENZYME_CKPT_FIRSTUTURN:
    return "firstuturn";
  case ENZYME_CKPT_UTURN:
    return "uturn";
  case ENZYME_CKPT_DONE:
    return "done";
  default:
    return "error";
  }
}

static inline void enzyme_ckpt_trace(const EnzymeCkptConfig *config,
                                     enzyme_ckpt_store *st,
                                     const EnzymeCkptAction *a) {
  if (a->flag == ENZYME_CKPT_FORWARD)
    st->stats.forward_steps += a->iteration - a->startiteration;
  if (a->flag == ENZYME_CKPT_FIRSTUTURN || a->flag == ENZYME_CKPT_UTURN)
    st->stats.taped_steps++;
  if (config->verbose > 1)
    fprintf(stderr, "action %s %lld %lld %lld\n",
            enzyme_ckpt_flag_name(a->flag), (long long)a->iteration,
            (long long)a->startiteration, (long long)a->cpnum);
}

static inline void enzyme_ckpt_finish(const EnzymeCkptConfig *config,
                                      enzyme_ckpt_store *st) {
  if (config->verbose > 0)
    fprintf(stderr,
            "enzyme checkpoint: %lld forward steps, %lld taped steps, "
            "%lld stores, %lld restores, %lld slots (%llu bytes)\n",
            (long long)st->stats.forward_steps,
            (long long)st->stats.taped_steps, (long long)st->stats.stores,
            (long long)st->stats.restores, (long long)st->stats.max_slots,
            (unsigned long long)st->stats.max_bytes);
  if (config->stats)
    *config->stats = st->stats;
  enzyme_ckpt_store_free(st);
}

/* --- Revolve: a port of Checkpointing.jl's RevolveState next_action! --- */

typedef struct enzyme_ckpt_revolve_state {
  enzyme_ckpt_store store;
  const EnzymeCkptConfig *config;
  int64_t steps, tail, acp, cstart, cend, numfwd, numinv, numstore, rwcp,
      prevcend;
  int firstuturned;
  int64_t *stepof;
} enzyme_ckpt_revolve_state;

static inline void *enzyme_ckpt_revolve_init(void *data, int64_t nsteps,
                                             uint64_t bytes) {
  const EnzymeCkptConfig *config = (const EnzymeCkptConfig *)data;
  enzyme_ckpt_revolve_state *s;
  if (nsteps < 0)
    enzyme_ckpt_fail("revolve needs the number of steps");
  s = (enzyme_ckpt_revolve_state *)calloc(1, sizeof(*s));
  enzyme_ckpt_store_init(&s->store, config, bytes);
  s->config = config;
  s->steps = nsteps;
  s->tail = 1;
  s->acp = enzyme_ckpt_binomial_slots(nsteps, config->snapshots);
  s->cstart = 0;
  s->cend = nsteps;
  s->rwcp = -1;
  s->stepof = (int64_t *)calloc(s->acp + 1, sizeof(int64_t));
  return s;
}

static inline void enzyme_ckpt_revolve_next(void *state,
                                            EnzymeCkptAction *out) {
  enzyme_ckpt_revolve_state *r = (enzyme_ckpt_revolve_state *)state;
  int32_t flag = ENZYME_CKPT_NONE;
  int64_t prevcstart;
  int rwcptest;
  if (r->numinv == 0) {
    memset(r->stepof, 0, (r->acp + 1) * sizeof(int64_t));
    r->stepof[0] = r->cstart - 1;
  }
  prevcstart = r->cstart;
  r->numinv++;
  rwcptest = (r->rwcp == -1);
  if (!rwcptest)
    rwcptest = r->stepof[r->rwcp] != r->cstart;
  if (r->cend - r->cstart == 0) {
    if (r->rwcp == -1 || r->cstart == r->stepof[0]) {
      r->rwcp--;
      flag = ENZYME_CKPT_DONE;
    } else {
      r->cstart = r->stepof[r->rwcp];
      r->prevcend = r->cend;
      flag = ENZYME_CKPT_RESTORE;
    }
  } else if (r->cend - r->cstart == 1) {
    r->cend--;
    r->prevcend = r->cend;
    if (r->rwcp >= 0 && r->stepof[r->rwcp] == r->cstart)
      r->rwcp--;
    if (!r->firstuturned) {
      flag = ENZYME_CKPT_FIRSTUTURN;
      r->firstuturned = 1;
    } else {
      flag = ENZYME_CKPT_UTURN;
    }
  } else if (rwcptest) {
    r->rwcp++;
    if (r->rwcp + 1 > r->acp)
      enzyme_ckpt_fail("revolve: insufficient allowed checkpoints");
    r->stepof[r->rwcp] = r->cstart;
    r->numstore++;
    r->prevcend = r->cend;
    flag = ENZYME_CKPT_STORE;
  } else if (r->prevcend < r->cend && r->acp == r->rwcp + 1) {
    enzyme_ckpt_fail("revolve: insufficient allowed checkpoints");
  } else {
    int64_t availcp = r->acp - r->rwcp, reps = 0;
    double range = 1.0, bino1, bino2, bino3, bino4, bino5;
    if (availcp < 1)
      enzyme_ckpt_fail("revolve: insufficient allowed checkpoints");
    while (range < (double)(r->cend - r->cstart)) {
      reps++;
      range = range * (double)(reps + availcp) / (double)reps;
    }
    bino1 = range * reps / (availcp + reps);
    bino2 = availcp > 1 ? bino1 * availcp / (availcp + reps - 1) : 1.0;
    if (availcp == 1)
      bino3 = 0.0;
    else if (availcp > 2)
      bino3 = bino2 * (availcp - 1) / (availcp + reps - 2);
    else
      bino3 = 1.0;
    bino4 = bino2 * (reps - 1) / availcp;
    if (availcp < 3)
      bino5 = 0.0;
    else if (availcp > 3)
      bino5 = bino3 * (availcp - 1) / reps;
    else
      bino5 = 1.0;
    if ((double)(r->cend - r->cstart) <= bino1 + bino3)
      r->cstart = (int64_t)(r->cstart + bino4);
    else if ((double)(r->cend - r->cstart) >= range - bino5)
      r->cstart = (int64_t)(r->cstart + bino1);
    else
      r->cstart = (int64_t)(r->cend - bino2 - bino3);
    if (r->cstart == prevcstart)
      r->cstart = prevcstart + 1;
    if (r->cstart == r->steps)
      r->numfwd += (r->cstart - 1) - prevcstart + r->tail;
    else
      r->numfwd += r->cstart - prevcstart;
    flag = ENZYME_CKPT_FORWARD;
  }
  out->flag = flag;
  out->startiteration = prevcstart;
  if (flag == ENZYME_CKPT_FIRSTUTURN)
    out->iteration = r->cstart + r->tail;
  else if (flag == ENZYME_CKPT_UTURN)
    out->iteration = r->cstart + 1;
  else
    out->iteration = r->cstart;
  out->cpnum = r->rwcp;
  enzyme_ckpt_trace(r->config, &r->store, out);
}

static inline void enzyme_ckpt_revolve_store(void *state, int64_t slot,
                                             int64_t step,
                                             const EnzymeCkptRegion *regions,
                                             uint64_t nregions) {
  (void)step;
  enzyme_ckpt_store_put(&((enzyme_ckpt_revolve_state *)state)->store, slot,
                        regions, nregions);
}

static inline void enzyme_ckpt_revolve_restore(void *state, int64_t slot,
                                               int64_t step,
                                               const EnzymeCkptRegion *regions,
                                               uint64_t nregions) {
  (void)step;
  enzyme_ckpt_store_get(&((enzyme_ckpt_revolve_state *)state)->store, slot,
                        regions, nregions);
}

static inline void enzyme_ckpt_revolve_finalize(void *state) {
  enzyme_ckpt_revolve_state *r = (enzyme_ckpt_revolve_state *)state;
  enzyme_ckpt_finish(r->config, &r->store);
  free(r->stepof);
  free(r);
}

static const EnzymeCheckpointScheme EnzymeCkptRevolve = {
    ENZYME_CKPT_ABI_VERSION,
    enzyme_ckpt_revolve_init,
    enzyme_ckpt_revolve_next,
    enzyme_ckpt_revolve_store,
    enzyme_ckpt_revolve_restore,
    NULL,
    enzyme_ckpt_revolve_finalize,
    NULL,
    NULL,
    NULL};

/* --- Binomial: Enzyme-MLIR's binomial schedule. ---
 *
 * The schedule of cacheBinomial and reverseBinomial in Enzyme-MLIR's
 * LoopCheckpointing.h, as actions. Slots form a stack. The forward sweep
 * stores slot k at step pos[k] and advances
 * enzyme_ckpt_binomial_progress(steps left, slots left) steps, which sum to
 * the number of steps. To reverse step cur - 1 it restores the top slot and
 * replays to cur - 1, storing a slot at each step it starts an advance from,
 * the same way; the slots it stored are the stack from then on, and a slot
 * at cur - 1 itself is popped. The driver's first turn runs the last step
 * from the state before it, so the forward sweep stops a step short of the
 * end; reversing the last step replays to there, as Enzyme-MLIR does. The
 * actions are made one turn at a time. */

typedef struct enzyme_ckpt_binomial_state {
  enzyme_ckpt_store store;
  const EnzymeCkptConfig *config;
  int64_t steps, slots, sp, cur;
  int64_t *pos;
  EnzymeCkptAction *queue;
  int64_t qlen, qpos;
} enzyme_ckpt_binomial_state;

static inline void enzyme_ckpt_binomial_push(enzyme_ckpt_binomial_state *b,
                                             int32_t flag, int64_t iteration,
                                             int64_t start, int64_t cpnum) {
  EnzymeCkptAction *a = &b->queue[b->qlen++];
  a->flag = flag;
  a->iteration = iteration;
  a->startiteration = start;
  a->cpnum = cpnum;
}

/* Queue the reversal of step cur - 1. */
static inline void enzyme_ckpt_binomial_turn(enzyme_ckpt_binomial_state *b,
                                             int first) {
  int64_t cur = b->cur, capo = b->sp - 1, ck = b->pos[capo];
  int64_t q = ck, a = capo;
  enzyme_ckpt_binomial_push(b, ENZYME_CKPT_RESTORE, ck, ck, capo);
  while (q + 1 < cur) {
    int64_t rem = cur - q, left = b->slots - a, np, ub;
    /* The top slot already holds this state. */
    if (a != capo)
      enzyme_ckpt_binomial_push(b, ENZYME_CKPT_STORE, q, q, a);
    b->pos[a] = q;
    np = q + enzyme_ckpt_binomial_progress(rem, left < rem ? left : rem);
    ub = np == cur ? cur - 1 : np;
    if (ub > q)
      enzyme_ckpt_binomial_push(b, ENZYME_CKPT_FORWARD, ub, q, a);
    q = np;
    a++;
  }
  b->sp = a == capo ? capo : a;
  enzyme_ckpt_binomial_push(
      b, first ? ENZYME_CKPT_FIRSTUTURN : ENZYME_CKPT_UTURN, cur, cur - 1, 0);
  b->cur--;
}

static inline void *enzyme_ckpt_binomial_init(void *data, int64_t nsteps,
                                              uint64_t bytes) {
  const EnzymeCkptConfig *config = (const EnzymeCkptConfig *)data;
  enzyme_ckpt_binomial_state *b;
  int64_t k, q = 0;
  if (nsteps < 0)
    enzyme_ckpt_fail("the binomial schedule needs the number of steps");
  b = (enzyme_ckpt_binomial_state *)calloc(1, sizeof(*b));
  enzyme_ckpt_store_init(&b->store, config, bytes);
  b->config = config;
  b->steps = nsteps;
  b->slots = enzyme_ckpt_binomial_slots(nsteps, config->snapshots);
  b->pos = (int64_t *)calloc(b->slots, sizeof(int64_t));
  /* The forward sweep, then the first turn. */
  b->queue =
      (EnzymeCkptAction *)calloc(4 * b->slots + 8, sizeof(EnzymeCkptAction));
  if (nsteps == 0) {
    enzyme_ckpt_binomial_push(b, ENZYME_CKPT_DONE, 0, 0, -1);
    return b;
  }
  for (k = 0; k < b->slots && q < nsteps; k++) {
    int64_t rem = nsteps - q, left = b->slots - k, np, ub;
    enzyme_ckpt_binomial_push(b, ENZYME_CKPT_STORE, q, q, k);
    b->pos[k] = q;
    np = q + enzyme_ckpt_binomial_progress(rem, left < rem ? left : rem);
    ub = np == nsteps ? nsteps - 1 : np;
    if (ub > q)
      enzyme_ckpt_binomial_push(b, ENZYME_CKPT_FORWARD, ub, q, k);
    q = np;
  }
  b->sp = k;
  b->cur = nsteps;
  enzyme_ckpt_binomial_turn(b, 1);
  return b;
}

static inline void enzyme_ckpt_binomial_next(void *state,
                                             EnzymeCkptAction *out) {
  enzyme_ckpt_binomial_state *b = (enzyme_ckpt_binomial_state *)state;
  if (b->qpos == b->qlen) {
    b->qlen = b->qpos = 0;
    if (b->cur == 0)
      enzyme_ckpt_binomial_push(b, ENZYME_CKPT_DONE, 0, 0, -1);
    else
      enzyme_ckpt_binomial_turn(b, 0);
  }
  *out = b->queue[b->qpos++];
  enzyme_ckpt_trace(b->config, &b->store, out);
}

static inline void enzyme_ckpt_binomial_store(void *state, int64_t slot,
                                              int64_t step,
                                              const EnzymeCkptRegion *regions,
                                              uint64_t nregions) {
  (void)step;
  enzyme_ckpt_store_put(&((enzyme_ckpt_binomial_state *)state)->store, slot,
                        regions, nregions);
}

static inline void enzyme_ckpt_binomial_restore(void *state, int64_t slot,
                                                int64_t step,
                                                const EnzymeCkptRegion *regions,
                                                uint64_t nregions) {
  (void)step;
  enzyme_ckpt_store_get(&((enzyme_ckpt_binomial_state *)state)->store, slot,
                        regions, nregions);
}

static inline void enzyme_ckpt_binomial_finalize(void *state) {
  enzyme_ckpt_binomial_state *b = (enzyme_ckpt_binomial_state *)state;
  enzyme_ckpt_finish(b->config, &b->store);
  free(b->pos);
  free(b->queue);
  free(b);
}

static const EnzymeCheckpointScheme EnzymeCkptBinomial = {
    ENZYME_CKPT_ABI_VERSION,
    enzyme_ckpt_binomial_init,
    enzyme_ckpt_binomial_next,
    enzyme_ckpt_binomial_store,
    enzyme_ckpt_binomial_restore,
    NULL,
    enzyme_ckpt_binomial_finalize,
    NULL,
    NULL,
    NULL};

/* --- Periodic: config.snapshots segments. ---
 *
 * The forward sweep stores the start of every segment. Reversing a segment
 * restores its start and stores the state before each of its steps, so the
 * segment's steps are reversed from those. Slots 0..K-1 hold segment starts,
 * slots K.. the steps of the segment being reversed. With one segment every
 * step is stored once (EnzymeCkptStoreAll). */

typedef struct enzyme_ckpt_periodic_state {
  enzyme_ckpt_store store;
  const EnzymeCkptConfig *config;
  int64_t steps, segments;
  EnzymeCkptPeriodicSplit split;
  EnzymeCkptAction *queue;
  int64_t qlen, qpos, qcap;
  int64_t segment; /* the segment whose actions are queued */
  int first;
  /* A while loop still running: the next step, and whether to store before
   * it. */
  int online, pending;
  int64_t pos;
} enzyme_ckpt_periodic_state;

static inline int64_t enzyme_ckpt_seg_start(enzyme_ckpt_periodic_state *p,
                                            int64_t k) {
  return k >= p->segments ? p->steps : enzyme_ckpt_periodic_start(p->split, k);
}

static inline void enzyme_ckpt_push(enzyme_ckpt_periodic_state *p,
                                    int32_t flag, int64_t iteration,
                                    int64_t startiteration, int64_t cpnum) {
  EnzymeCkptAction *a;
  if (p->qlen == p->qcap) {
    p->qcap = p->qcap ? 2 * p->qcap : 16;
    p->queue = (EnzymeCkptAction *)realloc(
        p->queue, p->qcap * sizeof(EnzymeCkptAction));
  }
  a = &p->queue[p->qlen++];
  a->flag = flag;
  a->iteration = iteration;
  a->startiteration = startiteration;
  a->cpnum = cpnum;
}

/* The slot of the state before step j of segment k: the segment's own slot k
 * for its first step, then slots K on, shared by all segments. */
static inline int64_t enzyme_ckpt_seg_slot(enzyme_ckpt_periodic_state *p,
                                           int64_t k, int64_t j) {
  int64_t s = enzyme_ckpt_seg_start(p, k);
  return j == s ? k : p->segments + (j - s - 1);
}

/* Queue the reversal of segment k, whose start is already the state. The last
 * segment's last step is the first u-turn. Every segment but the last already
 * has its start in slot k, from the forward sweep. */
static inline void enzyme_ckpt_queue_segment(enzyme_ckpt_periodic_state *p,
                                             int64_t k, int first) {
  int64_t s = enzyme_ckpt_seg_start(p, k), e = enzyme_ckpt_seg_start(p, k + 1);
  int64_t j;
  for (j = s; j < e - 1; j++) {
    if (j != s || first)
      enzyme_ckpt_push(p, ENZYME_CKPT_STORE, j, j,
                       enzyme_ckpt_seg_slot(p, k, j));
    enzyme_ckpt_push(p, ENZYME_CKPT_FORWARD, j + 1, j,
                     enzyme_ckpt_seg_slot(p, k, j));
  }
  enzyme_ckpt_push(p, first ? ENZYME_CKPT_FIRSTUTURN : ENZYME_CKPT_UTURN, e,
                   e - 1, enzyme_ckpt_seg_slot(p, k, e - 1));
  for (j = e - 2; j >= s; j--) {
    enzyme_ckpt_push(p, ENZYME_CKPT_RESTORE, j, j,
                     enzyme_ckpt_seg_slot(p, k, j));
    enzyme_ckpt_push(p, ENZYME_CKPT_UTURN, j + 1, j,
                     enzyme_ckpt_seg_slot(p, k, j));
  }
}

static inline void *enzyme_ckpt_periodic_init_k(const EnzymeCkptConfig *config,
                                                int64_t nsteps, uint64_t bytes,
                                                int64_t segments) {
  enzyme_ckpt_periodic_state *p;
  int64_t k;
  if (nsteps < 0)
    enzyme_ckpt_fail("periodic checkpointing needs the number of steps");
  p = (enzyme_ckpt_periodic_state *)calloc(1, sizeof(*p));
  enzyme_ckpt_store_init(&p->store, config, bytes);
  p->config = config;
  p->steps = nsteps;
  /* The segments Enzyme-MLIR's periodic checkpointing makes. */
  p->split = enzyme_ckpt_periodic_split(nsteps, segments);
  p->segments = enzyme_ckpt_periodic_segments(p->split);
  if (p->segments < 1)
    p->segments = 1;
  if (nsteps == 0) {
    enzyme_ckpt_push(p, ENZYME_CKPT_DONE, 0, 0, -1);
    return p;
  }
  for (k = 0; k + 1 < p->segments; k++) {
    enzyme_ckpt_push(p, ENZYME_CKPT_STORE, enzyme_ckpt_seg_start(p, k),
                     enzyme_ckpt_seg_start(p, k), k);
    enzyme_ckpt_push(p, ENZYME_CKPT_FORWARD, enzyme_ckpt_seg_start(p, k + 1),
                     enzyme_ckpt_seg_start(p, k), k);
  }
  p->segment = p->segments - 1;
  enzyme_ckpt_queue_segment(p, p->segment, 1);
  return p;
}

static inline void *enzyme_ckpt_periodic_init(void *data, int64_t nsteps,
                                              uint64_t bytes) {
  const EnzymeCkptConfig *config = (const EnzymeCkptConfig *)data;
  return enzyme_ckpt_periodic_init_k(config, nsteps, bytes, config->snapshots);
}

static inline void *enzyme_ckpt_store_all_init(void *data, int64_t nsteps,
                                               uint64_t bytes) {
  enzyme_ckpt_periodic_state *p;
  if (nsteps >= 0)
    return enzyme_ckpt_periodic_init_k((const EnzymeCkptConfig *)data, nsteps,
                                       bytes, 1);
  /* A while loop: store before every step until it ends. */
  p = (enzyme_ckpt_periodic_state *)calloc(1, sizeof(*p));
  enzyme_ckpt_store_init(&p->store, (const EnzymeCkptConfig *)data, bytes);
  p->config = (const EnzymeCkptConfig *)data;
  p->steps = -1;
  p->segments = 1;
  p->online = 1;
  p->pending = 1;
  return p;
}

/* The loop ended after n steps, with the state before each stored in slot
 * j: go back to the last and reverse from there. */
static inline void enzyme_ckpt_store_all_set_nsteps(void *state, int64_t n) {
  enzyme_ckpt_periodic_state *p = (enzyme_ckpt_periodic_state *)state;
  int64_t j;
  p->online = 0;
  p->steps = n;
  p->segment = 0;
  p->qlen = p->qpos = 0;
  enzyme_ckpt_push(p, ENZYME_CKPT_RESTORE, n - 1, n - 1, n - 1);
  enzyme_ckpt_push(p, ENZYME_CKPT_FIRSTUTURN, n, n - 1, n - 1);
  for (j = n - 2; j >= 0; j--) {
    enzyme_ckpt_push(p, ENZYME_CKPT_RESTORE, j, j, j);
    enzyme_ckpt_push(p, ENZYME_CKPT_UTURN, j + 1, j, j);
  }
}

static inline void enzyme_ckpt_periodic_next(void *state,
                                             EnzymeCkptAction *out) {
  enzyme_ckpt_periodic_state *p = (enzyme_ckpt_periodic_state *)state;
  if (p->online) {
    out->iteration = out->startiteration = p->pos;
    out->cpnum = p->pos;
    if (p->pending) {
      out->flag = ENZYME_CKPT_STORE;
    } else {
      out->flag = ENZYME_CKPT_FORWARD;
      out->iteration = ++p->pos;
    }
    p->pending = !p->pending;
    enzyme_ckpt_trace(p->config, &p->store, out);
    return;
  }
  if (p->qpos == p->qlen) {
    p->qlen = p->qpos = 0;
    if (p->segment == 0 || p->steps == 0) {
      enzyme_ckpt_push(p, ENZYME_CKPT_DONE, 0, 0, -1);
    } else {
      int64_t k = --p->segment;
      enzyme_ckpt_push(p, ENZYME_CKPT_RESTORE, enzyme_ckpt_seg_start(p, k),
                       enzyme_ckpt_seg_start(p, k), k);
      enzyme_ckpt_queue_segment(p, k, 0);
    }
  }
  *out = p->queue[p->qpos++];
  enzyme_ckpt_trace(p->config, &p->store, out);
}

static inline void enzyme_ckpt_periodic_store(void *state, int64_t slot,
                                              int64_t step,
                                              const EnzymeCkptRegion *regions,
                                              uint64_t nregions) {
  (void)step;
  enzyme_ckpt_store_put(&((enzyme_ckpt_periodic_state *)state)->store, slot,
                        regions, nregions);
}

static inline void enzyme_ckpt_periodic_restore(
    void *state, int64_t slot, int64_t step, const EnzymeCkptRegion *regions,
    uint64_t nregions) {
  (void)step;
  enzyme_ckpt_store_get(&((enzyme_ckpt_periodic_state *)state)->store, slot,
                        regions, nregions);
}

static inline void enzyme_ckpt_periodic_finalize(void *state) {
  enzyme_ckpt_periodic_state *p = (enzyme_ckpt_periodic_state *)state;
  enzyme_ckpt_finish(p->config, &p->store);
  free(p->queue);
  free(p);
}

static const EnzymeCheckpointScheme EnzymeCkptPeriodic = {
    ENZYME_CKPT_ABI_VERSION,
    enzyme_ckpt_periodic_init,
    enzyme_ckpt_periodic_next,
    enzyme_ckpt_periodic_store,
    enzyme_ckpt_periodic_restore,
    NULL,
    enzyme_ckpt_periodic_finalize,
    NULL,
    NULL,
    NULL};

static const EnzymeCheckpointScheme EnzymeCkptStoreAll = {
    ENZYME_CKPT_ABI_VERSION,
    enzyme_ckpt_store_all_init,
    enzyme_ckpt_periodic_next,
    enzyme_ckpt_periodic_store,
    enzyme_ckpt_periodic_restore,
    enzyme_ckpt_store_all_set_nsteps,
    enzyme_ckpt_periodic_finalize,
    NULL,
    NULL,
    NULL};

/* The scheme of a loop annotated with
 * [[enzyme_checkpointing_enable("binomial" or "regular", count)]]: Revolve
 * for mode 2, Periodic for mode 1. Enzyme calls it where it cannot find the
 * schemes in the module; a program whose annotated loops are in files that do
 * not include this header gets it from one that does. */
__attribute__((weak, used)) const EnzymeCheckpointScheme *
__enzyme_checkpoint_builtin(int64_t mode) {
  switch (mode) {
  case ENZYME_CKPT_SCHEDULE_REVOLVE:
    return &EnzymeCkptRevolve;
  case ENZYME_CKPT_SCHEDULE_STORE_ALL:
    return &EnzymeCkptStoreAll;
  case ENZYME_CKPT_SCHEDULE_BINOMIAL:
    return &EnzymeCkptBinomial;
  default:
    return &EnzymeCkptPeriodic;
  }
}

/* --- A schedule without a driver. ---
 *
 * For a caller that keeps the snapshots itself and needs only the actions,
 * with scalar arguments (Enzyme-MLIR, whose snapshots live in its own
 * buffers, calls these from the IR it generates):
 *
 *   void *h = __enzyme_ckpt_schedule_begin(mode, budget, nsteps);
 *   while (__enzyme_ckpt_schedule_next(h) != ENZYME_CKPT_DONE)
 *     ... __enzyme_ckpt_schedule_iteration(h), _start(h), _slot(h) ...
 *   __enzyme_ckpt_schedule_end(h);
 *
 * mode is a schedule of checkpoint_schedule.h: 1 for EnzymeCkptPeriodic, 2
 * for EnzymeCkptRevolve, 3 for EnzymeCkptStoreAll and 4 for
 * EnzymeCkptBinomial; budget is config.snapshots, its default if it is not
 * positive, as for an annotated loop without a count. The scheme's store and
 * restore are not called: at STORE the caller saves the state into slot
 * _slot(h), at RESTORE it loads it back. The slots the actions name are 0 to
 * __enzyme_ckpt_schedule_slots(h) - 1. Slots -2 and -1 of the driver are the
 * caller's own business.
 *
 * The definitions are compiled where ENZYME_CHECKPOINT_RUNTIME is defined,
 * as weak symbols, so that more than one file of a program may do so. With
 * ENZYME_CKPT_VERBOSE=1 in the environment a schedule prints a summary when
 * it ends, with 2 also every action. */
void *__enzyme_ckpt_schedule_begin(int64_t mode, int64_t budget,
                                   int64_t nsteps);
int32_t __enzyme_ckpt_schedule_next(void *h);
int32_t __enzyme_ckpt_schedule_flag(void *h);
int64_t __enzyme_ckpt_schedule_iteration(void *h);
int64_t __enzyme_ckpt_schedule_start(void *h);
int64_t __enzyme_ckpt_schedule_slot(void *h);
int64_t __enzyme_ckpt_schedule_slots(void *h);
void __enzyme_ckpt_schedule_end(void *h);

#ifdef ENZYME_CHECKPOINT_RUNTIME

typedef struct enzyme_ckpt_schedule {
  const EnzymeCheckpointScheme *scheme;
  void *state;
  EnzymeCkptConfig config;
  EnzymeCkptAction action;
  EnzymeCkptStats stats;
  int verbose;
  int64_t slots;
} enzyme_ckpt_schedule;

__attribute__((weak)) void *
__enzyme_ckpt_schedule_begin(int64_t mode, int64_t budget, int64_t nsteps) {
  enzyme_ckpt_schedule *s;
  const char *verbose = getenv("ENZYME_CKPT_VERBOSE");
  int64_t k, len;
  if (nsteps < 0)
    enzyme_ckpt_fail("a schedule needs the number of steps");
  s = (enzyme_ckpt_schedule *)calloc(1, sizeof(*s));
  s->config.snapshots = budget;
  s->config.stats = &s->stats;
  s->verbose = verbose ? atoi(verbose) : 0;
  /* The slots the reference schemes name, as they count them in init. */
  switch (mode) {
  case ENZYME_CKPT_SCHEDULE_PERIODIC:
  case ENZYME_CKPT_SCHEDULE_STORE_ALL: {
    EnzymeCkptPeriodicSplit split = enzyme_ckpt_periodic_split(
        nsteps, mode == ENZYME_CKPT_SCHEDULE_STORE_ALL ? 1 : budget);
    s->scheme = mode == ENZYME_CKPT_SCHEDULE_PERIODIC ? &EnzymeCkptPeriodic
                                                      : &EnzymeCkptStoreAll;
    k = enzyme_ckpt_periodic_segments(split);
    if (k < 1)
      k = 1;
    len = split.inner;
    /* Segment starts, then the steps of one segment but its first. */
    s->slots = k + (len > 1 ? len - 1 : 0);
    break;
  }
  case ENZYME_CKPT_SCHEDULE_REVOLVE:
  case ENZYME_CKPT_SCHEDULE_BINOMIAL:
    s->scheme = mode == ENZYME_CKPT_SCHEDULE_REVOLVE ? &EnzymeCkptRevolve
                                                     : &EnzymeCkptBinomial;
    s->slots = enzyme_ckpt_binomial_slots(nsteps, budget);
    break;
  default:
    enzyme_ckpt_fail("unknown built-in scheme");
  }
  s->state = s->scheme->init(&s->config, nsteps, 0);
  s->action.flag = ENZYME_CKPT_NONE;
  return s;
}

__attribute__((weak)) int32_t __enzyme_ckpt_schedule_next(void *h) {
  enzyme_ckpt_schedule *s = (enzyme_ckpt_schedule *)h;
  s->scheme->next_action(s->state, &s->action);
  if (s->action.flag == ENZYME_CKPT_ERROR || s->action.flag == ENZYME_CKPT_NONE)
    enzyme_ckpt_fail("the scheme returned no action");
  if ((s->action.flag == ENZYME_CKPT_STORE ||
       s->action.flag == ENZYME_CKPT_RESTORE) &&
      (s->action.cpnum < 0 || s->action.cpnum >= s->slots))
    enzyme_ckpt_fail("the scheme named a slot out of range");
  if (s->verbose > 1)
    fprintf(stderr, "action %s %lld %lld %lld\n",
            enzyme_ckpt_flag_name(s->action.flag),
            (long long)s->action.iteration, (long long)s->action.startiteration,
            (long long)s->action.cpnum);
  return s->action.flag;
}

__attribute__((weak)) int32_t __enzyme_ckpt_schedule_flag(void *h) {
  return ((enzyme_ckpt_schedule *)h)->action.flag;
}

__attribute__((weak)) int64_t __enzyme_ckpt_schedule_iteration(void *h) {
  return ((enzyme_ckpt_schedule *)h)->action.iteration;
}

__attribute__((weak)) int64_t __enzyme_ckpt_schedule_start(void *h) {
  return ((enzyme_ckpt_schedule *)h)->action.startiteration;
}

__attribute__((weak)) int64_t __enzyme_ckpt_schedule_slot(void *h) {
  return ((enzyme_ckpt_schedule *)h)->action.cpnum;
}

__attribute__((weak)) int64_t __enzyme_ckpt_schedule_slots(void *h) {
  return ((enzyme_ckpt_schedule *)h)->slots;
}

__attribute__((weak)) void __enzyme_ckpt_schedule_end(void *h) {
  enzyme_ckpt_schedule *s = (enzyme_ckpt_schedule *)h;
  /* Fills in s->stats, but for the stores and restores, which the scheme
   * counts when it makes them itself. */
  s->scheme->finalize(s->state);
  if (s->verbose > 0)
    fprintf(stderr,
            "enzyme checkpoint: %lld forward steps, %lld taped steps, "
            "%lld slots\n",
            (long long)s->stats.forward_steps, (long long)s->stats.taped_steps,
            (long long)s->slots);
  free(s);
}

#endif /* ENZYME_CHECKPOINT_RUNTIME */

/* --- Tapenade's binomial scheduler (ADFirstAidKit adBinomial.c). ---
 *
 * With ENZYME_CKPT_ADBINOMIAL defined, EnzymeCkptADBinomial drives the loop
 * with the schedule Tapenade's $AD BINOMIAL-CKP directive would use, for a
 * like-for-like comparison with a Tapenade adjoint. adBinomial.c is not part
 * of Enzyme: link the copy that comes with Tapenade (MITgcm ships one in
 * tools/TAP_support/ADFirstAidKit). adBinomial keeps its sessions on a static
 * stack, so loops using it may nest but must not interleave, and it allows
 * about 98 snapshots in all. */
#ifdef ENZYME_CKPT_ADBINOMIAL

extern void adBinomial_init(int length, int nbSnap, int firstStep);
extern int adBinomial_next(int *action, int *step);

enum {
  ENZYME_ADB_PUSHSNAP = 1,
  ENZYME_ADB_LOOKSNAP = 2,
  ENZYME_ADB_POPSNAP = 3,
  ENZYME_ADB_ADVANCE = 4,
  ENZYME_ADB_FIRSTTURN = 5,
  ENZYME_ADB_TURN = 6
};

typedef struct enzyme_ckpt_adbinomial_state {
  enzyme_ckpt_store store;
  const EnzymeCkptConfig *config;
  int64_t steps;
  /* The snapshot stack: the step each snapshot was taken before. */
  int64_t depth;
  int64_t pos[128];
  int done;
} enzyme_ckpt_adbinomial_state;

static inline void *enzyme_ckpt_adbinomial_init(void *data, int64_t nsteps,
                                                uint64_t bytes) {
  const EnzymeCkptConfig *config = (const EnzymeCkptConfig *)data;
  enzyme_ckpt_adbinomial_state *s;
  int64_t snaps = config->snapshots < 1 ? 1 : config->snapshots;
  if (nsteps < 0)
    enzyme_ckpt_fail("adBinomial needs the number of steps");
  if (snaps > 97)
    enzyme_ckpt_fail("adBinomial allows at most 97 snapshots");
  s = (enzyme_ckpt_adbinomial_state *)calloc(1, sizeof(*s));
  enzyme_ckpt_store_init(&s->store, config, bytes);
  s->config = config;
  s->steps = nsteps;
  if (nsteps == 0)
    s->done = 1;
  else
    adBinomial_init((int)nsteps, (int)snaps, 1);
  return s;
}

/* adBinomial counts steps from 1 and issues one ADVANCE per step. */
static inline void enzyme_ckpt_adbinomial_next(void *state,
                                               EnzymeCkptAction *out) {
  enzyme_ckpt_adbinomial_state *s = (enzyme_ckpt_adbinomial_state *)state;
  int action, step;
  out->cpnum = s->depth - 1;
  if (s->done || !adBinomial_next(&action, &step)) {
    s->done = 1;
    out->flag = ENZYME_CKPT_DONE;
    out->iteration = out->startiteration = 0;
  } else {
    switch (action) {
    case ENZYME_ADB_PUSHSNAP:
      if (s->depth == 128)
        enzyme_ckpt_fail("adBinomial snapshot stack overflow");
      out->flag = ENZYME_CKPT_STORE;
      out->iteration = out->startiteration = step;
      out->cpnum = s->depth;
      s->pos[s->depth++] = step;
      break;
    case ENZYME_ADB_LOOKSNAP:
    case ENZYME_ADB_POPSNAP:
      out->flag = ENZYME_CKPT_RESTORE;
      out->iteration = out->startiteration = s->pos[s->depth - 1];
      if (action == ENZYME_ADB_POPSNAP)
        s->depth--;
      break;
    case ENZYME_ADB_ADVANCE:
      out->flag = ENZYME_CKPT_FORWARD;
      out->startiteration = step - 1;
      out->iteration = step;
      break;
    case ENZYME_ADB_FIRSTTURN:
    case ENZYME_ADB_TURN:
      out->flag = action == ENZYME_ADB_FIRSTTURN ? ENZYME_CKPT_FIRSTUTURN
                                                 : ENZYME_CKPT_UTURN;
      out->startiteration = step - 1;
      out->iteration = step;
      break;
    default:
      enzyme_ckpt_fail("unknown adBinomial action");
    }
  }
  enzyme_ckpt_trace(s->config, &s->store, out);
}

static inline void enzyme_ckpt_adbinomial_store(
    void *state, int64_t slot, int64_t step, const EnzymeCkptRegion *regions,
    uint64_t nregions) {
  (void)step;
  enzyme_ckpt_store_put(&((enzyme_ckpt_adbinomial_state *)state)->store, slot,
                        regions, nregions);
}

static inline void enzyme_ckpt_adbinomial_restore(
    void *state, int64_t slot, int64_t step, const EnzymeCkptRegion *regions,
    uint64_t nregions) {
  (void)step;
  enzyme_ckpt_store_get(&((enzyme_ckpt_adbinomial_state *)state)->store, slot,
                        regions, nregions);
}

static inline void enzyme_ckpt_adbinomial_finalize(void *state) {
  enzyme_ckpt_adbinomial_state *s = (enzyme_ckpt_adbinomial_state *)state;
  enzyme_ckpt_finish(s->config, &s->store);
  free(s);
}

static const EnzymeCheckpointScheme EnzymeCkptADBinomial = {
    ENZYME_CKPT_ABI_VERSION,
    enzyme_ckpt_adbinomial_init,
    enzyme_ckpt_adbinomial_next,
    enzyme_ckpt_adbinomial_store,
    enzyme_ckpt_adbinomial_restore,
    NULL,
    enzyme_ckpt_adbinomial_finalize,
    NULL,
    NULL,
    NULL};

#endif /* ENZYME_CKPT_ADBINOMIAL */

#ifdef __cplusplus
}
#endif

#endif /* ENZYME_CHECKPOINT_H */
