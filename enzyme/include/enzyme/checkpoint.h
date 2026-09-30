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

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ENZYME_CKPT_ABI_VERSION 1

/* Values match Checkpointing.jl's ActionFlag. */
enum {
  ENZYME_CKPT_NONE = 0,
  ENZYME_CKPT_STORE = 1,
  ENZYME_CKPT_RESTORE = 2,
  ENZYME_CKPT_FORWARD = 3,
  ENZYME_CKPT_FIRSTUTURN = 4,
  ENZYME_CKPT_UTURN = 5,
  ENZYME_CKPT_ERROR = 6,
  ENZYME_CKPT_DONE = 7
};

/* Same layout as Checkpointing.jl's Action. */
typedef struct EnzymeCkptAction {
  int32_t flag;
  int64_t iteration;
  int64_t startiteration;
  int64_t cpnum;
} EnzymeCkptAction;

/* One piece of memory a snapshot holds. */
typedef struct EnzymeCkptRegion {
  void *ptr;
  uint64_t bytes;
  uint32_t addrspace;
  uint32_t flags;
} EnzymeCkptRegion;

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
  int64_t max_slots; /* largest number of slots in use at once */
  uint64_t max_bytes;
} EnzymeCkptStats;

typedef struct EnzymeCkptConfig {
  /* Revolve: number of snapshot slots. Periodic: number of segments. */
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
  void **slots;
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

static inline void enzyme_ckpt_store_put(enzyme_ckpt_store *st, int64_t slot,
                                         const EnzymeCkptRegion *regions,
                                         uint64_t nregions) {
  uint64_t r, off = 0;
  int64_t idx = slot + 2;
  if (slot < -2)
    enzyme_ckpt_fail("negative slot");
  if (idx >= st->nslots) {
    int64_t n = st->nslots ? st->nslots : 4, i;
    while (n <= idx)
      n *= 2;
    st->slots = (void **)realloc(st->slots, n * sizeof(void *));
    for (i = st->nslots; i < n; i++)
      st->slots[i] = NULL;
    st->nslots = n;
  }
  for (r = 0; r < nregions; r++)
    if (regions[r].addrspace != 0)
      enzyme_ckpt_fail("reference store only handles address space 0");
  if (enzyme_ckpt_on_disk(st, idx)) {
    char path[4096];
    FILE *f;
    enzyme_ckpt_path(st, idx, path, sizeof(path));
    f = fopen(path, "wb");
    if (!f)
      enzyme_ckpt_fail("cannot open spill file");
    for (r = 0; r < nregions; r++)
      if (fwrite(regions[r].ptr, 1, regions[r].bytes, f) != regions[r].bytes)
        enzyme_ckpt_fail("short write to spill file");
    fclose(f);
    st->slots[idx] = (void *)1;
  } else {
    if (!st->slots[idx])
      st->slots[idx] = malloc(st->bytes ? st->bytes : 1);
    for (r = 0; r < nregions; r++) {
      memcpy((char *)st->slots[idx] + off, regions[r].ptr, regions[r].bytes);
      off += regions[r].bytes;
    }
  }
  if (slot < 0)
    return;
  st->stats.stores++;
  if (slot + 1 > st->used)
    st->used = slot + 1;
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
    for (r = 0; r < nregions; r++)
      if (fread(regions[r].ptr, 1, regions[r].bytes, f) != regions[r].bytes)
        enzyme_ckpt_fail("short read from spill file");
    fclose(f);
  } else {
    for (r = 0; r < nregions; r++) {
      memcpy(regions[r].ptr, (char *)st->slots[idx] + off, regions[r].bytes);
      off += regions[r].bytes;
    }
  }
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
    }
  }
  free(st->slots);
  st->slots = NULL;
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
  s->acp = config->snapshots < nsteps ? config->snapshots : nsteps;
  if (s->acp < 1)
    s->acp = 1;
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
  return (k * p->steps) / p->segments;
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

/* Queue the reversal of segment k, whose start is already the state. The last
 * segment's last step is the first u-turn. */
static inline void enzyme_ckpt_queue_segment(enzyme_ckpt_periodic_state *p,
                                             int64_t k, int first) {
  int64_t s = enzyme_ckpt_seg_start(p, k), e = enzyme_ckpt_seg_start(p, k + 1);
  int64_t K = p->segments, j;
  for (j = s; j < e - 1; j++) {
    enzyme_ckpt_push(p, ENZYME_CKPT_STORE, j, j, K + (j - s));
    enzyme_ckpt_push(p, ENZYME_CKPT_FORWARD, j + 1, j, K + (j - s));
  }
  enzyme_ckpt_push(p, first ? ENZYME_CKPT_FIRSTUTURN : ENZYME_CKPT_UTURN, e,
                   e - 1, K + (e - 1 - s));
  for (j = e - 2; j >= s; j--) {
    enzyme_ckpt_push(p, ENZYME_CKPT_RESTORE, j, j, K + (j - s));
    enzyme_ckpt_push(p, ENZYME_CKPT_UTURN, j + 1, j, K + (j - s));
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
  p->segments = segments < nsteps ? segments : nsteps;
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
 * 1 + j: go back to the last and reverse from there. */
static inline void enzyme_ckpt_store_all_set_nsteps(void *state, int64_t n) {
  enzyme_ckpt_periodic_state *p = (enzyme_ckpt_periodic_state *)state;
  int64_t j;
  p->online = 0;
  p->steps = n;
  p->segment = 0;
  p->qlen = p->qpos = 0;
  enzyme_ckpt_push(p, ENZYME_CKPT_RESTORE, n - 1, n - 1, n);
  enzyme_ckpt_push(p, ENZYME_CKPT_FIRSTUTURN, n, n - 1, n);
  for (j = n - 2; j >= 0; j--) {
    enzyme_ckpt_push(p, ENZYME_CKPT_RESTORE, j, j, 1 + j);
    enzyme_ckpt_push(p, ENZYME_CKPT_UTURN, j + 1, j, 1 + j);
  }
}

static inline void enzyme_ckpt_periodic_next(void *state,
                                             EnzymeCkptAction *out) {
  enzyme_ckpt_periodic_state *p = (enzyme_ckpt_periodic_state *)state;
  if (p->online) {
    out->iteration = out->startiteration = p->pos;
    out->cpnum = 1 + p->pos;
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
