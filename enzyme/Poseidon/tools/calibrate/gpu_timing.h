//===- gpu_timing.h - how the GPU calibrators time a launch ---------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The event timing ops_gpu.cu and stream.cu share, and the price in the unit
// of the per-operation rows (machine-wide cycles at the device clock, times 2)
// they derive from it.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_TOOLS_CALIBRATE_GPU_TIMING_H
#define POSEIDON_TOOLS_CALIBRATE_GPU_TIMING_H

#include "../../runtime/DeviceApi.h"
#include "../../runtime/profile/ProfileFormat.h"

#include <algorithm>
#include <cmath>

namespace gpu_timing {

/// The time in nanoseconds of one call of `launch`, every launch checked. An
/// AMD GPU's clock settles slowly after a change of load, so there `launch`
/// runs for kWarmMs before it is timed, a batch lasts at least kBatchMs, and
/// the time is the median of kBatches batches of at least `reps` calls.
/// Elsewhere it is the mean of `reps` calls after three untimed ones.
template <typename F> double timeNs(F launch, int reps) {
  gpuEvent_t e0, e1;
  POSEIDON_GPU_CHECK(gpuEventCreate(&e0));
  POSEIDON_GPU_CHECK(gpuEventCreate(&e1));
  auto batch = [&](int calls) {
    POSEIDON_GPU_CHECK(gpuEventRecord(e0));
    for (int i = 0; i < calls; ++i)
      launch();
    POSEIDON_GPU_CHECK(gpuEventRecord(e1));
    POSEIDON_GPU_CHECK(gpuEventSynchronize(e1));
    POSEIDON_GPU_CHECK(gpuGetLastError());
    float ms;
    POSEIDON_GPU_CHECK(gpuEventElapsedTime(&ms, e0, e1));
    return (double)ms;
  };
#ifdef __HIP__
  constexpr int kBatches = 7;
  constexpr double kWarmMs = 20.0, kBatchMs = 5.0;
  double ms = batch(reps);
  for (double warm = ms; warm < kWarmMs; warm += ms)
    ms = batch(reps);
  if (ms < kBatchMs)
    reps = (int)std::ceil(reps * kBatchMs / std::max(ms, 1e-3));
  double t[kBatches];
  for (int b = 0; b < kBatches; ++b)
    t[b] = batch(reps) * 1e6 / reps;
  std::sort(t, t + kBatches);
  const double ns = t[kBatches / 2];
#else
  for (int i = 0; i < 3; ++i)
    launch();
  POSEIDON_GPU_CHECK(gpuDeviceSynchronize());
  POSEIDON_GPU_CHECK(gpuGetLastError());
  const double ns = batch(reps) * 1e6 / reps;
#endif
  POSEIDON_GPU_CHECK(gpuEventDestroy(e0));
  POSEIDON_GPU_CHECK(gpuEventDestroy(e1));
  return ns;
}

/// The clock rate of device 0 in GHz, queried once. The rows are in cycles,
/// so a rate that is not positive, which would price every row at 0, stops the
/// calibration. From CUDA 13 on, gpuDeviceProp_t carries no clock rate.
inline double clockGhz() {
  static const double ghz = [] {
    int kHz = 0;
    POSEIDON_GPU_CHECK(gpuDeviceGetAttribute(&kHz, gpuDevAttrClockRate, 0));
    if (kHz <= 0)
      poseidon::fatal("device 0 reports a clock rate of %d kHz; the rows are "
                      "in cycles and need a positive one",
                      kHz);
    return (double)kHz / 1e6;
  }();
  return ghz;
}

/// The price of one unit of work in the unit of the per-operation rows, from
/// the times `t1` and `t2` in nanoseconds of two runs that differ by `units`
/// units: the difference of the two cancels the launch.
inline double cyclesPerUnit(double t1, double t2, double units) {
  return (t2 - t1) / units * clockGhz() * 2.0;
}

} // namespace gpu_timing

#endif // POSEIDON_TOOLS_CALIBRATE_GPU_TIMING_H
