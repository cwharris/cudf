/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/algorithm/scan_state_machine.cuh>

#include <thrust/iterator/constant_iterator.h>

#include <algorithm>

template <typename T>
struct simple_seed_op {
  inline constexpr T operator()(uint32_t idx, T input) { return 0; }
};

template <typename T>
struct simple_scan_op {
  template <typename Callback>
  inline __device__ T operator()(T lhs, T rhs, Callback& output)
  {
    T result = lhs + rhs;

    if (result % 2 == 0) { output(result); }
    return result;
  }
};

template <typename T>
struct simple_intersection_op {
  inline __device__ T operator()(T lhs, T rhs) { return lhs + rhs; }
};

static void BM_scan_artifacts(benchmark::State& state)
{
  // using T = uint64_t;

  // uint32_t input_size = state.range(0);

  // auto seed_op      = simple_seed_op<T>{};
  // auto scan_op      = simple_scan_op<T>{};
  // auto intersect_op = simple_intersection_op<T>{};

  // auto input   = thrust::make_constant_iterator<T>(1);
  // auto d_input = thrust::device_vector<T>(input, input + input_size);

  // for (auto _ : state) {
  //   cuda_event_timer raii(state, true);
  //   auto d_result = scan_artifacts<T>(d_input.begin(),  //
  //                                     d_input.end(),
  //                                     seed_op,
  //                                     scan_op,
  //                                     intersect_op);
  // }

  // state.SetBytesProcessed(state.iterations() * input_size * sizeof(T));
}

class ScanArtifactsBenchmark : public cudf::benchmark {
};

#define DUMMY_BM_BENCHMARK_DEFINE(name)                                        \
  BENCHMARK_DEFINE_F(ScanArtifactsBenchmark, name)(::benchmark::State & state) \
  {                                                                            \
    BM_scan_artifacts(state);                                                  \
  }                                                                            \
  BENCHMARK_REGISTER_F(ScanArtifactsBenchmark, name)                           \
    ->Ranges({{1 << 7, 1 << 30}})                                              \
    ->UseManualTime()                                                          \
    ->Unit(benchmark::kMillisecond);

DUMMY_BM_BENCHMARK_DEFINE(scan_artifacts);
