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
#include <cudf/io/detail/csv_record_offsets.cuh>

#include <benchmark/benchmark.h>

#include <thrust/iterator/constant_iterator.h>

rmm::device_vector<char> to_device_vector(std::string input)
{
  return rmm::device_vector<char>(input.c_str(), input.c_str() + input.size());
}

static void BM_scan_state_machines(benchmark::State& state)
{
  auto const input_size = state.range(0);
  auto input            = thrust::make_constant_iterator<char>('a');
  auto d_input          = thrust::device_vector<char>(input, input + input_size);

  // auto d_input = to_device_vector("single\ncolumn\ncsv\n");

  for (auto _ : state) {
    cuda_event_timer raii(state, true);
    benchmark::DoNotOptimize(cudf::io::detail::csv_gather_row_offsets(d_input));
  }

  state.SetBytesProcessed(state.iterations() * input_size * sizeof(char));
}

class ScanStateMachineBenchmark : public cudf::benchmark {
};

#define SCAN_STATE_MACHINE_BM_BENCHMARK_DEFINE(name)                              \
  BENCHMARK_DEFINE_F(ScanStateMachineBenchmark, name)(::benchmark::State & state) \
  {                                                                               \
    BM_scan_state_machines(state);                                                \
  }                                                                               \
  BENCHMARK_REGISTER_F(ScanStateMachineBenchmark, name)                           \
    ->RangeMultiplier(32)                                                         \
    ->Range(1 << 10, 1 << 30)                                                     \
    ->UseManualTime()                                                             \
    ->Unit(benchmark::kMillisecond);

SCAN_STATE_MACHINE_BM_BENCHMARK_DEFINE(scan_state_machines);
