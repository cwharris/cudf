#include <algorithm>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include <cudf/algorithm/csv_gpu_row_count.cuh>
#include <cudf/algorithm/scan_artifacts.cuh>
#include <cudf/utilities/span.hpp>
#include "gtest/gtest.h"
#include "rmm/device_scalar.hpp"
#include "rmm/device_uvector.hpp"

#include <rmm/thrust_rmm_allocator.h>

#include <thrust/iterator/constant_iterator.h>

class InclusiveCopyIfTest : public cudf::test::BaseFixture {
};

struct simple_outputs {
  fsm_output<uint32_t> a;
  fsm_output<double> b;

  inline __device__ simple_outputs operator+(simple_outputs other) const
  {
    // printf("bid(%i) tid(%i): simple_outputs (%i %i) + (%i %i)\n",  //
    //        blockIdx.x,
    //        threadIdx.x,
    //        a.output_count,
    //        b.output_count,
    //        other.a.output_count,
    //        other.b.output_count);

    return {
      a + other.a,
      b + other.b,
    };
  }
};

struct simple_state {
  uint32_t sum;
  inline __device__ simple_state operator+(simple_state other) const
  {
    // printf("bid(%i) tid(%i): simple_state %i + %i\n",  //
    //        blockIdx.x,
    //        threadIdx.x,
    //        sum,
    //        other.sum);

    return {
      sum + other.sum,
    };
  }
};

struct simple_state_seed_op {
  inline __device__ simple_state operator()(uint32_t idx, uint32_t input)  //
  {
    // printf("bid(%i) tid(%i): simple_state_seed_op %i + %i\n",  //
    //        blockIdx.x,
    //        threadIdx.x,
    //        idx,
    //        input);

    return {};
  }
};

struct simple_state_step_op {
  template <bool output_enabled>
  inline __device__ simple_state operator()(  //
    simple_outputs& outputs,
    simple_state prev_state,
    uint32_t rhs)
  {
    // printf("bid(%i) tid(%i): simple_state_step_op (%i, %i), %i, %i\n",  //
    //        blockIdx.x,
    //        threadIdx.x,
    //        outputs.a.output_count,
    //        outputs.b.output_count,
    //        prev_state.sum,
    //        rhs);

    auto state = simple_state{
      prev_state.sum + rhs,
    };

    if (prev_state.sum % 3 == 0) {
      outputs.a.emit<output_enabled>(state.sum);
      outputs.b.emit<output_enabled>(state.sum * 2.0);
    }

    return state;
  }
};

struct simple_state_join_op {
  inline __device__ simple_state operator()(simple_state lhs, simple_state rhs)  //
  {
    // printf("bid(%i) tid(%i): simple_state_step_op %i => %i\n",  //
    //        blockIdx.x,
    //        threadIdx.x,
    //        lhs.sum,
    //        rhs.sum);

    return lhs + rhs;
  }
};

TEST_F(InclusiveCopyIfTest, CanScanSelectIf)
{
  auto input = thrust::make_constant_iterator<uint32_t>(1);

  auto seed_op = simple_state_seed_op{};
  auto step_op = simple_state_step_op{};
  auto join_op = simple_state_join_op{};

  const uint32_t input_size = 1 << 10;

  thrust::device_vector<uint32_t> d_input(input, input + input_size);

  auto d_output_state = rmm::device_scalar<simple_state>();
  auto d_output       = rmm::device_scalar<simple_outputs>();

  rmm::device_buffer temp_storage;

  // phase 1: count outputs.
  temp_storage = scan_artifacts(std::move(temp_storage),  //
                                d_input.begin(),
                                d_input.end(),
                                d_output_state.data(),
                                d_output.data(),
                                seed_op,
                                step_op,
                                join_op);

  auto h_outputs = d_output.value();

  EXPECT_EQ(static_cast<uint32_t>(input_size), d_output_state.value().sum);
  EXPECT_EQ(static_cast<uint32_t>(input_size), d_output_state.value().sum);

  EXPECT_EQ(static_cast<uint32_t>(input_size / 3), h_outputs.a.output_count);
  EXPECT_EQ(static_cast<uint32_t>(input_size / 3), h_outputs.b.output_count);

  // phase 2: allocate outputs

  rmm::device_uvector<uint32_t> output_a(h_outputs.a.output_count, 0);
  rmm::device_uvector<double> output_b(h_outputs.a.output_count, 0);

  h_outputs.a.output_buffer = output_a.data();
  h_outputs.b.output_buffer = output_b.data();
}

TEST_F(InclusiveCopyIfTest, CanTransitionCsvStates)
{
  // auto input = std::string("hello, world");

  // auto d_input = rmm::device_vector<char>(input.c_str(), input.c_str() + input.size());

  // auto d_row_offsets = csv_gather_row_offsets(d_input);

  // thrust::host_vector<uint32_t> h_row_offsets(d_row_offsets.size());

  // cudaMemcpy(h_row_offsets.data(),  //
  //            d_row_offsets.data(),
  //            d_row_offsets.size() * sizeof(char),
  //            cudaMemcpyDeviceToHost);

  // ASSERT_EQ(static_cast<uint32_t>(0), h_row_offsets.size());

  // auto d_result = scan_artifacts<uint32_t>(d_input.begin(),  //
  //                                          d_input.end(),
  //                                          seed_op,
  //                                          scan_op,
  //                                          intersect_op);

  // thrust::host_vector<uint32_t> h_result(d_result.size());
  // cudaMemcpy(
  //   h_result.data(), d_result.data(), sizeof(uint32_t) * d_result.size(),
  //   cudaMemcpyDeviceToHost);

  // for (uint32_t i = 0; i < h_result.size(); i++) {  //
  //   ASSERT_EQ(static_cast<uint32_t>((i / 2) * 3 + 3), h_result[i]);
  // }
}

CUDF_TEST_PROGRAM_MAIN()
