#include <algorithm>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include <cudf/algorithm/scan_state_machine.cuh>
#include <cudf/io/detail/csv_record_offsets.cuh>
#include <cudf/utilities/span.hpp>
#include "rmm/device_buffer.hpp"
#include "rmm/device_scalar.hpp"
#include "rmm/device_uvector.hpp"

#include <rmm/thrust_rmm_allocator.h>

#include <thrust/iterator/constant_iterator.h>

class ScanStateMachineTest : public cudf::test::BaseFixture {
};

struct simple_output {
  fsm_output<uint32_t> a;
  fsm_output<double> b;

  inline __device__ simple_output operator+(simple_output other) const
  {
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
    return {
      sum + other.sum,
    };
  }

  inline __device__ simple_state operator+(uint32_t input) const
  {
    return {
      sum + input,
    };
  }
};

struct simple_seed_op {
  inline __device__ simple_state operator()(uint32_t position) { return {}; }
};

struct simple_step_op {
  inline __device__ simple_state operator()(simple_state prev_state, uint32_t rhs)
  {
    return prev_state + rhs;
  }
};

struct simple_output_op {
  template <bool output_enabled>
  inline __device__ simple_output
  operator()(simple_output out, simple_state prev, simple_state next, uint32_t rhs)
  {
    if (prev.sum % 3 == 0) { out.a.emit<output_enabled>(prev.sum); }
    if (prev.sum % 2 == 0) { out.b.emit<output_enabled>(prev.sum * 2.0); }

    return out;
  }

  // // TODO: add a finalizer
  // template <typename output_enabled>
  // simple_output operator()(simple_output out, simple_state final){}
};

struct simple_join_op {
  inline __device__ simple_state operator()(simple_state lhs, simple_state rhs)  //
  {
    return lhs + rhs;
  }
};

TEST_F(ScanStateMachineTest, CanScanSimpleState)
{
  auto input = thrust::make_constant_iterator<uint32_t>(1);

  auto seed_op   = simple_seed_op{};
  auto step_op   = simple_step_op{};
  auto join_op   = simple_join_op{};
  auto output_op = simple_output_op{};

  const uint32_t input_size             = (1 << 10) + 4;
  const uint32_t expected_output_size_a = input_size / 3 + 1;
  const uint32_t expected_output_size_b = input_size / 2;

  thrust::device_vector<uint32_t> d_input(input, input + input_size);

  auto d_output_state = rmm::device_scalar<simple_state>();
  auto d_output       = rmm::device_scalar<simple_output>();

  rmm::device_buffer temp_storage;

  // phase 1: count outputs.
  scan_state_machine(temp_storage,  //
                     d_input.begin(),
                     d_input.end(),
                     d_output_state.data(),
                     d_output.data(),
                     seed_op,
                     step_op,
                     join_op,
                     output_op);

  auto h_output       = d_output.value();
  auto h_output_state = d_output_state.value();

  EXPECT_EQ(input_size, h_output_state.sum);
  EXPECT_EQ(input_size, h_output_state.sum);

  ASSERT_EQ(expected_output_size_a, h_output.a.output_count);
  ASSERT_EQ(expected_output_size_b, h_output.b.output_count);

  // phase 2: allocate outputs

  auto output_a = rmm::device_uvector<uint32_t>(h_output.a.output_count, 0);
  auto output_b = rmm::device_uvector<double>(h_output.b.output_count, 0);

  h_output                 = {};
  h_output.a.output_buffer = output_a.data();
  h_output.b.output_buffer = output_b.data();

  d_output.set_value(h_output);
  d_output_state.set_value({});

  scan_state_machine(temp_storage,  //
                     d_input.begin(),
                     d_input.end(),
                     d_output_state.data(),
                     d_output.data(),
                     seed_op,
                     step_op,
                     join_op,
                     output_op);

  h_output       = d_output.value();
  h_output_state = d_output_state.value();

  EXPECT_EQ(input_size, h_output_state.sum);
  EXPECT_EQ(input_size, h_output_state.sum);

  ASSERT_EQ(expected_output_size_a, h_output.a.output_count);
  ASSERT_EQ(expected_output_size_b, h_output.b.output_count);

  ASSERT_EQ(output_a.data(), h_output.a.output_buffer);
  ASSERT_EQ(output_b.data(), h_output.b.output_buffer);

  auto h_output_a = std::vector<uint32_t>(h_output.a.output_count);
  auto h_output_b = std::vector<double>(h_output.b.output_count);

  cudaMemcpy(h_output_a.data(),
             h_output.a.output_buffer,
             h_output.a.output_count * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);

  cudaMemcpy(h_output_b.data(),
             h_output.b.output_buffer,
             h_output.b.output_count * sizeof(double),
             cudaMemcpyDeviceToHost);

  for (uint32_t i = 0; i < h_output_a.size(); i++) {
    EXPECT_EQ(static_cast<uint32_t>(i * 3), h_output_a[i]);
  }

  for (uint32_t i = 0; i < h_output_b.size(); i++) {
    ASSERT_EQ(static_cast<double>(i * 4.0), h_output_b[i]);
  }
}

TEST_F(ScanStateMachineTest, CanTransitionCsvStates)
{
  auto input = std::string(
    "hello, world\n"
    "and,\"not\nh,ing\"\n"
    "new\n"
    "hello, world\n"
    "and,\"not\nh,ing\"\n"
    "new\n"
    "hello, world\n"
    "and,\"not\nh,ing\"\n"
    "new\n");

  auto d_input = rmm::device_vector<char>(input.c_str(), input.c_str() + input.size());

  auto d_row_offsets = cudf::io::detail::csv_gather_row_offsets(d_input);

  ASSERT_EQ(static_cast<uint32_t>(9), d_row_offsets.size());

  auto h_row_offsets = std::vector<uint32_t>(d_row_offsets.size());

  cudaStreamSynchronize(0);

  cudaMemcpy(h_row_offsets.data(),  //
             d_row_offsets.data(),
             d_row_offsets.size() * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);

  EXPECT_EQ(static_cast<uint32_t>(0), h_row_offsets[0]);
  EXPECT_EQ(static_cast<uint32_t>(13), h_row_offsets[1]);
  EXPECT_EQ(static_cast<uint32_t>(29), h_row_offsets[2]);

  EXPECT_EQ(static_cast<uint32_t>(33), h_row_offsets[3]);
  EXPECT_EQ(static_cast<uint32_t>(46), h_row_offsets[4]);
  EXPECT_EQ(static_cast<uint32_t>(62), h_row_offsets[5]);

  EXPECT_EQ(static_cast<uint32_t>(66), h_row_offsets[6]);
  EXPECT_EQ(static_cast<uint32_t>(79), h_row_offsets[7]);
  EXPECT_EQ(static_cast<uint32_t>(95), h_row_offsets[8]);
}

TEST_F(ScanStateMachineTest, CanTransitionCsvStates2)
{
  using namespace cudf::io::detail;
  auto a = csv_machine_state(5, csv_state::record_end);
  auto b = csv_machine_state(2, csv_state::record_end);

  auto result = a & b;

  ASSERT_EQ(static_cast<uint32_t>(1), result.num_states);
  EXPECT_EQ(static_cast<uint32_t>(7), result.position);
  EXPECT_EQ(csv_state::record_end, result.states[0].head);
  EXPECT_EQ(csv_state::record_end, result.states[0].tail);
}

TEST_F(ScanStateMachineTest, CanTransitionCsvStates3)
{
  using namespace cudf::io::detail;
  auto a = csv_machine_state(4, csv_state_segment(csv_state::record_end, csv_state::field));
  auto b = csv_machine_state(8, csv_state_segment(csv_state::field, csv_state::field_end));

  auto result = a & b;

  ASSERT_EQ(static_cast<uint32_t>(1), result.num_states);
  EXPECT_EQ(static_cast<uint32_t>(12), result.position);
  EXPECT_EQ(csv_state::record_end, result.states[0].head);
  EXPECT_EQ(csv_state::field_end, result.states[0].tail);
}

CUDF_TEST_PROGRAM_MAIN()
