#include <algorithm>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include <cudf/algorithm/scan_state_machine.cuh>
#include <cudf/utilities/span.hpp>
#include "rmm/device_buffer.hpp"
#include "rmm/device_scalar.hpp"
#include "rmm/device_uvector.hpp"

#include <rmm/thrust_rmm_allocator.h>

#include <thrust/iterator/constant_iterator.h>

class ScanStateMachineTest : public cudf::test::BaseFixture {
};

struct simple_output {
  dfa_output<uint32_t> a;
  dfa_output<double> b;

  inline constexpr simple_output operator+(simple_output other) const
  {
    return {a + other.a, b + other.b};
  }
};

struct simple_state {
  uint32_t sum;
  inline constexpr simple_state operator+(simple_state other) const { return {sum + other.sum}; }
  inline constexpr simple_state operator+(uint32_t input) const { return {sum + input}; }
};

struct simple_aggregate {
  uint32_t sum;
  inline constexpr simple_aggregate operator+(simple_aggregate const rhs) const { return *this; }
};

struct simple_scan_state_op {
  inline constexpr simple_state operator()(simple_state prev_state, uint32_t rhs) const
  {
    return prev_state + rhs;
  }
};
struct simple_scan_aggregates_op {
  inline constexpr simple_aggregate operator()(simple_aggregate agg, simple_state state) const
  {
    return {state.sum};
  }
};

struct simple_output_op {
  template <bool output_enabled>
  inline constexpr simple_output operator()(simple_output out, simple_aggregate const agg)
  {
    if (agg.sum % 3 == 0) { out.a.emit<output_enabled>(agg.sum); }
    if (agg.sum % 2 == 0) { out.b.emit<output_enabled>(agg.sum * 2.0); }

    return out;
  }

  // TODO: add a "final state" operator?
};

TEST_F(ScanStateMachineTest, CanScanSimpleState)
{
  auto input = thrust::make_constant_iterator<uint32_t>(1);

  auto scan_state_op      = simple_scan_state_op{};
  auto scan_aggregates_op = simple_scan_aggregates_op{};
  auto output_op          = simple_output_op{};

  const uint32_t input_size             = (1 << 15) + 4;
  const uint32_t expected_output_size_a = input_size / 3;
  const uint32_t expected_output_size_b = input_size / 2;

  thrust::device_vector<uint32_t> d_input(input, input + input_size);

  auto d_inout_state      = rmm::device_scalar<simple_state>();
  auto d_inout_aggregates = rmm::device_scalar<simple_aggregate>();
  auto d_inout_outputs    = rmm::device_scalar<simple_output>();

  rmm::device_buffer temp_storage;

  // phase 1: count outputs.
  scan_state_machine(temp_storage,  //
                     d_input.begin(),
                     d_input.end(),
                     d_inout_state.data(),
                     d_inout_aggregates.data(),
                     d_inout_outputs.data(),
                     scan_state_op,
                     scan_aggregates_op,
                     output_op);

  auto h_output       = d_inout_outputs.value();
  auto h_output_state = d_inout_state.value();

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

  d_inout_state.set_value({});
  d_inout_aggregates.set_value({});
  d_inout_outputs.set_value(h_output);

  scan_state_machine(temp_storage,  //
                     d_input.begin(),
                     d_input.end(),
                     d_inout_state.data(),
                     d_inout_aggregates.data(),
                     d_inout_outputs.data(),
                     scan_state_op,
                     scan_aggregates_op,
                     output_op);

  h_output       = d_inout_outputs.value();
  h_output_state = d_inout_state.value();

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
    EXPECT_EQ(static_cast<uint32_t>(i * 3) + 3, h_output_a[i]);
  }

  for (uint32_t i = 0; i < h_output_b.size(); i++) {
    ASSERT_EQ(static_cast<double>(i * 4.0) + 8, h_output_b[i]);
  }
}

CUDF_TEST_PROGRAM_MAIN()
