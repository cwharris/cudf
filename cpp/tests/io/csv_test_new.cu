#include <cudf/io/detail/csv_record_offsets.cuh>
#include <cudf_test/base_fixture.hpp>
#include <limits>

class CsvStateMachineTest : public cudf::test::BaseFixture {
};

TEST_F(CsvStateMachineTest, CanTransitionCsvStates)
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

  ASSERT_EQ(d_row_offsets.size(), static_cast<uint32_t>(9));

  auto h_row_offsets = std::vector<uint64_t>(d_row_offsets.size());

  cudaStreamSynchronize(0);

  cudaMemcpy(h_row_offsets.data(),  //
             d_row_offsets.data(),
             d_row_offsets.size() * sizeof(uint64_t),
             cudaMemcpyDeviceToHost);

  EXPECT_EQ(static_cast<uint64_t>(0), h_row_offsets[0]);
  EXPECT_EQ(static_cast<uint64_t>(13), h_row_offsets[1]);
  EXPECT_EQ(static_cast<uint64_t>(29), h_row_offsets[2]);

  EXPECT_EQ(static_cast<uint64_t>(33), h_row_offsets[3]);
  EXPECT_EQ(static_cast<uint64_t>(46), h_row_offsets[4]);
  EXPECT_EQ(static_cast<uint64_t>(62), h_row_offsets[5]);

  EXPECT_EQ(static_cast<uint64_t>(66), h_row_offsets[6]);
  EXPECT_EQ(static_cast<uint64_t>(79), h_row_offsets[7]);
  EXPECT_EQ(static_cast<uint64_t>(95), h_row_offsets[8]);
}

TEST_F(CsvStateMachineTest, CanTransitionCsvStatesWithRowRange)
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

  auto d_row_offsets = cudf::io::detail::csv_gather_row_offsets(  //
    d_input,
    {},
    {
      0,
      std::numeric_limits<uint64_t>::max(),
      4,
      7,
    });

  ASSERT_EQ(d_row_offsets.size(), static_cast<uint32_t>(3));

  auto h_row_offsets = std::vector<uint64_t>(d_row_offsets.size());

  cudaStreamSynchronize(0);

  cudaMemcpy(h_row_offsets.data(),  //
             d_row_offsets.data(),
             d_row_offsets.size() * sizeof(uint64_t),
             cudaMemcpyDeviceToHost);

  EXPECT_EQ(static_cast<uint64_t>(33), h_row_offsets[0]);
  EXPECT_EQ(static_cast<uint64_t>(46), h_row_offsets[1]);
  EXPECT_EQ(static_cast<uint64_t>(62), h_row_offsets[2]);
}

TEST_F(CsvStateMachineTest, CanTransitionCsvStatesWithByteRange)
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

  auto d_row_offsets = cudf::io::detail::csv_gather_row_offsets(  //
    d_input,
    {},
    {33, 63});

  ASSERT_EQ(d_row_offsets.size(), static_cast<uint32_t>(3));

  auto h_row_offsets = std::vector<uint64_t>(d_row_offsets.size());

  cudaStreamSynchronize(0);

  cudaMemcpy(h_row_offsets.data(),  //
             d_row_offsets.data(),
             d_row_offsets.size() * sizeof(uint64_t),
             cudaMemcpyDeviceToHost);

  EXPECT_EQ(static_cast<uint64_t>(33), h_row_offsets[0]);
  EXPECT_EQ(static_cast<uint64_t>(46), h_row_offsets[1]);
  EXPECT_EQ(static_cast<uint64_t>(62), h_row_offsets[2]);
}

TEST_F(CsvStateMachineTest, CanTransitionStateSegments)
{
  using namespace cudf::io::detail;

  EXPECT_EQ(csv_state::record_end, static_cast<csv_state>(csv_superstate()));
}

TEST_F(CsvStateMachineTest, CanTransitionCsvStates2)
{
  using namespace cudf::io::detail;

  // auto a = csv_machine_state{csv_state::record_end, csv_state::record_end, 0};

  // auto a = csv_machine_state{csv_state::record_end, 67};
  // auto b = csv_machine_state{csv_state::record_end, 32};

  // auto result = a + b;

  // EXPECT_EQ(static_cast<uint64_t>(99), result.byte_count);
}

TEST_F(CsvStateMachineTest, CanTransitionCsvStates3)
{
  using namespace cudf::io::detail;

  auto a = csv_superstate() + csv_token::comment + csv_token::other;
  auto b = csv_superstate() + csv_token::newline + csv_token::other;

  auto result = a + b;

  EXPECT_EQ(csv_state::field, static_cast<csv_state>(result));
}

TEST_F(CsvStateMachineTest, CanTransitionCsvStates5)
{
  // using namespace cudf::io::detail;

  // auto scan_op = csv_aggregates_scan_op{};
  // auto agg     = csv_aggregates{};

  // agg = scan_op(agg, {csv_superstate(csv_state::field), 1});

  // EXPECT_EQ(agg.previous_state, csv_state::record_end);
  // EXPECT_EQ(agg.current_state, csv_state::field);
  // EXPECT_EQ(agg.current_position, static_cast<uint64_t>(1));
  // EXPECT_EQ(agg.record_begin, static_cast<uint64_t>(1));
  // EXPECT_EQ(agg.record_count, static_cast<uint64_t>(1));

  // agg = scan_op(agg, {csv_superstate(csv_state::field), 2});

  // EXPECT_EQ(agg.previous_state, csv_state::field);
  // EXPECT_EQ(agg.current_state, csv_state::field);
  // EXPECT_EQ(agg.current_position, static_cast<uint64_t>(2));
  // EXPECT_EQ(agg.record_begin, static_cast<uint64_t>(1));
  // EXPECT_EQ(agg.record_count, static_cast<uint64_t>(1));
}

CUDF_TEST_PROGRAM_MAIN()
