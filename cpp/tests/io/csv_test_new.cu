#include <cudf/io/detail/csv_record_offsets.cuh>
#include <cudf_test/base_fixture.hpp>
#include <limits>

/*
// uint64_t *row_ctx,
// device_span<uint64_t> const offsets_out,
// size_t chunk_size,
// size_t parse_pos,
// const parse_options_view &options,
// cudaStream_t stream
// options.terminator,
// options.delimiter,
// options.quotechar ? options.quotechar : 0x100,
// options.comment   ? options.comment   : 0x100);
device_span<char const> const data,
size_t start_offset,
size_t data_size,
size_t byte_range_start,
size_t byte_range_end,
size_t skip_rows,
*/

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

  ASSERT_EQ(static_cast<uint32_t>(9), d_row_offsets.size());

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

  ASSERT_EQ(static_cast<uint32_t>(3), d_row_offsets.size());

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

  ASSERT_EQ(static_cast<uint32_t>(3), d_row_offsets.size());

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

CUDF_TEST_PROGRAM_MAIN()
