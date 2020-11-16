#include <cudf/io/detail/csv_record_offsets.cuh>
#include <cudf_test/base_fixture.hpp>
#include <limits>

class CsvStateMachineTest : public cudf::test::BaseFixture {
};

rmm::device_vector<char> to_device_vector(std::string input)
{
  return rmm::device_vector<char>(input.c_str(), input.c_str() + input.size());
}

template <typename T>
std::vector<uint64_t> to_host_vector(rmm::device_uvector<T> const& d_source, cudaStream_t stream)
{
  auto h_result = std::vector<T>(d_source.size());

  cudaMemcpyAsync(h_result.data(),  //
                  d_source.data(),
                  d_source.size() * sizeof(T),
                  cudaMemcpyDeviceToHost,
                  stream);

  return h_result;
}

TEST_F(CsvStateMachineTest, CanDetectTerminatedRecords)
{
  cudaStream_t stream = 0;
  auto d_input        = to_device_vector("single\ncolumn\ncsv\n");

  auto d_row_offsets = cudf::io::detail::csv_gather_row_offsets(d_input, {}, {}, stream);
  auto h_row_offsets = to_host_vector(d_row_offsets, stream);

  cudaStreamSynchronize(stream);

  ASSERT_EQ(h_row_offsets.size(), static_cast<uint32_t>(3));

  EXPECT_EQ(static_cast<uint64_t>(0), h_row_offsets[0]);
  EXPECT_EQ(static_cast<uint64_t>(7), h_row_offsets[1]);
  EXPECT_EQ(static_cast<uint64_t>(14), h_row_offsets[2]);
}

TEST_F(CsvStateMachineTest, CanTransitionCsvStates)
{
  cudaStream_t stream = 0;
  auto d_input        = to_device_vector(
    "hello, world\n"
    "and,\"not\nh,ing\"\n"
    "new\n"
    "hello, world\n"
    "and,\"not\nh,ing\"\n"
    "new\n"
    "hello, world\n"
    "and,\"not\nh,ing\"\n"
    "new\n");

  auto d_row_offsets = cudf::io::detail::csv_gather_row_offsets(d_input, {}, {}, stream);
  auto h_row_offsets = to_host_vector(d_row_offsets, stream);

  cudaStreamSynchronize(stream);

  ASSERT_EQ(h_row_offsets.size(), static_cast<uint32_t>(9));

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
  cudaStream_t stream = 0;
  auto d_input        = to_device_vector(
    "hello, world\n"
    "and,\"not\nh,ing\"\n"
    "new\n"
    "hello, world\n"
    "and,\"not\nh,ing\"\n"
    "new\n"
    "hello, world\n"
    "and,\"not\nh,ing\"\n"
    "new\n");

  auto d_row_offsets = cudf::io::detail::csv_gather_row_offsets(  //
    d_input,
    {},
    {
      0,
      std::numeric_limits<uint64_t>::max(),
      4,
      7,
    });

  auto h_row_offsets = to_host_vector(d_row_offsets, stream);

  cudaStreamSynchronize(stream);

  ASSERT_EQ(h_row_offsets.size(), static_cast<uint32_t>(3));

  EXPECT_EQ(static_cast<uint64_t>(33), h_row_offsets[0]);
  EXPECT_EQ(static_cast<uint64_t>(46), h_row_offsets[1]);
  EXPECT_EQ(static_cast<uint64_t>(62), h_row_offsets[2]);
}

TEST_F(CsvStateMachineTest, CanTransitionCsvStatesWithByteRange)
{
  cudaStream_t stream = 0;

  auto d_input = to_device_vector(
    "hello, world\n"
    "and,\"not\nh,ing\"\n"
    "new\n"
    "hello, world\n"
    "and,\"not\nh,ing\"\n"
    "new\n"
    "hello, world\n"
    "and,\"not\nh,ing\"\n"
    "new\n");

  auto d_row_offsets = cudf::io::detail::csv_gather_row_offsets(  //
    d_input,
    {},
    {33, 63});

  auto h_row_offsets = to_host_vector(d_row_offsets, stream);

  cudaStreamSynchronize(stream);

  ASSERT_EQ(h_row_offsets.size(), static_cast<uint32_t>(3));

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

  auto a = csv_superstate() + csv_token::comment + csv_token::other;
  auto b = csv_superstate() + csv_token::newline + csv_token::other;

  auto result = a + b;

  EXPECT_EQ(csv_state::field, static_cast<csv_state>(result));
}

CUDF_TEST_PROGRAM_MAIN()
