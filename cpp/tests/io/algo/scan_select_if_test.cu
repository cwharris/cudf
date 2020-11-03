#include <algorithm>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include "gtest/gtest.h"
#include "scan_select_if.cuh"
#include "thrust/transform.h"

#include <cudf/utilities/span.hpp>

#include <rmm/thrust_rmm_allocator.h>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

class InclusiveCopyIfTest : public cudf::test::BaseFixture {
};

struct simple_op {
  inline __device__ uint32_t operator()(uint32_t lhs, uint32_t rhs) { return lhs + rhs; }
  inline __device__ bool operator()(uint32_t value) { return value % 3 == 0; }
};

TEST_F(InclusiveCopyIfTest, CanScanSelectIf)
{
  auto input = thrust::make_constant_iterator<uint32_t>(1);

  auto op = simple_op{};

  const uint32_t size = 4096;

  thrust::host_vector<uint32_t> h_result = scan_select_if(input, input + size, op, op);

  // 4096 / 3 = 1365.333...
  ASSERT_EQ(static_cast<uint32_t>(1365), h_result.size());

  for (uint32_t i = 0; i < 1365; i++) {  //
    EXPECT_EQ(static_cast<uint32_t>(i * 3 + 3), h_result[i]);
  }
}

struct csv_row_start_state {
  uint32_t idx;
  bool flipped;
  char c;
};

struct csv_row_start_op {
  inline __device__ csv_row_start_state operator()(  //
    csv_row_start_state lhs,
    csv_row_start_state rhs)
  {
    auto flipped = lhs.flipped;

    if (rhs.c == '"') {
      flipped = not flipped;
      ;
    }

    if (rhs.flipped) {
      flipped = not flipped;
      ;
    }

    return {rhs.idx, rhs.flipped ? not lhs.flipped : lhs.flipped, rhs.c};
  }
  inline __device__ bool operator()(csv_row_start_state value)
  {  //
    return not value.flipped && value.c == '\n';
  }
};

TEST_F(InclusiveCopyIfTest, CanDetectCsvRowStart)
{
  auto message    = std::string("abcd\ncdef\n\",,\n\n,,\",\n");
  auto count_iter = thrust::make_counting_iterator<uint32_t>(0);

  rmm::device_vector<csv_row_start_state> input(message.size());

  std::transform(
    message.begin(), message.end(), count_iter, input.begin(), [](char value, uint32_t idx) {
      return csv_row_start_state{idx, false, value};
    });

  auto op = csv_row_start_op{};

  thrust::host_vector<csv_row_start_state> h_result =
    scan_select_if(input.begin(), input.end(), op, op);

  ASSERT_EQ(static_cast<uint32_t>(3), h_result.size());
}

CUDF_TEST_PROGRAM_MAIN()
