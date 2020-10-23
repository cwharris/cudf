#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include "gtest/gtest.h"
#include "scan_select_if.cuh"

#include <cudf/utilities/span.hpp>

#include <rmm/thrust_rmm_allocator.h>

#include <thrust/iterator/constant_iterator.h>

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

  const uint32_t size = 1 << 15;

  auto res = scan_select_if(input, input + size, op, op);

  cudaDeviceSynchronize();

  thrust::host_vector<uint32_t> h_result = res;

  // 4096 / 3 = 1365.333...
  ASSERT_EQ(static_cast<uint32_t>(10922), h_result.size());

  for (uint32_t i = 0; i < h_result.size(); i++) {  //
    ASSERT_EQ(static_cast<uint32_t>(i * 3 + 3), h_result[i]);
  }

  // FAIL();
}

struct complicated_op {
  struct state {
    char next;
    char prev;

    state(char c) : next(c), prev(0) {}
    state(char prev, char next) : next(next), prev(prev) {}
  };

  struct proj_op {
    /**
     * @brief upgrades input to stateful value.
     * @note called once per thread
     */
    inline state operator()(char value) { return state(value); }

    /**
     * @brief projects stateful value to output type
     * @note N calls for N results
     */
    inline char operator()(state result) { return result.next; }
  };

  struct scan_op {
    /**
     * @brief scans next input in to prior stateful value.
     * @note N-1 calls for N items per thread
     */
    inline state operator()(state lhs, char rhs) { return state(lhs.next, rhs); }

    /**
     * @brief scans two stateful values together
     * @note N-1 calls for N threads,
     */
    inline state operator()(state lhs, state rhs) { return state(lhs.next, rhs.next); }
  };

  struct pred_op {
    /**
     * @brief
     * @note N calls for N items
     */
    inline bool operator()(state rhs) { return true; }
  };
};

// struct csv_row_start_op {
//   inline __device__ uint32_t operator()(uint32_t lhs, uint32_t rhs) { return lhs + rhs; }
//   inline __device__ bool operator()(uint32_t value) { return true; }
// };

// TEST_F(InclusiveCopyIfTest, CanDetectCsvRowStart) {}

CUDF_TEST_PROGRAM_MAIN()
