#include <algorithm>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include <cudf/algorithm/scan_artifacts.cuh>
#include <cudf/utilities/span.hpp>

#include <rmm/thrust_rmm_allocator.h>

#include <thrust/iterator/constant_iterator.h>

class InclusiveCopyIfTest : public cudf::test::BaseFixture {
};

struct simple_seed_op {
  inline __device__ uint32_t operator()(uint32_t idx, uint32_t input)
  {
    printf("bid(%i) tid(%i): seed %u %u\n", blockIdx.x, threadIdx.x, idx, input);
    return 0;
  }
};

struct simple_scan_op {
  template <typename OutputCallback>
  inline __device__ uint32_t operator()(  //
    uint32_t lhs,
    uint32_t rhs,
    OutputCallback& output)  // TODO: make sure this can be passed by value without changing meaning
  {
    printf("bid(%i) tid(%i): scan %u %u\n", blockIdx.x, threadIdx.x, lhs, rhs);
    auto result = lhs + rhs;
    if (result % 3 == 0) {  // even? output it twice!
      output(result);
      output(result);
    }
    return result;
  }
};

struct simple_intersection_op {
  inline __device__ uint32_t operator()(uint32_t lhs, uint32_t rhs)
  {
    printf("bid(%i) tid(%i): intersect %u %u\n", blockIdx.x, threadIdx.x, lhs, rhs);
    return lhs + rhs;
  }
};

TEST_F(InclusiveCopyIfTest, CanScanSelectIf)
{
  auto input = thrust::make_constant_iterator<uint32_t>(1);

  auto seed_op      = simple_seed_op{};
  auto scan_op      = simple_scan_op{};
  auto intersect_op = simple_intersection_op{};

  // const uint32_t size = 1 << 24;
  const uint32_t input_size = 1 << 15;

  thrust::device_vector<uint32_t> d_input(input, input + input_size);

  auto d_result = scan_artifacts<uint32_t>(d_input.begin(),  //
                                           d_input.end(),
                                           seed_op,
                                           scan_op,
                                           intersect_op);

  ASSERT_EQ(static_cast<uint32_t>(input_size / 3) * 2, d_result.size());

  thrust::host_vector<uint32_t> h_result(d_result.size());
  cudaMemcpy(
    h_result.data(), d_result.data(), sizeof(uint32_t) * d_result.size(), cudaMemcpyDeviceToHost);

  for (uint32_t i = 0; i < h_result.size(); i++) {  //
    ASSERT_EQ(static_cast<uint32_t>((i / 2) * 3 + 3), h_result[i]);
  }
}

// struct successive_capitalization_state {
//   char curr;
//   char prev;
// };

// struct successive_capitalization_op {
//   inline constexpr successive_capitalization_state operator()(  //
//     successive_capitalization_state lhs,
//     successive_capitalization_state rhs)
//   {
//     return {rhs.curr, lhs.curr};
//   }

//   static inline constexpr bool is_capital(char value) { return value >= 'A' and value <= 'Z'; }

//   inline __device__ bool operator()(successive_capitalization_state value)
//   {
//     return is_capital(value.prev) and is_capital(value.curr);
//   }
// };

// TEST_F(InclusiveCopyIfTest, CanDetectSuccessiveCapitals)
// {
//   auto input_str = std::string("AbcDeFGLiJKlMnoP");

//   auto input = rmm::device_vector<successive_capitalization_state>(input_str.size());

//   std::transform(input_str.begin(),  //
//                  input_str.end(),
//                  input.begin(),
//                  [](char value) { return successive_capitalization_state{value}; });

//   auto op = successive_capitalization_op{};

//   auto d_result = scan_artifacts(  //
//     input.begin(),
//     input.end(),
//     op,
//     op);

//   auto h_result = thrust::host_vector<successive_capitalization_state>(d_result.size());

//   cudaMemcpy(h_result.data(),  //
//              d_result.data(),
//              d_result.size() * sizeof(successive_capitalization_state),
//              cudaMemcpyDeviceToHost);

//   ASSERT_EQ(static_cast<uint32_t>(3), h_result.size());

//   EXPECT_EQ(static_cast<char>('G'), h_result[0].curr);
//   EXPECT_EQ(static_cast<char>('L'), h_result[1].curr);
//   EXPECT_EQ(static_cast<char>('K'), h_result[2].curr);
// }

// enum class csv_token { unknown, comment_start, comment, new_record };

// enum class csv_state { nominal, commented };

// struct csv_token_parse_state {
//   char c;
//   csv_token token;
//   csv_state state;
// };

// csv_token_parse_state operator+(  //
//   csv_token_parse_state const& lhs,
//   csv_token_parse_state const& rhs)
// {
//   csv_token_parse_state result;

//   result.c = rhs.c;

//   switch (lhs.state) {
//     case csv_state::nominal: {
//       if (lhs.c == '\n') {
//         if (rhs.c == '#') {
//           return {rhs.c, csv_token::comment_start, csv_state::commented};
//         } else {
//           return {rhs.c, csv_token::new_record, csv_state::nominal};
//         }
//       }
//       return {rhs.c, csv_token::unknown, csv_state::nominal};
//     }
//     case csv_state::commented: {
//       if (lhs.c == '\n') {
//         if (rhs.c == '#') {
//           return {rhs.c, csv_token::comment_start, csv_state::commented};
//         } else {
//           return {rhs.c, csv_token::new_record, csv_state::nominal};
//         }
//       }
//       return {rhs.c, csv_token::comment, csv_state::commented};
//     }
//   }

//   return result;
// }

// struct csv_token_parse_op {
//   inline __device__ csv_token_parse_state operator()(  //
//     csv_token_parse_state lhs,
//     csv_token_parse_state rhs)
//   {
//     return rhs;
//   }
//   inline __device__ bool operator()(csv_token_parse_state value) { return true; }
// };

// TEST_F(InclusiveCopyIfTest, CanParseCsv)
// {
//   auto input_str = std::string(
//     "hello, world\n"
//     "new, record\n");

//   auto input = rmm::device_vector<successive_capitalization_state>(input_str.size());

//   std::transform(input_str.begin(),  //
//                  input_str.end(),
//                  input.begin(),
//                  [](char value) { return successive_capitalization_state{value}; });

//   auto op = successive_capitalization_op{};

//   auto d_result = scan_artifacts(  //
//     input.begin(),
//     input.end(),
//     op,
//     op);

//   auto h_result = thrust::host_vector<successive_capitalization_state>(d_result.size());

//   cudaMemcpy(h_result.data(),  //
//              d_result.data(),
//              d_result.size() * sizeof(successive_capitalization_state),
//              cudaMemcpyDeviceToHost);

//   ASSERT_EQ(static_cast<uint32_t>(3), h_result.size());

//   EXPECT_EQ(static_cast<char>('G'), h_result[0].curr);
//   EXPECT_EQ(static_cast<char>('L'), h_result[1].curr);
//   EXPECT_EQ(static_cast<char>('K'), h_result[2].curr);
// }

CUDF_TEST_PROGRAM_MAIN()
