/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#pragma once

namespace cudf {
namespace dfa {
namespace output {
namespace detail {

struct string_column_output {
  template <bool output_enabled>
  inline constexpr void append(char value);
  template <bool output_enabled>
  inline constexpr void terminate(bool is_null = false);
  inline constexpr string_column_output operator+(string_column_output other) const;
  std::unique_ptr<cudf::string_column> allocate(
    cudaStream_t stream,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());
};

// template <typename T>
// struct string_column_output {
//   column_output<char> chars;
//   column_output<uint32_t> offsets;
//   bitmask_output mask;

//   template <bool output_enabled>
//   inline constexpr void append(char value)
//   {
//     chars.emit<output_enabled>(value);
//   }

//   template <bool output_enabled>
//   inline constexpr uint32_t terminate(bool is_null = false)
//   {
//     offsets.emit<output_enabled>(chars.output_count);
//     mask.emit<output_enabled>(is_null);
//   }

//   inline constexpr string_column_output operator+(string_column_output other) const
//   {
//     return {
//       char_output + other.char_output,
//       offsets + other.offsets,
//       mask + other.mask,
//     };
//   }

//   std::unique_ptr<cudf::string_column> allocate(
//     cudaStream_t stream,
//     rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
//   {
//     auto offsets_column = offsets.allocate();
//     auto chars_column   = chars.allocate();
//     auto mask_buffer    = mask.allocate();

//     return make_strings_column(offsets_column.size() - 1,
//                                std::move(offsets_column),
//                                std::move(chars_column),
//                                mask.null_count(),
//                                std::move(mask_buffer) stream,
//                                mr);
//   }
// };

}  // namespace detail
}  // namespace output
}  // namespace dfa
}  // namespace cudf
