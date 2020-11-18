/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

struct bitmask_output {
  template <bool output_enabled>
  inline constexpr void append(bool value);
  inline constexpr bitmask_output operator+(bitmask_output other) const;
  rmm::device_buffer allocate(
    cudaStream_t stream,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());
};

}  // namespace detail
}  // namespace output
}  // namespace dfa
}  // namespace cudf
