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

template <typename T>
struct vector_output {
  T* data;
  uint32_t count;

  template <bool output_enabled>
  inline constexpr void emit(T value)
  {
    if (output_enabled) {
      data[count++] = value;
    } else {
      count++;
    }
  }

  inline constexpr vector_output operator+(vector_output other) const
  {
    return {
      data,
      count + other.count,
    };
  }

  rmm::device_uvector<T> allocate(
    cudaStream_t stream,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
  {
    auto buffer = rmm::device_uvector<T>(count, stream, mr);
    data        = buffer.data();
    count       = 0;
    return buffer;
  }
};

}  // namespace detail
}  // namespace output
}  // namespace dfa
}  // namespace cudf
