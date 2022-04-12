# Copyright (c) 2020-2022, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string

from cudf._lib.cpp.column.column cimport column


cdef extern from "cudf/io/text/byte_range_info.hpp" \
        namespace "cudf::io::text" nogil:

    cdef cppclass byte_range_info:
        byte_range_info() except +
        byte_range_info(size_t offset, size_t size) except +

cdef extern from "cudf/io/text/data_chunk_source.hpp" \
        namespace "cudf::io::text" nogil:

    cdef cppclass data_chunk_source:
        data_chunk_source() except +

cdef extern from "cudf/io/text/data_chunk_source_factories.hpp" \
        namespace "cudf::io::text" nogil:

    unique_ptr[data_chunk_source] \
        make_source_from_file(string filename) except +


cdef extern from "cudf/io/text/multibyte_split.hpp" \
        namespace "cudf::io::text" nogil:

    unique_ptr[column] multibyte_split(data_chunk_source source,
                                       string delimiter) except +

    unique_ptr[column] multibyte_split(data_chunk_source source,
                                       string delimiter,
                                       byte_range_info byte_range) except +

cdef extern from * namespace "cudf::io::text":
    """
    #include <cudf/io/text/data_chunk_source_factories.hpp>
    #include <rmm/device_buffer.hpp>
    #include <iostream>
    #include <Python.h>

    namespace cudf {
    namespace io {
    namespace text {
      std::unique_ptr<data_chunk_source> my_make_source(PyObject* data) {
         std::string c_str = PyBytes_AsString(data);
         std::cout << c_str << std::endl;
         std::cout << c_str.size() << std::endl;
         auto out = std::make_unique<string_data_chunk_source>(
            std::move(c_str)
         );
         return out;
      }
    }
    }
    }
    """
    unique_ptr[data_chunk_source] my_make_source(object)
