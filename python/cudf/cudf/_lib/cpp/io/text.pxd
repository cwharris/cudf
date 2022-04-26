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
        auto out = std::make_unique<string_data_chunk_source>(c_str);
        return out;
    }

    /*

    // create a custom data_chunk_source that holds a reference to a PyObject and
    // knows it either is a or can become a TextIOBase. this customer reader will return
    // a custom data chunk reader that will read from a TextIOBase a chunk at a time
    // when the data chunk reader is disposed of, so shall the TextIOBase be released.
    // when the data chunk source is disposed of, so shall the original PyObject be released.
    // the gil shall only be helf when necessary, probably only when reading from the TextIOBase.

    class python_text_data_chunk_reader : public data_chunk_reader {
        // TODO: implement
    }

    class python_text_data_chunk_source : public data_chunk_source {
        public:
            python_text_data_Chunk_source(PyObject* source) : _source(source) {}
            [[nodiscard]] std::unique_ptr<data_chunk_reader> create_reader() const override
            {
                return std::make_unique<python_text_data_chunk_reader(_source);
            }

        private:
            PyObject* _source; // TODO: pick what type this should be
    };

    */

    }
    }
    }
    """
    unique_ptr[data_chunk_source] my_make_source(object)
