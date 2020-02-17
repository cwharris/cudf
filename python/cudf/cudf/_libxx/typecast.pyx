# Copyright (c) 2020, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import numpy as np


from cudf._libxx.lib import *
from cudf._libxx.lib cimport *
from cudf._libxx.column cimport Column

cimport cudf._libxx.includes.unaryops as cpp_unaryops


def cast(Column input, dtype=np.float64):

    cdef column_view c_input = input.view()
    cdef type_id tid = np_to_cudf_types[np.dtype(dtype)]
    cdef data_type c_out_type = data_type(tid)

    return Column.from_unique_ptr(
        cpp_unaryops.cast(
            c_input,
            c_out_type
        )
    )
