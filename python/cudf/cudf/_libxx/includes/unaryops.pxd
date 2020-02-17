# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._libxx.lib cimport *

cdef extern from "cudf/unary.hpp" namespace "cudf::experimental" nogil:

    ctypedef enum unary_op:
        SIN 'cudf::experimental::unary_op::SIN'
        COS 'cudf::experimental::unary_op::COS'
        TAN 'cudf::experimental::unary_op::TAN'
        ARCSIN 'cudf::experimental::unary_op::ARCSIN'
        ARCCOS 'cudf::experimental::unary_op::ARCCOS'
        ARCTAN 'cudf::experimental::unary_op::ARCTAN'
        SINH 'cudf::experimental::unary_op::SINH'
        COSH 'cudf::experimental::unary_op::COSH'
        TANH 'cudf::experimental::unary_op::TANH'
        ARCSINH 'cudf::experimental::unary_op::ARCSINH'
        ARCCOSH 'cudf::experimental::unary_op::ARCCOSH'
        ARCTANH 'cudf::experimental::unary_op::ARCTANH'
        EXP 'cudf::experimental::unary_op::EXP'
        LOG 'cudf::experimental::unary_op::LOG'
        SQRT 'cudf::experimental::unary_op::SQRT'
        CBRT 'cudf::experimental::unary_op::CBRT'
        CEIL 'cudf::experimental::unary_op::CEIL'
        FLOOR 'cudf::experimental::unary_op::FLOOR'
        ABS 'cudf::experimental::unary_op::ABS'
        RINT 'cudf::experimental::unary_op::RINT'
        BIT_INVERT 'cudf::experimental::unary_op::BIT_INVERT'
        NOT 'cudf::experimental::unary_op::NOT'

    cdef unique_ptr[column] cast(
        column_view input,
        data_type out_type
    )
