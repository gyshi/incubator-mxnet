# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
import tvm
import topi
from .. import defop, AllTypes, AllTypesButHalf
from .. import assign_by_req, reduce_axes


def compute_exp2(dtype, ndim):
    A = tvm.placeholder([tvm.var() for _ in range(ndim)], name='A', dtype=dtype)
    if dtype in ['float32', 'float64']:
        B = tvm.compute([tvm.var() for _ in range(ndim)],
                        lambda *index: topi.power(2, A[index]), name='B')
    else:
        B = tvm.compute([tvm.var() for _ in range(ndim)],
                        lambda *index: topi.power(2, A[index].astype('float32')).astype(dtype),
                        name='B')
    s = tvm.create_schedule(B.op)
    return s, A, B


@defop(name="exp2", target="cpu", auto_broadcast=True,
       dtype=AllTypes, ndim=list(range(0, 6)))
def exp2(dtype, ndim):
    s, A, B = compute_exp2(dtype, ndim)
    axes = [axis for axis in B.op.axis]
    fused = s[B].fuse(*axes)
    s[B].parallel(fused)
    return s, [A, B]


@defop(name="cuda_exp2", target="cuda", auto_broadcast=True,
       dtype=AllTypesButHalf, ndim=list(range(0, 6)))
def exp2_gpu(dtype, ndim):
    s, A, B= compute_exp2(dtype, ndim)
    s = tvm.create_schedule(B.op)
    axes = [axis for axis in B.op.axis]
    fused = s[B].fuse(*axes)
    bx, tx = s[B].split(fused, factor=64)
    s[B].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[B].bind(tx, tvm.thread_axis("threadIdx.x"))
    return s, [A, B]


def compute_backward_exp2(dtype, ndim):
    A = tvm.placeholder([tvm.var() for _ in range(ndim)], name='A', dtype=dtype)
    B = tvm.placeholder([tvm.var() for _ in range(ndim)], name='B', dtype=dtype)
    C = tvm.placeholder([tvm.var() for _ in range(ndim)], name='C', dtype=dtype)
    D = tvm.compute([tvm.var() for _ in range(ndim)],
                    lambda *index:  A[index] * B[index] * C[index] / 2, name='D')
    s = tvm.create_schedule(D.op)
    return s, A, B, C, D


@defop(name="backward_exp2", target="cpu", auto_broadcast=True,
       dtype=AllTypes, ndim=list(range(0, 6)))
def backward_exp2(dtype, ndim):
    s, A, B, C, D = compute_backward_exp2(dtype, ndim)
    axes = [axis for axis in D.op.axis]
    fused = s[D].fuse(*axes)
    s[D].parallel(fused)
    return s, [A, B, C, D]


@defop(name="cuda_backward_exp2", target="cuda", auto_broadcast=True,
       dtype=AllTypesButHalf, ndim=list(range(0, 6)))
def backward_exp2_gpu(dtype, ndim):
    s, A, B, C, D= compute_backward_exp2(dtype, ndim)
    s = tvm.create_schedule(D.op)
    axes = [axis for axis in D.op.axis]
    fused = s[D].fuse(*axes)
    bx, tx = s[D].split(fused, factor=64)
    s[D].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[D].bind(tx, tvm.thread_axis("threadIdx.x"))
    return s, [A, B, C, D]
