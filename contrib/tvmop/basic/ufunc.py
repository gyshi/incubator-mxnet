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
from .. import defop, AllTypes

def compute_add(dtype, ndim):
    A = tvm.placeholder([tvm.var() for _ in range(ndim)], name='A', dtype=dtype)
    B = tvm.placeholder([tvm.var() for _ in range(ndim)], name='B', dtype=dtype)
    C = tvm.compute([tvm.var() for _ in range(ndim)],
                    lambda *index: A[index] + B[index], name='C')
    s = tvm.create_schedule(C.op)
    return s, A, B, C

@defop(name="vadd", target="cpu", auto_broadcast=True,
       dtype=AllTypes, ndim=list(range(1, 6)))
def vadd(dtype, ndim):
    s, A, B, C = compute_add(dtype, ndim)
    axes = [axis for axis in C.op.axis]
    fused = s[C].fuse(*axes)
    s[C].parallel(fused)

    return s, [A, B, C]

@defop(name="cuda_vadd", target="cuda", auto_broadcast=True,
       dtype=["float32", "float64"], ndim=list(range(1, 6)))
def vadd_gpu(dtype, ndim):
    s, A, B, C = compute_add(dtype, ndim)
    s = tvm.create_schedule(C.op)
    axes = [axis for axis in C.op.axis]
    fused = s[C].fuse(*axes)
    bx, tx = s[C].split(fused, factor=64)
    s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[C].bind(tx, tvm.thread_axis("threadIdx.x"))
    return s, [A, B, C]

def compute_exp2(dtype, ndim):
    A = tvm.placeholder([tvm.var() for _ in range(ndim)], name='A', dtype=dtype)
    B = tvm.compute([tvm.var() for _ in range(ndim)],
                    lambda *index: topi.power(2, A[index]), name='B')
    s = tvm.create_schedule(B.op)
    return s, A, B

@defop(name="exp2", target="cpu", auto_broadcast=True,
       dtype=["float32", "float64"], ndim=list(range(0, 6)))
def exp2(dtype, ndim):
    s, A, B = compute_exp2(dtype, ndim)
    # axes = [axis for axis in B.op.axis]
    # fused = s[B].fuse(*axes)
    # opt0
    # s[B].parallel(fused)
    # opt 1
    # fused = s[B].fuse(*axes)
    #
    #
    # s[B].reorder(fused)
    # s[B].parallel(fused)
    # opt 2
    # fused = s[B].fuse(*axes)
    # xo, xi = s[B].split(fused, factor=32)
    # s[B].reorder(xo, xi)
    #
    # opt 3
    # fused = s[B].fuse(*axes)
    # xo, xi = s[B].split(fused, factor=32)
    # s[B].reorder(xo, xi)
    # s[B].vectorize(xi)
    #
    # opt 4
    # fused = s[B].fuse(*axes)
    # xo, xi = s[B].split(fused, factor=32)
    # s[B].reorder(xo, xi)
    # s[B].vectorize(xi)
    # s[B].parallel(xo)

    return s, [A, B]

@defop(name="cuda_exp2", target="cuda", auto_broadcast=True,
       dtype=["float32", "float64"], ndim=list(range(0, 6)))
def exp2_gpu(dtype, ndim):
    s, A, B= compute_exp2(dtype, ndim)
    s = tvm.create_schedule(B.op)
    #opt0
    # axes = [axis for axis in B.op.axis]
    # fused = s[B].fuse(*axes)
    # bx, tx = s[B].split(fused, factor=64)
    # s[B].bind(bx, tvm.thread_axis("blockIdx.x"))
    # s[B].bind(tx, tvm.thread_axis("threadIdx.x"))
    #
    #opt1
    axes = [axis for axis in B.op.axis]
    fused = s[B].fuse(*axes)
    bx, tx = s[B].split(fused, factor=64)
    s[B].reorder(bx, tx)

    s[B].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[B].bind(tx, tvm.thread_axis("threadIdx.x"))
    s[B].vectorize(tx)
    return s, [A, B]

def compute_backward_exp2(dtype, ndim):
    A = tvm.placeholder([tvm.var() for _ in range(ndim)], name='A', dtype=dtype)
    B = tvm.placeholder([tvm.var() for _ in range(ndim)], name='B', dtype=dtype)
    C = tvm.compute([tvm.var() for _ in range(ndim)],
                    lambda *index:  A[index] * B[index] *tvm.const(0.6931471805599453, dtype), name='D')
    s = tvm.create_schedule(C.op)
    return s, A, B, C

@defop(name="backward_exp2", target="cpu", auto_broadcast=True,
       dtype=["float32", "float64"], ndim=list(range(0, 6)))
def backward_exp2(dtype, ndim):
    s, A, B, C = compute_backward_exp2(dtype, ndim)
    # axes = [axis for axis in C.op.axis]
    # fused = s[D].fuse(*axes)
    # s[D].parallel(fused)

    return s, [A, B, C]

@defop(name="cuda_backward_exp2", target="cuda", auto_broadcast=True,
       dtype=["float32", "float64"], ndim=list(range(0, 6)))
def backward_exp2_gpu(dtype, ndim):
    s, A, B, C= compute_backward_exp2(dtype, ndim)
    s = tvm.create_schedule(C.op)
    axes = [axis for axis in C.op.axis]
    fused = s[C].fuse(*axes)
    bx, tx = s[C].split(fused, factor=64)
    s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[C].bind(tx, tvm.thread_axis("threadIdx.x"))
    return s, [A, B, C]


@defop(name="relu", target="cpu", auto_broadcast=True,
       dtype=["float32"], ndim=list(range(0, 6)))
def relu(dtype, ndim):
    A = tvm.placeholder([tvm.var() for _ in range(ndim)], name='A', dtype=dtype)
    B = tvm.compute([tvm.var() for _ in range(ndim)],
                    lambda *index: tvm.if_then_else(A[index] > 0.0, A[index], 0.0), name='B')
    s = tvm.create_schedule(B.op)
    # axes = [axis for axis in B.op.axis]
    # fused = s[B].fuse(*axes)
    # s[B].parallel(fused)
    return s, [A, B]