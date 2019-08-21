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

# pylint: skip-file
from __future__ import absolute_import
import numpy as _np
import mxnet as mx
from mxnet import np, npx
from mxnet.base import MXNetError
from mxnet.gluon import HybridBlock
from mxnet.base import MXNetError
from mxnet.test_utils import same, assert_almost_equal, rand_shape_nd, rand_ndarray
from mxnet.test_utils import check_numeric_gradient, use_np
from mxnet.test_utils import verify_generator, gen_buckets_probs_with_ppf, retry
import scipy.stats as ss
from common import assertRaises, with_seed
import random
import collections
from mxnet.runtime import Features
import os
import subprocess
import time

_features = Features()

class Benchmark():
    def __init__(self, prefix=None):
        self.prefix = prefix + ' ' if prefix else ''

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        print('%stime: %.4f sec' % (self.prefix, time.time() - self.start))


@with_seed()
@use_np
def test_np_exp2():
    class Testexp2(HybridBlock):
        def __init__(self):
            super(Testexp2, self).__init__()

        def hybrid_forward(self, F, x, *args, **kwargs):
            return F.np.exp2(x)

    shapes = [
        (),
        (2,),
        (2, 1, 2),
        (2, 0, 2),
        (1, 2, 3, 4, 5),
        (6, 6, 6, 6, 6),
    ]

    # Test imperative
    for hybridize in [True, False]:
        for shape in shapes:
            test_exp2 = Testexp2()
            if hybridize:
                test_exp2.hybridize()
            x = rand_ndarray(shape).as_np_ndarray()
            np_out = _np.exp2(x.asnumpy())
            mx_out = test_exp2(x)
            assert mx_out.shape == np_out.shape
            assert_almost_equal(mx_out.asnumpy(), np_out, rtol = 1e-3, atol = 1e-5)


@with_seed()
@use_np
def test_np_tvm_exp2():
    class Testtvm_exp2(HybridBlock):
        def __init__(self):
            super(Testtvm_exp2, self).__init__()

        def hybrid_forward(self, F, x, *args, **kwargs):
            return F.np.tvm_exp2(x)

    shapes = [
        (),
        (2,),
        (2, 1, 2),
        (2, 0, 2),
        (1, 2, 3, 4, 5),
        (6, 6, 6, 6, 6),
    ]

    # Test imperative
    for hybridize in [True, False]:
        for shape in shapes:
            test_tvm_exp2 = Testtvm_exp2()
            if hybridize:
                test_tvm_exp2.hybridize()
            x = rand_ndarray(shape).as_np_ndarray()
            np_out = _np.exp2(x.asnumpy())
            mx_out = test_tvm_exp2(x)
            assert mx_out.shape == np_out.shape
            assert_almost_equal(mx_out.asnumpy(), np_out, rtol = 1e-3, atol = 1e-5)

