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

"""Namespace for operators used in Gluon dispatched by F=ndarray."""

from __future__ import absolute_import
from . import _op as _mx_nd_np

__all__ = ['norm']


def norm(x, ord=None, axis=None, keepdims=False):
    r"""Matrix or vector norm.

    This function can only support Frobenius norm for now.
    The Frobenius norm is given by [1]_:

        :math:`||A||_F = [\sum_{i,j} abs(a_{i,j})^2]^{1/2}`

    Parameters
    ----------
    x : ndarray
        Input array.
    ord : {'fro'}, optional
        Order of the norm.
    axis : {int, 2-tuple of ints, None}, optional
        If `axis` is an integer, it specifies the axis of `x` along which to
        compute the vector norms.  If `axis` is a 2-tuple, it specifies the
        axes that hold 2-D matrices, and the matrix norms of these matrices
        are computed.  If `axis` is None, the norm of the whole ndarray is
        returned.

    keepdims : bool, optional
        If this is set to True, the axes which are normed over are left in the
        result as dimensions with size one.  With this option the result will
        broadcast correctly against the original `x`.

    Returns
    -------
    n : float or ndarray
        Norm of the matrix or vector(s).

    References
    ----------
    .. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*,
           Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15
    """
    if axis is None or isinstance(axis, (int, tuple)):
        if axis != None:
            axis = (axis,)
            if len(axis) == 2:
                if ord == 'inf' or ord == '-inf':
                    row_axis, col_axis = axis
                    if not keepdims:
                        if row_axis > col_axis:
                            row_axis -= 1
                        if ord == 'inf':
                            return _mx_nd_np.sum(_mx_nd_np.abs(x),axis=col_axis, keepdims=keepdims).max(axis=row_axis,keepdims=keepdims)
                        else:
                            return _mx_nd_np.sum(_mx_nd_np.abs(x),axis=col_axis, keepdims=keepdims).min(axis=row_axis,keepdims=keepdims)
                if ord == 1 or ord == -1:
                    row_axis, col_axis = axis
                    if not keepdims:
                        if row_axis < col_axis:
                            col_axis -= 1
                        if ord == 1:
                            return _mx_nd_np.sum(_mx_nd_np.abs(x),axis=row_axis, keepdims=keepdims).max(axis=col_axis,keepdims=keepdims)
                        else:
                            return _mx_nd_np.sum(_mx_nd_np.abs(x),axis=row_axis, keepdims=keepdims).min(axis=col_axis,keepdims=keepdims)

        if ord == 'inf':
            return _mx_nd_np.norm(x, 2, axis, keepdims, 3)
        elif ord == '-inf':
            return _mx_nd_np.norm(x, 2, axis, keepdims, 4)
        elif ord == None:
            return _mx_nd_np.norm(x, 2, axis, keepdims, 0)
        elif ord == 2:
            return _mx_nd_np.norm(x, 2, axis, keepdims)
        elif ord == 'nuc':
            return _mx_nd_np.norm(x, 2, axis, keepdims, 2)
        elif ord == 'fro' or ord == 'f':
            return _mx_nd_np.norm(x, 2, axis, keepdims, 1)
        else:
            return _mx_nd_np.norm(x, ord, axis, keepdims)
    else:
        raise TypeError("'axis' must be None, an integer or a tuple of integers")