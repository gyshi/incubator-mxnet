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

# pylint: disable=too-many-lines
"""numpy namespace for operators used in Gluon APIs dispatched by F=symbol module."""

from __future__ import absolute_import
import ctypes
import numpy as _np
from . import _op as _mx_np_op
from ...base import _LIB, SymbolHandle, numeric_types, mx_uint
from ...util import check_call, set_module
from ...context import current_context
from ..symbol import Symbol
from .._internal import _set_np_symbol_class
from . import _internal as _npi

__all__ = ['zeros', 'ones', 'add', 'subtract', 'multiply', 'divide', 'mod', 'power', 'tensordot',
           'linspace', 'expand_dims', 'tile', 'arange', 'split', 'concatenate', 'stack', 'rot90']


def _num_outputs(sym):
    return len(sym.as_nd_ndarray())


@set_module('mxnet.symbol.numpy')
class _Symbol(Symbol):
    def __getitem__(self, key):
        num_outputs = _num_outputs(self)
        if num_outputs == 1:
            raise NotImplementedError
        if not isinstance(key, int):
            raise NotImplementedError
        if key >= num_outputs:
            # Important, python determines the end by this exception
            raise IndexError
        handle = SymbolHandle()
        check_call(_LIB.MXSymbolGetOutput(
            self.handle, mx_uint(key), ctypes.byref(handle)))
        return _Symbol(handle=handle)

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __iter__(self):
        raise AttributeError('_Symbol object has no attribute __iter__')

    def __add__(self, other):
        """x.__add__(y) <=> x + y"""
        return add(self, other)

    def __sub__(self, other):
        """x.__sub__(y) <=> x - y"""
        return subtract(self, other)

    def __rsub__(self, other):
        """x.__rsub__(y) <=> y - x"""
        return subtract(other, self)

    def __mul__(self, other):
        """x.__mul__(y) <=> x * y"""
        return multiply(self, other)

    def __rmul__(self, other):
        """x.__rmul__(y) <=> y * x"""
        return multiply(other, self)

    def __div__(self, other):
        raise AttributeError('_Symbol.__div__ is replaced by __truediv__. If you are using'
                             ' Python2, please use the statement from __future__ import division'
                             ' to change the / operator to mean true division throughout the'
                             ' module. If you are using Python3, this error should not have'
                             ' been encountered.')

    def __rdiv__(self, other):
        raise AttributeError('_Symbol.__rdiv__ is replaced by __rtruediv__. If you are using'
                             ' Python2, please use the statement from __future__ import division'
                             ' to change the / operator to mean true division throughout the'
                             ' module. If you are using Python3, this error should not have'
                             ' been encountered.')

    def __mod__(self, other):
        """x.__mod__(y) <=> x % y"""
        return mod(self, other)

    def __rmod__(self, other):
        """x.__rmod__(y) <=> y % x"""
        return mod(other, self)

    def __idiv__(self, other):
        raise NotImplementedError

    def __truediv__(self, other):
        """x.__truediv__(y) <=> x / y"""
        return divide(self, other)

    def __rtruediv__(self, other):
        """x.__rtruediv__(y) <=> y / x"""
        return divide(other, self)

    def __itruediv__(self, other):
        raise NotImplementedError

    def __pow__(self, other):
        """x.__pow__(y) <=> x ** y"""
        return power(self, other)

    def __rpow__(self, other):
        return power(other, self)

    def __neg__(self):
        """x.__neg__() <=> - x"""
        return self.__mul__(-1.0)

    def __deepcopy__(self, _):
        return super(_Symbol, self).as_np_ndarray()

    def __eq__(self, other):
        """x.__eq__(y) <=> x == y"""
        # TODO(junwu): Return boolean ndarray when dtype=bool_ is supported
        if isinstance(other, _Symbol):
            return _npi.equal(self, other)
        elif isinstance(other, numeric_types):
            return _npi.equal_scalar(self, float(other))
        else:
            raise TypeError("_Symbol does not support type {} as operand".format(str(type(other))))

    def __ne__(self, other):
        """x.__ne__(y) <=> x != y"""
        # TODO(junwu): Return boolean ndarray when dtype=bool_ is supported
        if isinstance(other, _Symbol):
            return _npi.not_equal(self, other)
        elif isinstance(other, numeric_types):
            return _npi.not_equal_scalar(self, float(other))
        else:
            raise TypeError("_Symbol does not support type {} as operand".format(str(type(other))))

    def __gt__(self, other):
        """x.__gt__(y) <=> x > y"""
        # TODO(junwu): Return boolean ndarray when dtype=bool_ is supported
        if isinstance(other, _Symbol):
            return _npi.greater(self, other)
        elif isinstance(other, numeric_types):
            return _npi.greater_scalar(self, float(other))
        else:
            raise TypeError("_Symbol does not support type {} as operand".format(str(type(other))))

    def __ge__(self, other):
        """x.__ge__(y) <=> x >= y"""
        # TODO(junwu): Return boolean ndarray when dtype=bool_ is supported
        if isinstance(other, _Symbol):
            return _npi.greater_equal(self, other)
        elif isinstance(other, numeric_types):
            return _npi.greater_equal_scalar(self, float(other))
        else:
            raise TypeError("_Symbol does not support type {} as operand".format(str(type(other))))

    def __lt__(self, other):
        """x.__lt__(y) <=> x < y"""
        # TODO(junwu): Return boolean ndarray when dtype=bool_ is supported
        if isinstance(other, _Symbol):
            return _npi.less(self, other)
        elif isinstance(other, numeric_types):
            return _npi.less_scalar(self, float(other))
        else:
            raise TypeError("_Symbol does not support type {} as operand".format(str(type(other))))

    def __le__(self, other):
        """x.__le__(y) <=> x <= y"""
        # TODO(junwu): Return boolean ndarray when dtype=bool_ is supported
        if isinstance(other, _Symbol):
            return _npi.less_equal(self, other)
        elif isinstance(other, numeric_types):
            return _npi.less_equal_scalar(self, float(other))
        else:
            raise TypeError("_Symbol does not support type {} as operand".format(str(type(other))))

    def __len__(self):
        raise NotImplementedError

    def as_nd_ndarray(self):
        """Convert _Symbol to mxnet.symbol.Symbol to use its convenience fluent methods."""
        hdl = SymbolHandle()
        check_call(_LIB.MXShallowCopySymbol(self.handle, ctypes.byref(hdl)))
        return Symbol(handle=hdl)

    def as_np_ndarray(self):
        """For the convenience of conversion between legacy and np symbols."""
        return self

    @property
    # pylint: disable= invalid-name, undefined-variable
    def T(self):
        """Same as self.transpose()."""
        return self.transpose()
    # pylint: enable= invalid-name, undefined-variable

    def astype(self, dtype, **kwargs):  # pylint: disable=arguments-differ
        raise NotImplementedError

    def dot(self, b, out=None):
        raise NotImplementedError

    def reshape(self, *args, **kwargs):  # pylint: disable=arguments-differ
        """Returns an array containing the same data with a new shape.

        Notes
        -----
        Unlike the free function `numpy.reshape`, this method on `ndarray` allows
        the elements of the shape parameter to be passed in as separate arguments.
        For example, ``a.reshape(10, 11)`` is equivalent to
        ``a.reshape((10, 11))``.
        """
        order = 'C'
        if len(kwargs) > 1:
            raise TypeError('function takes at most 1 keyword argument')
        if len(kwargs) == 1:
            if 'order' not in kwargs:
                raise TypeError('{} is an invalid keyword argument for this function'
                                .format(kwargs.keys()[0]))
            order = kwargs.pop('order', 'C')
            if order != 'C':
                raise NotImplementedError('only supports C-order,'
                                          ' while received {}'.format(order))
        if len(args) == 0:
            raise TypeError('reshape() takes exactly 1 argument (0 given)')
        if len(args) == 1 and isinstance(args[0], tuple):
            return _mx_np_op.reshape(self, newshape=args[0], order=order)
        else:
            return _mx_np_op.reshape(self, newshape=args, order=order)

    def argmax(self, axis=None, out=None):  # pylint: disable=arguments-differ
        raise NotImplementedError

    def reshape_like(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`reshape_like`.

        The arguments are the same as for :py:func:`reshape_like`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute reshape_like')

    def zeros_like(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`zeros_like`.

        The arguments are the same as for :py:func:`zeros_like`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute zeros_like')

    def ones_like(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`ones_like`.

        The arguments are the same as for :py:func:`ones_like`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute ones_like')

    def broadcast_axes(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`broadcast_axes`.

        The arguments are the same as for :py:func:`broadcast_axes`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute broadcast_like')

    def repeat(self, repeats, axis=None):  # pylint: disable=arguments-differ
        """Repeat elements of an array."""
        raise NotImplementedError

    def pad(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`pad`.

        The arguments are the same as for :py:func:`pad`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute pad')

    def swapaxes(self, axis1, axis2):  # pylint: disable=arguments-differ
        """Return a copy of the array with axis1 and axis2 interchanged.
        Refer to `mxnet.numpy.swapaxes` for full documentation.
        """
        raise NotImplementedError

    def split(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`split`.

        The arguments are the same as for :py:func:`split`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute split')

    def split_v2(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`split_v2`.

        The arguments are the same as for :py:func:`split_v2`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute split_v2')

    def slice(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`slice`.

        The arguments are the same as for :py:func:`slice`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute slice')

    def slice_axis(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`slice_axis`.

        The arguments are the same as for :py:func:`slice_axis`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute slice_axis')

    def slice_like(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`slice_like`.

        The arguments are the same as for :py:func:`slice_like`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute slice_like')

    def take(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`take`.

        The arguments are the same as for :py:func:`take`, with
        this array as data.
        """
        raise NotImplementedError

    def one_hot(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`one_hot`.

        The arguments are the same as for :py:func:`one_hot`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute one_hot')

    def pick(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`pick`.

        The arguments are the same as for :py:func:`pick`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute pick')

    def sort(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`sort`.

        The arguments are the same as for :py:func:`sort`, with
        this array as data.
        """
        raise NotImplementedError

    def topk(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`topk`.

        The arguments are the same as for :py:func:`topk`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute topk')

    def argsort(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`argsort`.

        The arguments are the same as for :py:func:`argsort`, with
        this array as data.
        """
        raise NotImplementedError

    def argmax_channel(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`argmax_channel`.

        The arguments are the same as for :py:func:`argmax_channel`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute argmax_channel')

    def argmin(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`argmin`.

        The arguments are the same as for :py:func:`argmin`, with
        this array as data.
        """
        raise NotImplementedError

    def clip(self, min=None, max=None, out=None):  # pylint: disable=arguments-differ
        """Return an array whose values are limited to [min, max].
        One of max or min must be given.
        """
        raise NotImplementedError

    def abs(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`abs`.

        The arguments are the same as for :py:func:`abs`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute abs')

    def sign(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`sign`.

        The arguments are the same as for :py:func:`sign`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute abs')

    def flatten(self, order='C'):  # pylint: disable=arguments-differ
        """Return a copy of the array collapsed into one dimension."""
        return self.reshape(-1, order=order)

    def shape_array(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`shape_array`.

        The arguments are the same as for :py:func:`shape_array`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute shape_array')

    def size_array(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`size_array`.

        The arguments are the same as for :py:func:`size_array`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute size_array')

    def expand_dims(self, *args, **kwargs):  # pylint: disable=arguments-differ,unused-argument
        """Convenience fluent method for :py:func:`expand_dims`.

        The arguments are the same as for :py:func:`expand_dims`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute expand_dims')

    def tile(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`tile`.

        The arguments are the same as for :py:func:`tile`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute tile')

    def transpose(self, *axes):  # pylint: disable=arguments-differ
        """The arguments are the same as for :py:func:`transpose`, with
        this array as data.
        """
        return _mx_np_op.transpose(self, axes=axes if len(axes) != 0 else None)

    def flip(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`flip`.

        The arguments are the same as for :py:func:`flip`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute flip')

    def depth_to_space(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`depth_to_space`.

        The arguments are the same as for :py:func:`depth_to_space`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute depth_to_space')

    def space_to_depth(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`space_to_depth`.

        The arguments are the same as for :py:func:`space_to_depth`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute space_to_depth')

    def diag(self, k=0, **kwargs):
        """Convenience fluent method for :py:func:`diag`.

        The arguments are the same as for :py:func:`diag`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute diag')

    def sum(self, axis=None, dtype=None, out=None, keepdims=False):  # pylint: disable=arguments-differ
        """Return the sum of the array elements over the given axis."""
        return _mx_np_op.sum(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    def nansum(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`nansum`.

        The arguments are the same as for :py:func:`nansum`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute nansum')

    def prod(self, axis=None, dtype=None, out=None, keepdims=False):  # pylint: disable=arguments-differ
        """Return the product of the array elements over the given axis."""
        return _mx_np_op.prod(self, axis=axis, dtype=dtype, keepdims=keepdims, out=out)

    def nanprod(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`nanprod`.

        The arguments are the same as for :py:func:`nanprod`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute nanprod')

    def mean(self, axis=None, dtype=None, out=None, keepdims=False):  # pylint: disable=arguments-differ
        """Convenience fluent method for :py:func:`mean`.

        The arguments are the same as for :py:func:`mean`, with
        this array as data.
        """
        raise NotImplementedError

    def cumsum(self, axis=None, dtype=None, out=None):
        """Return the cumulative sum of the elements along the given axis."""
        raise NotImplementedError

    def max(self, axis=None, out=None, keepdims=False):  # pylint: disable=arguments-differ
        """Return the maximum along a given axis."""
        raise NotImplementedError

    def min(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`min`.

        The arguments are the same as for :py:func:`min`, with
        this array as data.
        """
        raise NotImplementedError

    def norm(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`norm`.

        The arguments are the same as for :py:func:`norm`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute norm')

    def round(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`round`.

        The arguments are the same as for :py:func:`round`, with
        this array as data.
        """
        raise NotImplementedError

    def rint(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`rint`.

        The arguments are the same as for :py:func:`rint`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute rint')

    def fix(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`fix`.

        The arguments are the same as for :py:func:`fix`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute fix')

    def floor(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`floor`.

        The arguments are the same as for :py:func:`floor`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute floor')

    def ceil(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`ceil`.

        The arguments are the same as for :py:func:`ceil`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute ceil')

    def trunc(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`trunc`.

        The arguments are the same as for :py:func:`trunc`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute trunc')

    def sin(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`sin`.

        The arguments are the same as for :py:func:`sin`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute sin')

    def cos(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`cos`.

        The arguments are the same as for :py:func:`cos`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute cos')

    def tan(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`tan`.

        The arguments are the same as for :py:func:`tan`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute tan')

    def arcsin(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`arcsin`.

        The arguments are the same as for :py:func:`arcsin`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute arcsin')

    def arccos(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`arccos`.

        The arguments are the same as for :py:func:`arccos`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute arccos')

    def arctan(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`arctan`.

        The arguments are the same as for :py:func:`arctan`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute arctan')

    def degrees(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`degrees`.

        The arguments are the same as for :py:func:`degrees`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute degrees')

    def radians(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`radians`.

        The arguments are the same as for :py:func:`radians`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute radians')

    def sinh(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`sinh`.

        The arguments are the same as for :py:func:`sinh`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute sinh')

    def cosh(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`cosh`.

        The arguments are the same as for :py:func:`cosh`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute cosh')

    def tanh(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`tanh`.

        The arguments are the same as for :py:func:`tanh`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute tanh')

    def arcsinh(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`arcsinh`.

        The arguments are the same as for :py:func:`arcsinh`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute arcsinh')

    def arccosh(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`arccosh`.

        The arguments are the same as for :py:func:`arccosh`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute arccosh')

    def arctanh(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`arctanh`.

        The arguments are the same as for :py:func:`arctanh`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute arctanh')

    def exp(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`exp`.

        The arguments are the same as for :py:func:`exp`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute exp')

    def expm1(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`expm1`.

        The arguments are the same as for :py:func:`expm1`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute expm1')

    def log(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`log`.

        The arguments are the same as for :py:func:`log`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute log')

    def log10(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`log10`.

        The arguments are the same as for :py:func:`log10`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute log10')

    def log2(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`log2`.

        The arguments are the same as for :py:func:`log2`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute log2')

    def log1p(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`log1p`.

        The arguments are the same as for :py:func:`log1p`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute log1p')

    def sqrt(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`sqrt`.

        The arguments are the same as for :py:func:`sqrt`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute sqrt')

    def rsqrt(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`rsqrt`.

        The arguments are the same as for :py:func:`rsqrt`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute rsqrt')

    def cbrt(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`cbrt`.

        The arguments are the same as for :py:func:`cbrt`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute cqrt')

    def rcbrt(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`rcbrt`.

        The arguments are the same as for :py:func:`rcbrt`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute rcqrt')

    def square(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`square`.

        The arguments are the same as for :py:func:`square`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute square')

    def reciprocal(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`reciprocal`.

        The arguments are the same as for :py:func:`reciprocal`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute reciprocal')

    def relu(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`relu`.

        The arguments are the same as for :py:func:`relu`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute relu')

    def sigmoid(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`sigmoid`.

        The arguments are the same as for :py:func:`sigmoid`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute sigmoid')

    def softmax(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`softmax`.

        The arguments are the same as for :py:func:`softmax`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute softmax')

    def log_softmax(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`log_softmax`.

        The arguments are the same as for :py:func:`log_softmax`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute log_softmax')

    def softmin(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`softmin`.

        The arguments are the same as for :py:func:`softmin`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute softmin')

    def squeeze(self, axis=None):  # pylint: disable=arguments-differ
        """Remove single-dimensional entries from the shape of a."""
        return _mx_np_op.squeeze(self, axis=axis)

    def broadcast_to(self, *args, **kwargs):
        raise AttributeError('_Symbol object has no attribute broadcast_to')

    def broadcast_like(self, *args, **kwargs):
        raise AttributeError('_Symbol object has no attribute broadcast_like')


@set_module('mxnet.symbol.numpy')
def zeros(shape, dtype=_np.float32, order='C', ctx=None):
    """Return a new array of given shape and type, filled with zeros.
    This function currently only supports storing multi-dimensional data
    in row-major (C-style).

    Parameters
    ----------
    shape : int or tuple of int
        The shape of the empty array.
    dtype : str or numpy.dtype, optional
        An optional value type. Default is `numpy.float32`. Note that this
        behavior is different from NumPy's `zeros` function  where `float64`
        is the default value, because `float32` is considered as the default
        data type in deep learning.
    order : {'C'}, optional, default: 'C'
        How to store multi-dimensional data in memory, currently only row-major
        (C-style) is supported.
    ctx : Context, optional
        An optional device context (default is the current default context).

    Returns
    -------
    out : Symbol
        Array of zeros with the given shape, dtype, and ctx.
    """
    if order != 'C':
        raise NotImplementedError
    if ctx is None:
        ctx = current_context()
    dtype = _np.float32 if dtype is None else dtype
    return _npi.zeros(shape=shape, ctx=ctx, dtype=dtype)


@set_module('mxnet.symbol.numpy')
def ones(shape, dtype=_np.float32, order='C', ctx=None):
    """Return a new array of given shape and type, filled with zeros.
    This function currently only supports storing multi-dimensional data
    in row-major (C-style).

    Parameters
    ----------
    shape : int or tuple of int
        The shape of the empty array.
    dtype : str or numpy.dtype, optional
        An optional value type. Default is `numpy.float32`. Note that this
        behavior is different from NumPy's `ones` function where `float64`
        is the default value, because `float32` is considered as the default
        data type in deep learning.
    order : {'C'}, optional, default: 'C'
        How to store multi-dimensional data in memory, currently only row-major
        (C-style) is supported.
    ctx : Context, optional
        An optional device context (default is the current default context).

    Returns
    -------
    out : ndarray
        Array of zeros with the given shape, dtype, and ctx.
    """
    if order != 'C':
        raise NotImplementedError
    if ctx is None:
        ctx = current_context()
    dtype = _np.float32 if dtype is None else dtype
    return _npi.ones(shape=shape, ctx=ctx, dtype=dtype)


#pylint: disable= too-many-arguments, no-member, protected-access
def _ufunc_helper(lhs, rhs, fn_array, fn_scalar, lfn_scalar, rfn_scalar=None, out=None):
    """ Helper function for element-wise operation.
    The function will perform numpy-like broadcasting if needed and call different functions.

    Parameters
    --------
    lhs : Symbol or numeric value
        Left-hand side operand.

    rhs : Symbol or numeric value
        Right-hand operand,

    fn_array : function
        Function to be called if both lhs and rhs are of ``Symbol`` type.

    fn_scalar : function
        Function to be called if both lhs and rhs are numeric values.

    lfn_scalar : function
        Function to be called if lhs is ``Symbol`` while rhs is numeric value

    rfn_scalar : function
        Function to be called if lhs is numeric value while rhs is ``Symbol``;
        if none is provided, then the function is commutative, so rfn_scalar is equal to lfn_scalar

    Returns
    --------
    mxnet.numpy.ndarray
        result array
    """
    if isinstance(lhs, numeric_types):
        if isinstance(rhs, numeric_types):
            return fn_scalar(lhs, rhs, out=out)
        else:
            if rfn_scalar is None:
                # commutative function
                return lfn_scalar(rhs, float(lhs), out=out)
            else:
                return rfn_scalar(rhs, float(lhs), out=out)
    elif isinstance(rhs, numeric_types):
        return lfn_scalar(lhs, float(rhs), out=out)
    elif isinstance(rhs, Symbol):
        return fn_array(lhs, rhs, out=out)
    else:
        raise TypeError('type %s not supported' % str(type(rhs)))
#pylint: enable= too-many-arguments, no-member, protected-access


@set_module('mxnet.symbol.numpy')
def add(x1, x2, out=None):
    return _ufunc_helper(x1, x2, _npi.add, _np.add, _npi.add_scalar, None, out)


@set_module('mxnet.symbol.numpy')
def subtract(x1, x2, out=None):
    return _ufunc_helper(x1, x2, _npi.subtract, _np.subtract, _npi.subtract_scalar,
                         _npi.rsubtract_scalar, out)


@set_module('mxnet.symbol.numpy')
def multiply(x1, x2, out=None):
    return _ufunc_helper(x1, x2, _npi.multiply, _np.multiply, _npi.multiply_scalar, None, out)


@set_module('mxnet.symbol.numpy')
def divide(x1, x2, out=None):
    return _ufunc_helper(x1, x2, _npi.true_divide, _np.divide, _npi.true_divide_scalar,
                         _npi.rtrue_divide_scalar, out)


@set_module('mxnet.symbol.numpy')
def mod(x1, x2, out=None):
    return _ufunc_helper(x1, x2, _npi.mod, _np.mod, _npi.mod_scalar, _npi.rmod_scalar, out)


@set_module('mxnet.symbol.numpy')
def power(x1, x2, out=None):
    return _ufunc_helper(x1, x2, _npi.power, _np.power, _npi.power_scalar, _npi.rpower_scalar, out)


@set_module('mxnet.symbol.numpy')
def tensordot(a, b, axes=2):
    r"""
    tensordot(a, b, axes=2)
    Compute tensor dot product along specified axes for arrays >= 1-D.
    Given two tensors (arrays of dimension greater than or equal to one),
    `a` and `b`, and an ndarray object containing two ndarray
    objects, ``(a_axes, b_axes)``, sum the products of `a`'s and `b`'s
    elements (components) over the axes specified by ``a_axes`` and
    ``b_axes``. The third argument can be a single non-negative
    integer_like scalar, ``N``; if it is such, then the last ``N``
    dimensions of `a` and the first ``N`` dimensions of `b` are summed
    over.
    Parameters
    ----------
    a, b : _Symbol
        Tensors to "dot".
    axes : int or (2,) ndarray
        * integer_like
        If an int N, sum over the last N axes of `a` and the first N axes
        of `b` in order. The sizes of the corresponding axes must match.
        * (2,) array_like
        Or, a list of axes to be summed over, first sequence applying to `a`,
        second to `b`. Both elements array_like must be of the same length.
    Notes
    -----
    Three common use cases are:
        * ``axes = 0`` : tensor product :math:`a\otimes b`
        * ``axes = 1`` : tensor dot product :math:`a\cdot b`
        * ``axes = 2`` : (default) tensor double contraction :math:`a:b`
    When `axes` is integer_like, the sequence for evaluation will be: first
    the -Nth axis in `a` and 0th axis in `b`, and the -1th axis in `a` and
    Nth axis in `b` last.
    When there is more than one axis to sum over - and they are not the last
    (first) axes of `a` (`b`) - the argument `axes` should consist of
    two sequences of the same length, with the first axis to sum over given
    first in both sequences, the second axis second, and so forth.
    """
    if _np.isscalar(axes):
        return _npi.tensordot_int_axes(a, b, axes)

    if len(axes) != 2:
        raise ValueError('Axes must consist of two arrays.')
    a_axes_summed, b_axes_summed = axes
    if _np.isscalar(a_axes_summed):
        a_axes_summed = (a_axes_summed,)
    if _np.isscalar(b_axes_summed):
        b_axes_summed = (b_axes_summed,)

    if len(a_axes_summed) != len(b_axes_summed):
        raise ValueError('Axes length mismatch')

    return _npi.tensordot(a, b, a_axes_summed, b_axes_summed)


@set_module('mxnet.symbol.numpy')
def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0, ctx=None): # pylint: disable=too-many-arguments
    r"""
    Return evenly spaced numbers over a specified interval.

    Returns num evenly spaced samples, calculated over the interval [start, stop].
    The endpoint of the interval can optionally be excluded.

    Parameters
    ----------
    start : real number
        The starting value of the sequence.
    stop : real number
        The end value of the sequence, unless endpoint is set to False. In
        that case, the sequence consists of all but the last of num + 1
        evenly spaced samples, so that stop is excluded. Note that the step
        size changes when endpoint is False.
    num : int, optional
        Number of samples to generate. Default is 50. Must be non-negative.
    endpoint : bool, optional
        If True, stop is the last sample. Otherwise, it is not included.
        Default is True.
    retstep : bool, optional
        If True, return (samples, step), where step is the spacing between samples.
    dtype : dtype, optional
        The type of the output array. If dtype is not given, infer the data
        type from the other input arguments.
    axis : int, optional
        The axis in the result to store the samples. Relevant only if start or
        stop are array-like. By default (0), the samples will be along a new
        axis inserted at the beginning. Use -1 to get an axis at the end.

    Returns
    -------
    samples : _Symbol
        There are num equally spaced samples in the closed interval
        `[start, stop]` or the half-open interval `[start, stop)`
        (depending on whether endpoint is True or False).
    step : float, optional
        Only returned if retstep is True
        Size of spacing between samples.


    See Also
    --------
    arange : Similar to `linspace`, but uses a step size (instead of the
             number of samples).

    Notes
    -----

    This function differs from the original `numpy.linspace
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html>`_ in
    the following aspects:

    - `start` and `stop` do not support list, numpy ndarray and mxnet ndarray
    - axis could only be 0
    - There could be an additional `ctx` argument to specify the device, e.g. the i-th
      GPU.
    """
    if isinstance(start, (list, _np.ndarray)) or isinstance(stop, (list, _np.ndarray)):
        raise NotImplementedError('start and stop only support int')
    if axis != 0:
        raise NotImplementedError("the function only support axis 0")
    if ctx is None:
        ctx = current_context()
    if retstep:
        step = (stop - start) / (num - 1)
        return _npi.linspace(start=start, stop=stop, num=num, endpoint=endpoint, ctx=ctx, dtype=dtype), step
    else:
        return _npi.linspace(start=start, stop=stop, num=num, endpoint=endpoint, ctx=ctx, dtype=dtype)


@set_module('mxnet.symbol.numpy')
def expand_dims(a, axis):
    """Expand the shape of an array.

    Insert a new axis that will appear at the `axis` position in the expanded

    Parameters
    ----------
    a : _Symbol
        Input array.
    axis : int
        Position in the expanded axes where the new axis is placed.

    Returns
    -------
    res : _Symbol
        Output array. The number of dimensions is one greater than that of
        the input array.
    """
    return _npi.expand_dims(a, axis)


def _unary_func_helper(x, fn_array, fn_scalar, out=None, **kwargs):
    """Helper function for unary operators.

    Parameters
    ----------
    x : _Symbol or scalar
        Input of the unary operator.
    fn_array : function
        Function to be called if x is of ``_Symbol`` type.
    fn_scalar : function
        Function to be called if x is a Python scalar.
    out : _Symbol
        Dummy parameter to keep the consistency with the ndarray counterpart.

    Returns
    -------
    out : _Symbol or scalar
        Result _Symbol or scalar.
    """
    if isinstance(x, numeric_types):
        return fn_scalar(x, **kwargs)
    elif isinstance(x, _Symbol):
        return fn_array(x, out=out, **kwargs)
    else:
        raise TypeError('type {} not supported'.format(str(type(x))))


@set_module('mxnet.symbol.numpy')
def tile(A, reps):
    r"""
    Construct an array by repeating A the number of times given by reps.

    If `reps` has length ``d``, the result will have dimension of
    ``max(d, A.ndim)``.

    If ``A.ndim < d``, `A` is promoted to be d-dimensional by prepending new
    axes. So a shape (3,) array is promoted to (1, 3) for 2-D replication,
    or shape (1, 1, 3) for 3-D replication. If this is not the desired
    behavior, promote `A` to d-dimensions manually before calling this
    function.

    If ``A.ndim > d``, `reps` is promoted to `A`.ndim by pre-pending 1's to it.
    Thus for an `A` of shape (2, 3, 4, 5), a `reps` of (2, 2) is treated as
    (1, 1, 2, 2).

    Parameters
    ----------
    A : _Symbol or scalar
        An input array or a scalar to repeat.
    reps : a single integer or tuple of integers
        The number of repetitions of `x` along each axis.

    Returns
    -------
    c : _Symbol
        The tiled output array.
    """
    return _unary_func_helper(A, _npi.tile, _np.tile, reps=reps)


@set_module('mxnet.symbol.numpy')
def arange(start, stop=None, step=1, dtype=None, ctx=None):
    """Return evenly spaced values within a given interval.

    Values are generated within the half-open interval ``[start, stop)``
    (in other words, the interval including `start` but excluding `stop`).
    For integer arguments the function is equivalent to the Python built-in
    `range` function, but returns an ndarray rather than a list.

    Parameters
    ----------
    start : number, optional
        Start of interval. The interval includes this value.  The default
        start value is 0.
    stop : number
        End of interval. The interval does not include this value, except
        in some cases where `step` is not an integer and floating point
        round-off affects the length of `out`.
    step : number, optional
        Spacing between values. For any output `out`, this is the distance
        between two adjacent values, ``out[i+1] - out[i]``.  The default
        step size is 1.  If `step` is specified as a position argument,
        `start` must also be given.
    dtype : dtype
        The type of the output array. The default is `float32`.

    Returns
    -------
    arange : ndarray
        Array of evenly spaced values.

        For floating point arguments, the length of the result is
        ``ceil((stop - start)/step)``.  Because of floating point overflow,
        this rule may result in the last element of `out` being greater
        than `stop`.
    """
    if dtype is None:
        dtype = 'float32'
    if ctx is None:
        ctx = current_context()
    if stop is None:
        stop = start
        start = 0
    if step is None:
        step = 1
    if start is None and stop is None:
        raise ValueError('start and stop cannot be both None')
    if step == 0:
        raise ZeroDivisionError('step cannot be 0')
    return _npi.arange(start=start, stop=stop, step=step, dtype=dtype, ctx=ctx)


@set_module('mxnet.symbol.numpy')
def split(ary, indices_or_sections, axis=0):
    """Split an array into multiple sub-arrays.
    Parameters
    ----------
    ary : ndarray
        Array to be divided into sub-arrays.
    indices_or_sections : int or 1-D array
        If `indices_or_sections` is an integer, N, the array will be divided
        into N equal arrays along `axis`.  If such a split is not possible,
        an error is raised.
        If `indices_or_sections` is a 1-D array of sorted integers, the entries
        indicate where along `axis` the array is split.  For example,
        ``[2, 3]`` would, for ``axis=0``, result in
          - ary[:2]
          - ary[2:3]
          - ary[3:]
        If an index exceeds the dimension of the array along `axis`,
        an empty sub-array is returned correspondingly.
    axis : int, optional
        The axis along which to split, default is 0.
    Returns
    -------
    sub-arrays : list of ndarrays
        A list of sub-arrays.
    Raises
    ------
    ValueError
        If `indices_or_sections` is given as an integer, but
        a split does not result in equal division."""
    indices = []
    sections = 0
    if isinstance(indices_or_sections, int):
        sections = indices_or_sections
    elif isinstance(indices_or_sections, tuple):
        indices = [0] + list(indices_or_sections)
    else:
        raise ValueError('indices_or_sections must either int or tuple of ints')
    ret = _npi.split(ary, indices, axis, False, sections)
    return ret


@set_module('mxnet.symbol.numpy')
def concatenate(seq, axis=0, out=None):
    """Join a sequence of arrays along an existing axis.
    Parameters
    ----------
    a1, a2, ... : sequence of array_like
        The arrays must have the same shape, except in the dimension
        corresponding to `axis` (the first, by default).
    axis : int, optional
        The axis along which the arrays will be joined.  If axis is None,
        arrays are flattened before use.  Default is 0.
    out : ndarray, optional
        If provided, the destination to place the result. The shape must be
        correct, matching that of what concatenate would have returned if no
        out argument were specified.
    Returns
    -------
    res : ndarray
        The concatenated array.
    """
    return _npi.concatenate(*seq, dim=axis, out=out)


@set_module('mxnet.symbol.numpy')
def stack(arrays, axis=0, out=None):
    """Join a sequence of arrays along a new axis.
        The axis parameter specifies the index of the new axis in the dimensions of the result.
        For example, if `axis=0` it will be the first dimension and if `axis=-1` it will be the last dimension.
    Parameters
    ----------
    arrays : sequence of array_like
        Each array must have the same shape.
    axis : int, optional
        The axis in the result array along which the input arrays are stacked.
    out : ndarray, optional
        If provided, the destination to place the result. The shape must be correct,
        matching that of what stack would have returned if no out argument were specified.
    Returns
    -------
    stacked : ndarray
        The stacked array has one more dimension than the input arrays."""
    def get_list(arrays):
        if not hasattr(arrays, '__getitem__') and hasattr(arrays, '__iter__'):
            raise ValueError("expected iterable for arrays but got {}".format(type(arrays)))
        return [arr for arr in arrays]

    arrays = get_list(arrays)
    return _npi.stack(*arrays, axis=axis, out=out)


@set_module('mxnet.symbol.numpy')
def rot90(m, k=1, axes=(0,1)):
    """
    Rotate an array by 90 degrees in the plane specified by axes.
    Rotation direction is from the first towards the second axis.
    Parameters
    ----------
    m : _Symbol
        Array of two or more dimensions.
    k : integer
        Number of times the array is rotated by 90 degrees.
    axes: (2,) array_like
        The array is rotated in the plane defined by the axes.
        Axes must be different.

    Returns
    -------
    y : _Symbol
        A rotated view of `m`.

    -----
    rot90(m, k=1, axes=(1,0)) is the reverse of rot90(m, k=1, axes=(0,1))
    rot90(m, k=1, axes=(1,0)) is equivalent to rot90(m, k=-1, axes=(0,1))
    Examples
    --------
    >>> m = np.array([[1,2],[3,4]], 'int')
    >>> m
    array([[1, 2],
           [3, 4]], dtype=int64)
    >>> np.rot90(m)
    array([[2, 4],
           [1, 3]], dtype=int64)
    >>> np.rot90(m, 2)
    array([[4, 3],
           [2, 1]], dtype=int64)
    >>> m = np.arange(8).reshape((2,2,2))
    >>> np.rot90(m, 1, (1,2))
    array([[[1., 3.],
            [0., 2.]],

           [[5., 7.],
            [4., 6.]]])
    """
    return _npi.rot90(m, k=k, axes=axes)


_set_np_symbol_class(_Symbol)
