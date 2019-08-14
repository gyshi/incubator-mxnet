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

"""Doc placeholder for numpy ops with prefix _np."""


def _np_ones_like(a):
    """Return an array of ones with the same shape and type as a given array.

    Parameters
    ----------
    a : ndarray
        The shape and data-type of `a` define these same attributes of
        the returned array.

    Returns
    -------
    out : ndarray
        Array of ones with the same shape and type as `a`.
    """
    pass


def _np_zeros_like(a):
    """Return an array of zeros with the same shape and type as a given array.

    Parameters
    ----------
    a : ndarray
        The shape and data-type of `a` define these same attributes of
        the returned array.

    Returns
    -------
    out : ndarray
        Array of zeros with the same shape and type as `a`.
    """
    pass


def _np_roll(a, shift, axis=None):
    """Roll array elements along a given axis.

    Elements that roll beyond the last position are re-introduced at
    the first.

    Parameters
    ----------
    a : ndarray
    Input array.
    shift : int or tuple of ints
    The number of places by which elements are shifted.  If a tuple,
    then `axis` must be a tuple of the same size, and each of the
    given axes is shifted by the corresponding number.  If an int
    while `axis` is a tuple of ints, then the same value is used for
    all given axes.
    axis : int or tuple of ints, optional
    Axis or axes along which elements are shifted.  By default, the
    array is flattened before shifting, after which the original
    shape is restored.

    Returns
    -------
    res : ndarray
    Output array, with the same shape as `a`.

    Notes
    -----
    Supports rolling over multiple dimensions simultaneously.

    Examples
    --------
    >>> x = np.arange(10)
    >>> np.roll(x, 2)
    array([8., 9., 0., 1., 2., 3., 4., 5., 6., 7.])
    >>> np.roll(x, -2)
    array([2., 3., 4., 5., 6., 7., 8., 9., 0., 1.])

    >>> x2 = np.reshape(x, (2,5))
    >>> x2
    array([[0., 1., 2., 3., 4.],
           [5., 6., 7., 8., 9.]])
    >>> np.roll(x2, 1)
    array([[9., 0., 1., 2., 3.],
           [4., 5., 6., 7., 8.]])
    >>> np.roll(x2, -1)
    array([[1., 2., 3., 4., 5.],
           [6., 7., 8., 9., 0.]])
    >>> np.roll(x2, 1, axis=0)
    array([[5., 6., 7., 8., 9.],
           [0., 1., 2., 3., 4.]])
    >>> np.roll(x2, -1, axis=0)
    array([[5., 6., 7., 8., 9.],
           [0., 1., 2., 3., 4.]])
    >>> np.roll(x2, 1, axis=1)
    array([[4., 0., 1., 2., 3.],
           [9., 5., 6., 7., 8.]])
    >>> np.roll(x2, -1, axis=1)
    array([[1., 2., 3., 4., 0.],
           [6., 7., 8., 9., 5.]])
    """
    pass
