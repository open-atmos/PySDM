"""
https://github.com/KatanaGraph/katana/blob/master/python/katana/numba_support/numpy_atomic.py
"""

from numba import types
from numba.core import cgutils
from numba.core.typing.arraydecl import get_array_index_type
from numba.extending import lower_builtin, type_callable
from numba.np.arrayobj import basic_indexing, make_array, normalize_indices


def _atomic_rmw(
    context, builder, operation, arrayty, val, ptr
):  # pylint: disable=too-many-arguments
    assert arrayty.aligned  # We probably have to have aligned arrays.
    dataval = context.get_value_as_data(builder, arrayty.dtype, val)
    return builder.atomic_rmw(operation, ptr, dataval, "monotonic")


def _declare_atomic_array_op(iop, uop, fop):
    def decorator(func):
        @type_callable(func)
        def func_type(context):
            def typer(ary, idx, val):
                out = get_array_index_type(ary, idx)
                if out is not None:
                    res = out.result
                    if context.can_convert(val, res):
                        return res
                return None

            return typer

        _ = func_type

        @lower_builtin(func, types.Buffer, types.Any, types.Any)
        # pylint: disable=too-many-locals
        def func_impl(context, builder, sig, args):
            """
            array[a] = scalar_or_array
            array[a,..,b] = scalar_or_array
            """
            aryty, idxty, valty = sig.args
            ary, idx, val = args

            if isinstance(idxty, types.BaseTuple):
                index_types = idxty.types
                indices = cgutils.unpack_tuple(builder, idx, count=len(idxty))
            else:
                index_types = (idxty,)
                indices = (idx,)

            ary = make_array(aryty)(context, builder, ary)

            # First try basic indexing to see if a single array location is denoted.
            index_types, indices = normalize_indices(
                context, builder, index_types, indices
            )
            dataptr, shapes, _strides = basic_indexing(
                context,
                builder,
                aryty,
                ary,
                index_types,
                indices,
                boundscheck=context.enable_boundscheck,
            )
            if shapes:
                raise NotImplementedError("Complex shapes are not supported")

            # Store source value the given location
            val = context.cast(builder, val, valty, aryty.dtype)
            operation = None
            if isinstance(aryty.dtype, types.Integer) and aryty.dtype.signed:
                operation = iop
            elif isinstance(aryty.dtype, types.Integer) and not aryty.dtype.signed:
                operation = uop
            elif isinstance(aryty.dtype, types.Float):
                operation = fop
            if operation is None:
                raise TypeError("Atomic operation not supported on " + str(aryty))
            return _atomic_rmw(context, builder, operation, aryty, val, dataptr)

        _ = func_impl

        return func

    return decorator


@_declare_atomic_array_op("add", "add", "fadd")
def atomic_add(ary, index, value):
    """
    Atomically, perform `ary[i] += v` and return the previous value of `ary[i]`.

    i must be a simple index for a single element of ary.
    Broadcasting and vector operations are not supported.

    This should be used from numba compiled code.
    """
    orig = ary[index]
    ary[index] += value
    return orig


@_declare_atomic_array_op("sub", "sub", "fsub")
def atomic_sub(ary, index, value):
    """
    Atomically, perform `ary[i] -= v` and return the previous value of `ary[i]`.

    i must be a simple index for a single element of ary.
    Broadcasting and vector operations are not supported.

    This should be used from numba compiled code.
    """
    orig = ary[index]
    ary[index] -= value
    return orig


@_declare_atomic_array_op("max", "umax", None)
def atomic_max(ary, index, value):
    """
    Atomically, perform `ary[i] = max(ary[i], v)` and return the previous value of `ary[i]`.
    This operation does not support floating-point values.

    i must be a simple index for a single element of ary.
    Broadcasting and vector operations are not supported.

    This should be used from numba compiled code.
    """
    orig = ary[index]
    ary[index] = max(ary[index], value)
    return orig


@_declare_atomic_array_op("min", "umin", None)
def atomic_min(ary, index, value):
    """
    Atomically, perform `ary[i] = min(ary[i], v)` and return the previous value of `ary[i]`.
    This operation does not support floating-point values.

    i must be a simple index for a single element of ary.
    Broadcasting and vector operations are not supported.

    This should be used from numba compiled code.
    """
    orig = ary[index]
    ary[index] = min(ary[index], value)
    return orig
