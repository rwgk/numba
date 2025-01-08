import operator
from numba import types
from numba.core.extending import overload, intrinsic
from llvmlite import ir


_DEVICE_POINTER_SIZE = 8
_DEVICE_POINTER_BITWIDTH = _DEVICE_POINTER_SIZE * 8


def sizeof_pointee(context, ptr):
    size = context.get_abi_sizeof(ptr.type.pointee)
    return ir.Constant(ir.IntType(_DEVICE_POINTER_BITWIDTH), size)


def make_pointer_op_intrinsic(op_name):
    @intrinsic
    def pointer_op_intrinsic(context, ptr, offset):
        def codegen(context, builder, sig, args):
            ptr, index = args
            base = builder.ptrtoint(ptr, ir.IntType(_DEVICE_POINTER_BITWIDTH))
            offset = builder.mul(index, sizeof_pointee(context, ptr))
            result = getattr(builder, op_name)(base, offset)
            return builder.inttoptr(result, ptr.type)

        return ptr(ptr, offset), codegen

    return pointer_op_intrinsic


pointer_add_intrinsic = make_pointer_op_intrinsic("add")
pointer_sub_intrinsic = make_pointer_op_intrinsic("sub")


@overload(operator.add)
def pointer_add(ptr, offset):
    if not isinstance(ptr, types.CPointer) or not isinstance(
        offset, types.Integer
    ):
        return

    def impl(ptr, offset):
        return pointer_add_intrinsic(ptr, offset)

    return impl


@overload(operator.sub)
def pointer_sub(ptr, offset):
    if not isinstance(ptr, types.CPointer) or not isinstance(
        offset, types.Integer
    ):
        return

    def impl(ptr, offset):
        return pointer_sub_intrinsic(ptr, offset)

    return impl
