from numba import cuda
from numba.core import types
from numba.cuda.testing import CUDATestCase
import unittest

import numba.cuda.pointer_arithmetic # noqa: F401


def pointer_advance(this, distance):
    this[0] = this[0] + distance


class TestPointerArithmetic(CUDATestCase):

    def test_compile_pointer_pointer_int32(self):
        thisty = types.CPointer(types.CPointer(types.int32))
        ptx, _ = cuda.compile(
            pointer_advance,
            sig=types.void(thisty, types.uint64),
            output="ptx")
        self.assertGreater(len(ptx), 0)


if __name__ == '__main__':
    unittest.main()
