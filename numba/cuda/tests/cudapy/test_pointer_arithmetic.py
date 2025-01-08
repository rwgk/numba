from numba import cuda
from numba.core import types
from numba.cuda.testing import CUDATestCase
import unittest

import numba.cuda.pointer_arithmetic # noqa: F401


def pointer_add(ptr, distance):
    ptr + distance


def pointer_sub(ptr, distance):
    ptr - distance


class TestPointerArithmetic(CUDATestCase):

    def test_pointer_add(self):
        ptx, _ = cuda.compile(
            pointer_add,
            sig=types.void(types.CPointer(types.int32), types.uint64),
            output="ptx")
        self.assertGreater(len(ptx), 0)

    def test_pointer_sub(self):
        ptx, _ = cuda.compile(
            pointer_sub,
            sig=types.void(types.CPointer(types.int32), types.uint64),
            output="ptx")
        self.assertGreater(len(ptx), 0)


if __name__ == '__main__':
    unittest.main()
