import pytest
import manifold
import torch

def test_parse_simple_kernel():
    @manifold.kernel
    def matmul_kernel(A: manifold.inp, B: manifold.inp, C: manifold.out):
        A.set_slice("[..., i, :]")
        B.set_slice("[..., j]")
        C.set_slice("[..., i, j]")

        C = (A * B).reduce
