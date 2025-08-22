import pytest
import manifold
import torch

def test_parse_simple_kernel():
    @manifold.kernel
    def matmul_kernel(A: manifold.inp, B: manifold.inp, C: manifold.out):
        A.set_slice("[..., i, :]")
        B.set_slice("[..., :, j]")
        C.set_slice("[..., i, j]")

        C.store(manifold.reduce(manifold.dot(A, B)))
        yield C

    A = torch.randn(2, 3)
    B = torch.randn(3, 4)
    C = torch.zeros(2, 4)
    kernel = matmul_kernel(A, B, C)
    print(kernel)
    assert "def matmul_kernel(A, B, C):" in kernel
