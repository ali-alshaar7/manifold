import pytest
import manifold
import torch

def test_parse_simple_kernel():
    @manifold.kernel
    def matmul_kernel(A: manifold.inp, B: manifold.inp, C: manifold.out):
        A.set_slice("[..., i]")
        B.set_slice("[..., i]")
        C.set_slice("[..., i]")

        C.store(manifold.add(A, B))
        yield C

    A = torch.randn(2, 3)
    B = torch.randn(3, 4)
    C = torch.zeros(2, 4)
    kernel = matmul_kernel(A, B, C)
    print(kernel)
    assert "def triton_kernel(A, B, C):" in kernel
