"""
Manifold Operations - Core mathematical operations for Manifold DSL
"""
from typing import Any
from dataclasses import dataclass
from enum import Enum


class OpType(Enum):
    """Types of operations supported by Manifold"""
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    DOT = "dot"
    POW = "pow"
    NEG = "neg"
    REDUCE = "reduce"
    ABS = "abs"
    SQRT = "sqrt"
    EXP = "exp"
    LOG = "log"
    SIN = "sin"
    COS = "cos"
    TAN = "tan"


@dataclass
class ManifoldOp:
    """Base class for Manifold operations"""
    op_type: OpType
    operands: list[Any]
    triton_op: str = None
    
    def __post_init__(self):
        if not isinstance(self.op_type, OpType):
            raise ValueError(f"Invalid op_type: {self.op_type}")
    
    def __hash__(self):
        """Make ManifoldOp hashable using object identity (pointer)"""
        return hash(id(self))

def add(a, b):
    """Element-wise addition"""
    return ManifoldOp(OpType.ADD, [a, b], triton_op="+")

def sub(a, b):
    """Element-wise subtraction"""
    return ManifoldOp(OpType.SUB, [a, b], triton_op="-")

def mul(a, b):
    """Element-wise multiplication"""
    return ManifoldOp(OpType.MUL, [a, b], triton_op="*")

def div(a, b):
    """Element-wise division"""
    return ManifoldOp(OpType.DIV, [a, b], triton_op="/")

def pow(a, b):
    """Element-wise power"""
    return ManifoldOp(OpType.POW, [a, b], triton_op="tl.math.pow")

def neg(a):
    """Element-wise negation"""
    return ManifoldOp(OpType.NEG, [a], triton_op="-")

def abs(a):
    """Element-wise absolute value"""
    return ManifoldOp(OpType.ABS, [a], triton_op="tl.abs")

def sqrt(a):
    """Element-wise square root"""
    return ManifoldOp(OpType.SQRT, [a], triton_op="tl.sqrt")

def exp(a):
    """Element-wise exponential"""
    return ManifoldOp(OpType.EXP, [a], triton_op="tl.exp")

def log(a):
    """Element-wise logarithm"""
    return ManifoldOp(OpType.LOG, [a], triton_op="tl.log")

def sin(a):
    """Element-wise sine"""
    return ManifoldOp(OpType.SIN, [a], triton_op="tl.sin")

def cos(a):
    """Element-wise cosine"""
    return ManifoldOp(OpType.COS, [a], triton_op="tl.cos")

def tan(a):
    """Element-wise tangent"""
    return ManifoldOp(OpType.TAN, [a], triton_op="tl.tan")

def dot(a, b):
    """Dot product"""
    return ManifoldOp(OpType.DOT, [a, b], triton_op="tl.dot")

def reduce(a):
    """vector reduction"""
    return ManifoldOp(OpType.REDUCE, [a], triton_op="tl.sum")
