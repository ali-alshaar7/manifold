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
    
    def __post_init__(self):
        if not isinstance(self.op_type, OpType):
            raise ValueError(f"Invalid op_type: {self.op_type}")

def add(a, b):
    """Element-wise addition"""
    return ManifoldOp(OpType.ADD, [a, b])

def sub(a, b):
    """Element-wise subtraction"""
    return ManifoldOp(OpType.SUB, [a, b])

def mul(a, b):
    """Element-wise multiplication"""
    return ManifoldOp(OpType.MUL, [a, b])

def div(a, b):
    """Element-wise division"""
    return ManifoldOp(OpType.DIV, [a, b])

def pow(a, b):
    """Element-wise power"""
    return ManifoldOp(OpType.POW, [a, b])

def neg(a):
    """Element-wise negation"""
    return ManifoldOp(OpType.NEG, [a])

def abs(a):
    """Element-wise absolute value"""
    return ManifoldOp(OpType.ABS, [a])

def sqrt(a):
    """Element-wise square root"""
    return ManifoldOp(OpType.SQRT, [a])

def exp(a):
    """Element-wise exponential"""
    return ManifoldOp(OpType.EXP, [a])

def log(a):
    """Element-wise logarithm"""
    return ManifoldOp(OpType.LOG, [a])

def sin(a):
    """Element-wise sine"""
    return ManifoldOp(OpType.SIN, [a])

def cos(a):
    """Element-wise cosine"""
    return ManifoldOp(OpType.COS, [a])

def tan(a):
    """Element-wise tangent"""
    return ManifoldOp(OpType.TAN, [a])

def dot(a, b):
    """Dot product"""
    return ManifoldOp(OpType.DOT, [a, b])

def reduce(a):
    """vector reduction"""
    return ManifoldOp(OpType.REDUCE, [a])
