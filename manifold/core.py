import inspect
from typing import get_type_hints
import torch
import ast

from .ops import ManifoldOp, OpType

def kernel(func):
    """
    Manifold kernel decorator that converts Python functions into Triton kernels.
    """
    def wrapper(*args, **kwargs):
        # Get type hints for function parameters
        type_hints = get_type_hints(func)
        
        # Generate Triton kernel
        triton_kernel = _generate_triton_kernel(type_hints)
        
        return triton_kernel
    
    return wrapper

def _generate_triton_kernel(type_hints):
    """Generate a Triton.jit kernel function based on Manifold operations."""
    import ast
    
    # Create the import statement
    import_node = ast.Import(names=[ast.alias(name='triton')])
    
    # Create the decorator
    decorator = ast.Attribute(value=ast.Name(id='triton', ctx=ast.Load()), 
                            attr='jit', ctx=ast.Load())
    
    # Create the function arguments
    args = ast.arguments(
        posonlyargs=[],
        args=[ast.arg(arg=param_name) for param_name in type_hints.keys()],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    
    # Create the function body
    function_body = []
    
    # Create the function definition
    function_def = ast.FunctionDef(
        name='triton_kernel',
        args=args,
        body=function_body,
        decorator_list=[decorator],
        returns=None
    )
    
    # Create the module
    module = ast.Module(body=[import_node, function_def], type_ignores=[])
    
    # Convert AST back to code
    ast.fix_missing_locations(module)
    kernel_code = ast.unparse(module)
    
    return kernel_code

def _generate_triton_op(op: ManifoldOp, var_name: str) -> ast.AST:
    """Generate Triton AST node for a Manifold operation."""
    if op.op_type == OpType.ADD:
        return ast.BinOp(
            left=_generate_triton_op(op.operands[0], var_name),
            op=ast.Add(),
            right=_generate_triton_op(op.operands[1], var_name)
        )
    elif op.op_type == OpType.SUB:
        return ast.BinOp(
            left=_generate_triton_op(op.operands[0], var_name),
            op=ast.Sub(),
            right=_generate_triton_op(op.operands[1], var_name)
        )
    elif op.op_type == OpType.MUL:
        return ast.BinOp(
            left=_generate_triton_op(op.operands[0], var_name),
            op=ast.Mult(),
            right=_generate_triton_op(op.operands[1], var_name)
        )
    elif op.op_type == OpType.DIV:
        return ast.BinOp(
            left=_generate_triton_op(op.operands[0], var_name),
            op=ast.Div(),
            right=_generate_triton_op(op.operands[1], var_name)
        )
    elif op.op_type == OpType.POW:
        return ast.BinOp(
            left=_generate_triton_op(op.operands[0], var_name),
            op=ast.Pow(),
            right=_generate_triton_op(op.operands[1], var_name)
        )
    elif op.op_type == OpType.NEG:
        return ast.UnaryOp(
            op=ast.USub(),
            operand=_generate_triton_op(op.operands[0], var_name)
        )
    elif op.op_type == OpType.ABS:
        return ast.Call(
            func=ast.Attribute(value=ast.Name(id='triton', ctx=ast.Load()),
                             attr='abs', ctx=ast.Load()),
            args=[_generate_triton_op(op.operands[0], var_name)],
            keywords=[]
        )
    elif op.op_type == OpType.SQRT:
        return ast.Call(
            func=ast.Attribute(value=ast.Name(id='triton', ctx=ast.Load()),
                             attr='sqrt', ctx=ast.Load()),
            args=[_generate_triton_op(op.operands[0], var_name)],
            keywords=[]
        )
    elif op.op_type == OpType.EXP:
        return ast.Call(
            func=ast.Attribute(value=ast.Name(id='triton', ctx=ast.Load()),
                             attr='exp', ctx=ast.Load()),
            args=[_generate_triton_op(op.operands[0], var_name)],
            keywords=[]
        )
    elif op.op_type == OpType.LOG:
        return ast.Call(
            func=ast.Attribute(value=ast.Name(id='triton', ctx=ast.Load()),
                             attr='log', ctx=ast.Load()),
            args=[_generate_triton_op(op.operands[0], var_name)],
            keywords=[]
        )
    elif op.op_type == OpType.SIN:
        return ast.Call(
            func=ast.Attribute(value=ast.Name(id='triton', ctx=ast.Load()),
                             attr='sin', ctx=ast.Load()),
            args=[_generate_triton_op(op.operands[0], var_name)],
            keywords=[]
        )
    elif op.op_type == OpType.COS:
        return ast.Call(
            func=ast.Attribute(value=ast.Name(id='triton', ctx=ast.Load()),
                             attr='cos', ctx=ast.Load()),
            args=[_generate_triton_op(op.operands[0], var_name)],
            keywords=[]
        )
    elif op.op_type == OpType.TAN:
        return ast.Call(
            func=ast.Attribute(value=ast.Name(id='triton', ctx=ast.Load()),
                             attr='tan', ctx=ast.Load()),
            args=[_generate_triton_op(op.operands[0], var_name)],
            keywords=[]
        )
    elif isinstance(op, str):  # Handle input variable names
        return ast.Name(id=op, ctx=ast.Load())
    else:
        raise ValueError(f"Unsupported operation type: {op.op_type}")

def _traverse_operations(output: str) -> list[ast.AST]:
    """Traverse from output to inputs to generate Triton operations."""
    operations = []
    
    def traverse(op):
        if isinstance(op, str):
            # Base case: we've reached an input
            return
        elif isinstance(op, ManifoldOp):
            # Recursively traverse operands
            for operand in op.operands:
                traverse(operand)
            
            # Generate Triton operation
            operations.append(_generate_triton_op(op, op.operands[0]))
    
    traverse(output)
    return operations

class inp:
    def __init__(self, name: str):
        self.name = name
        self.spec = None
    
    def set_slice(self, spec: str):
        self.spec = spec
        return self

class out:
    def __init__(self, name: str):
        self.name = name
        self.spec = None
        self.source = None
    
    def set_slice(self, spec: str):
        self.spec = spec
        return self

    def store(self, source: ManifoldOp | inp):
        self.source = source