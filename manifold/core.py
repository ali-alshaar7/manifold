import inspect
from typing import get_type_hints, Tuple
import torch
import ast

from .ops import ManifoldOp, OpType

def kernel(func):
    """
    Manifold kernel decorator that converts Python functions into Triton kernels.
    """
    def wrapper(*args, **kwargs):
        type_hints = get_type_hints(func)

        manifold_args = _convert_tensor_to_manifold_symbolic_tensor(args, type_hints)
        for result in func(*manifold_args, **kwargs):
            triton_kernel = _generate_triton_kernel(type_hints, func.__name__)
        
        return triton_kernel
    
    return wrapper

def _convert_tensor_to_manifold_symbolic_tensor(args, type_hints):
    new_args = []
    for arg, param_name, arg_type in zip(args, type_hints.keys(), type_hints.values()):
        if isinstance(arg, torch.Tensor):
            if arg_type == out:
                new_arg = out(name=param_name, shape=arg.shape)
            elif arg_type == inp:
                new_arg = inp(name=param_name, shape=arg.shape)
            else:
                raise NotImplementedError
            new_args.append(new_arg)
        else:
            new_args.append(arg)
    
    return new_args

def _generate_triton_kernel(type_hints, func_name):
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
        name=func_name,
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

class inp:
    def __init__(self, name: str, shape: Tuple[int]):
        self.name = name
        self.slice = None
        self.shape = shape
    
    def set_slice(self, slice: str):
        self.slice = slice
        return self

class out:
    def __init__(self, name: str, shape: Tuple[int]):
        self.name = name
        self.shape = shape
        self.slice = None
        self.source = None
    
    def set_slice(self, slice: str):
        self.slice = slice
        return self

    def store(self, source: ManifoldOp | inp):
        self.source = source