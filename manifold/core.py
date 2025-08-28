import inspect
from typing import get_type_hints, List
from collections import deque, defaultdict
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
            assert isinstance(result, out)
            dag = _traverse_op_graph(result)
            ordered_dag = _topo_sort(dag)
            _calculate_slices(ordered_dag)
            triton_kernel = _generate_triton_kernel_header(manifold_args, func.__name__)
        
        return triton_kernel
    
    return wrapper

def _traverse_op_graph(result):
    dag = {}
    visited = set()
    queue = deque([result])

    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)

        if isinstance(node, out) and node.source is not None:
            # node is an out tensor, produced by an op
            dag[node] = [node.source]
            queue.append(node.source)
        elif isinstance(node, ManifoldOp):  
            # node is an op
            dag[node] = list(node.operands)
            for operand in node.operands:
                queue.append(operand)
        elif isinstance(node, inp):
            dag[node] = []
        else:
            raise NotImplementedError

    return dag

def _topo_sort(dag):
    children = defaultdict(list)
    indegree = {node: 0 for node in dag}

    for node, parents in dag.items():
        for p in parents:
            children[p].append(node)
            indegree[node] += 1
            if p not in indegree:  # ensure parent is in indegree map
                indegree[p] = 0

    # Collect zero-indegree nodes (leaves)
    queue = deque([n for n, d in indegree.items() if d == 0])
    order = []

    while queue:
        n = queue.popleft()
        order.append(n)
        for c in children[n]:
            indegree[c] -= 1
            if indegree[c] == 0:
                queue.append(c)

    if len(order) != len(indegree):
        raise ValueError("Cycle detected in DAG!")

    return order

def _calculate_slices(dag):
    tensor_slices = {}
    for node in dag:
        if isinstance(node, inp) or isinstance(node, out):
            tensor_slices[node] = _parse_slice(node.slice, len(node.shape))

    print(tensor_slices)

def _parse_slice(slice: str, rank: int):
    slice_content = slice.strip()[1:-1]
    elements = [elem.strip() for elem in slice_content.split(',')]
    
    result = {
        'batch_dims': [],
        'index_dims': {},
        'vector_dims': [],
        'constant_dims': {},
    }
    
    real_dims = 0
    batch_dim_start = -1
    for i, elem in enumerate(elements):
        if elem != '...':
            real_dims += 1
        else:
            batch_dim_start = i
        
    batch_dims = rank - real_dims
    assert batch_dims >= 0
    if batch_dim_start == -1:
        assert rank == real_dims

    if batch_dims == 0:
        del elements[batch_dim_start]  # Remove the '...' element at the specific index
    elif batch_dims > 1:
        for i in range(batch_dims - 1):
            elements.insert(batch_dim_start, "...")

    for i, elem in enumerate(elements):
        if elem == "...":
            result["batch_dims"].append(i)
        elif elem == ":":
            result["vector_dims"].append(i)
        elif elem.isnumeric():
            result["constant_dims"][i] = [elem]
        elif elem.isalpha():
            result["index_dims"][i] = [elem]
        else:
            raise TypeError

    return result

def _convert_tensor_to_manifold_symbolic_tensor(args, type_hints):
    new_args = []
    for arg, param_name, arg_type in zip(args, type_hints.keys(), type_hints.values()):
        if isinstance(arg, torch.Tensor):
            if arg_type == out:
                new_arg = out(name=param_name, shape=list(arg.shape), strides=list(arg.stride()))
            elif arg_type == inp:
                new_arg = inp(name=param_name, shape=list(arg.shape), strides=list(arg.stride()))
            else:
                raise NotImplementedError
            new_args.append(new_arg)
        else:
            new_args.append(arg)
    
    return new_args

def _generate_triton_kernel_header(manifold_args, func_name):
    
    # Create the import statement
    import_node = ast.Import(names=[ast.alias(name='triton')])
    
    # Create the decorator
    decorator = ast.Attribute(value=ast.Name(id='triton', ctx=ast.Load()), 
                            attr='jit', ctx=ast.Load())
    
    # Create the function arguments
    args = ast.arguments(
        posonlyargs=[],
        args=[ast.arg(arg=arg.name) for arg in manifold_args],
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
    def __init__(self, name: str, shape: List[int], strides: List[int]):
        self.name = name
        self.slice = None
        self.shape = shape
        self.strides = strides
    
    def set_slice(self, slice: str):
        self.slice = slice
        return self

class out:
    def __init__(self, name: str, shape: List[int], strides: List[int]):
        self.name = name
        self.shape = shape
        self.strides = strides
        self.slice = None
        self.source = None
    
    def set_slice(self, slice: str):
        self.slice = slice
        return self

    def store(self, source: ManifoldOp | inp):
        self.source = source