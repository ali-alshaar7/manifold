import inspect
from typing import get_type_hints
import torch

def kernel(func):
    """
    A decorator to register a function as a manifold kernel.
    Automatically wraps tensors in inp/out types based on function signature.
    """
    signature = inspect.signature(func)
    type_hints = get_type_hints(func)
    
    def _parse_inputs(*args, **kwargs):
        """Parse input arguments and convert them to Manifold inp/out types."""
        new_args = []
        for i, arg in enumerate(args):
            param_name = list(signature.parameters)[i]
            param_type = type_hints.get(param_name)
            
            if isinstance(arg, torch.Tensor):
                if param_type is inp:
                    arg = inp()
                elif param_type is out:
                    arg = out()
            
            new_args.append(arg)
        
        new_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                param_type = type_hints.get(k)
                if param_type is inp:
                    v = inp()
                elif param_type is out:
                    v = out()
            new_kwargs[k] = v
        
        return new_args, new_kwargs

    def _generate_triton_kernel(type_hints):
        """Generate a Triton.jit kernel function based on type hints."""
        # Generate Triton kernel signature
        kernel_code = f"""
import triton

@triton.jit
def triton_kernel({', '.join(f'{param_name}' for param_name, _ in type_hints.items())}):
    pass
    """

        return kernel_code

    def wrapper(*args, **kwargs):
        """Wrapper function that processes inputs and generates Triton kernel."""
        # Parse inputs
        new_args, new_kwargs = _parse_inputs(*args, **kwargs)
        
        # Generate Triton kernel
        triton_kernel = _generate_triton_kernel(type_hints)
        
        return triton_kernel
    
    return wrapper

class inp:
    def set_slice(self, spec: str):
        self.spec = spec

class out:
    def set_slice(self, spec: str):
        self.spec = spec