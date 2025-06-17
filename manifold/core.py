def kernel(func):
    """
    A decorator to register a function as a GPU kernel.
    """
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    return wrapper