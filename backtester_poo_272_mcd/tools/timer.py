import time

def timer(func):
    """Decorator used to compute the execution time of a method"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"{func.__qualname__} took {elapsed_time:.4f} sec to execute")
        return result
    return wrapper