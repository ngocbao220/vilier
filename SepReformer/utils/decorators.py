import functools

def logger_wraps(*, entry=True, exit=True, level="DEBUG"):
    def wrapper(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            result = func(*args, **kwargs)
            return result
        return wrapped
    return wrapper
