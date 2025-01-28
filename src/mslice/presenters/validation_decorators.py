from functools import wraps


def require_main_presenter(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        if not args[0]._main_presenter:
            raise Exception(
                "Main presenter has not been registered, it must be registered by the register_master method"
                "before This presenter can be used"
            )
        return function(*args, **kwargs)

    return wrapper
