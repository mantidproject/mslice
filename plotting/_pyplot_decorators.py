from functools import wraps

def draw_colorbar(function,gcf, colorbar):
    @wraps(function)
    def wrapper(*args, **kwargs):
        function(*args, **kwargs)
        cb = getattr(gcf(),'_colorbar_axes',None)
        if cb:
            cb.remove()
        cb = colorbar()
        gcf()._colorbar_axes = cb

    return wrapper
