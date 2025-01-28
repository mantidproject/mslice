from matplotlib.legend import Legend


if hasattr(Legend, "set_draggable"):
    SET_DRAGGABLE_METHOD = "set_draggable"
else:
    SET_DRAGGABLE_METHOD = "draggable"


def legend_set_draggable(legend, state, use_blit=False, update="loc"):
    """Utility function to support varying Legend api around draggable status across
    the versions of matplotlib we support. Function arguments match those from matplotlib.
    See matplotlib documentation for argument descriptions
    """
    getattr(legend, SET_DRAGGABLE_METHOD)(state, use_blit, update)
