from figuremanager import FigureManager,activate_category
from script_generation import script_log
from matplotlib.cbook import dedent, silent_list, is_string_like, is_numlike


def draw_if_interactive():
    # We will always draw because mslice might be running without matplotlib interactive
    for figure in FigureManager._figures.values():
        figure.canvas.draw()


def figure(num=None):
    return FigureManager.get_figure_number(num).get_figure()


def gcf():
    return FigureManager.get_active_figure().get_figure()


def draw_colorbar(function):
    def wrapper(*args, **kwargs):
        function(*args, **kwargs)
        cb = getattr(gcf(),'_colorbar_axes',None)
        if cb:
            cb.remove()
        cb = colorbar()
        gcf()._colorbar_axes = cb
    wrapper.__name__ = function.__name__
    return wrapper

# From here on just copy and paste from matplotlib.pyplot and decorate as appropriate
module_name = "plotting.pyplot"


@script_log(module_name)
def hold(state=None):
    """set hold of current axes to 'state', if no state is provided then it will toggle the hold state"""
    gca().hold(None)


@script_log(module_name)
def colorbar(mappable=None, cax=None, ax=None, **kw):
    if mappable is None:
        mappable = gci()
        if mappable is None:
            raise RuntimeError('No mappable was found to use for colorbar '
                               'creation. First define a mappable such as '
                               'an image (with imshow) or a contour set ('
                               'with contourf).')
    if ax is None:
        ax = gca()

    ret = gcf().colorbar(mappable, cax = cax, ax=ax, **kw)
    draw_if_interactive()
    return ret


def gca(**kwargs):
    """
    Get the current :class:`~matplotlib.axes.Axes` instance on the
    current figure matching the given keyword args, or create one.

    Examples
    ---------
    To get the current polar axes on the current figure::

        plt.gca(projection='polar')

    If the current axes doesn't exist, or isn't a polar one, the appropriate
    axes will be created and then returned.

    See Also
    --------
    matplotlib.figure.Figure.gca : The figure's gca method.
    """
    ax =  gcf().gca(**kwargs)
    return ax


def sci(im):
    """
    Set the current image.  This image will be the target of colormap
    commands like :func:`~matplotlib.pyplot.jet`,
    :func:`~matplotlib.pyplot.hot` or
    :func:`~matplotlib.pyplot.clim`).  The current image is an
    attribute of the current axes.
    """
    gca()._sci(im)


def gci():
    """
    Get the current colorable artist.  Specifically, returns the
    current :class:`~matplotlib.cm.ScalarMappable` instance (image or
    patch collection), or *None* if no images or patch collections
    have been defined.  The commands :func:`~matplotlib.pyplot.imshow`
    and :func:`~matplotlib.pyplot.figimage` create
    :class:`~matplotlib.image.Image` instances, and the commands
    :func:`~matplotlib.pyplot.pcolor` and
    :func:`~matplotlib.pyplot.scatter` create
    :class:`~matplotlib.collections.Collection` instances.  The
    current image is an attribute of the current axes, or the nearest
    earlier axes in the current figure that contains an image.
    """
    return gcf()._gci()

@script_log(module_name)
@activate_category("1d")
def plot(*args, **kwargs):
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()
    hold = kwargs.pop('hold', None)
    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.plot(*args, **kwargs)
        draw_if_interactive()
    finally:
        ax.hold(washold)

    return ret

@activate_category("1d")
def errorbar(x, y, yerr=None, xerr=None, fmt='', ecolor=None, elinewidth=None,
             capsize=None, barsabove=False, lolims=False, uplims=False,
             xlolims=False, xuplims=False, errorevery=1, capthick=None,
             hold=None, data=None, **kwargs):
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()

    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt=fmt, ecolor=ecolor,
                          elinewidth=elinewidth, capsize=capsize,
                          barsabove=barsabove, lolims=lolims, uplims=uplims,
                          xlolims=xlolims, xuplims=xuplims,
                          errorevery=errorevery, capthick=capthick, data=data,
                          **kwargs)
    finally:
        ax.hold(washold)

    return ret

def autoscale(enable=True, axis='both', tight=None):
    ret = gca().autoscale(enable=enable, axis=axis, tight=tight)
    return ret

@activate_category("2d")
@draw_colorbar
#@script_log("plotting.pyplot")
def imshow(X, cmap=None, norm=None, aspect=None, interpolation=None, alpha=None,
           vmin=None, vmax=None, origin=None, extent=None, shape=None,
           filternorm=1, filterrad=4.0, imlim=None, resample=None, url=None,
           hold=None, **kwargs):
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()

    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.imshow(X, cmap=cmap, norm=norm, aspect=aspect,
                        interpolation=interpolation, alpha=alpha, vmin=vmin,
                        vmax=vmax, origin=origin, extent=extent, shape=shape,
                        filternorm=filternorm, filterrad=filterrad,
                        imlim=imlim, resample=resample, url=url, **kwargs)
        draw_if_interactive()
    finally:
        ax.hold(washold)
    sci(ret)
    return ret

@activate_category('2d')
@draw_colorbar
@script_log(module_name)
def tripcolor(*args, **kwargs):
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()
    hold = kwargs.pop('hold', None)
    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.tripcolor(*args, **kwargs)
        draw_if_interactive()
    finally:
        ax.hold(washold)
    sci(ret)
    return ret

#*************************************************************************************************************************************
@script_log(module_name)
def xlabel(s, *args, **kwargs):
    """
    Set the *x* axis label of the current axis.
    Set the *x* axis label of the current axis.

    Default override is::

      override = {
          'fontsize'            : 'small',
          'verticalalignment'   : 'top',
          'horizontalalignment' : 'center'
          }

    .. seealso::

        :func:`~matplotlib.pyplot.text`
            For information on how override and the optional args work
    """
    l =  gca().set_xlabel(s, *args, **kwargs)
    draw_if_interactive()
    return l

@script_log(module_name)
def legend(*args, **kwargs):
    ret = gca().legend(*args, **kwargs)
    draw_if_interactive()
    return ret

@script_log(module_name)
def ylabel(s, *args, **kwargs):
    """
    Set the *y* axis label of the current axis.

    Defaults override is::

        override = {
           'fontsize'            : 'small',
           'verticalalignment'   : 'center',
           'horizontalalignment' : 'right',
           'rotation'='vertical' : }

    .. seealso::

        :func:`~matplotlib.pyplot.text`
            For information on how override and the optional args
            work.
    """
    l = gca().set_ylabel(s, *args, **kwargs)
    draw_if_interactive()
    return l

@script_log(module_name)
def xscale(*args, **kwargs):
    """
    Set the scaling of the *x*-axis.

    call signature::

      xscale(scale, **kwargs)

    The available scales are: %(scale)s

    Different keywords may be accepted, depending on the scale:

    %(scale_docs)s
    """
    ax = gca()
    ax.set_xscale(*args, **kwargs)
    draw_if_interactive()

@script_log(module_name)
def yscale(*args, **kwargs):
    """
    Set the scaling of the *y*-axis.

    call signature::

      yscale(scale, **kwargs)

    The available scales are: %(scale)s

    Different keywords may be accepted, depending on the scale:

    %(scale_docs)s
    """
    ax = gca()
    ax.set_yscale(*args, **kwargs)
    draw_if_interactive()

@script_log(module_name)
def xticks(*args, **kwargs):
    """
    Get or set the *x*-limits of the current tick locations and labels.

    ::

      # return locs, labels where locs is an array of tick locations and
      # labels is an array of tick labels.
      locs, labels = xticks()

      # set the locations of the xticks
      xticks( arange(6) )

      # set the locations and labels of the xticks
      xticks( arange(5), ('Tom', 'Dick', 'Harry', 'Sally', 'Sue') )

    The keyword args, if any, are :class:`~matplotlib.text.Text`
    properties. For example, to rotate long labels::

      xticks( arange(12), calendar.month_name[1:13], rotation=17 )
    """
    ax = gca()

    if len(args)==0:
        locs = ax.get_xticks()
        labels = ax.get_xticklabels()
    elif len(args)==1:
        locs = ax.set_xticks(args[0])
        labels = ax.get_xticklabels()
    elif len(args)==2:
        locs = ax.set_xticks(args[0])
        labels = ax.set_xticklabels(args[1], **kwargs)
    else: raise TypeError('Illegal number of arguments to xticks')
    if len(kwargs):
        for l in labels:
            l.update(kwargs)

    draw_if_interactive()
    return locs, silent_list('Text xticklabel', labels)

@script_log(module_name)
def yticks(*args, **kwargs):
    """
    Get or set the *y*-limits of the current tick locations and labels.

    ::

      # return locs, labels where locs is an array of tick locations and
      # labels is an array of tick labels.
      locs, labels = yticks()

      # set the locations of the yticks
      yticks( arange(6) )

      # set the locations and labels of the yticks
      yticks( arange(5), ('Tom', 'Dick', 'Harry', 'Sally', 'Sue') )

    The keyword args, if any, are :class:`~matplotlib.text.Text`
    properties. For example, to rotate long labels::

      yticks( arange(12), calendar.month_name[1:13], rotation=45 )
    """
    ax = gca()

    if len(args)==0:
        locs = ax.get_yticks()
        labels = ax.get_yticklabels()
    elif len(args)==1:
        locs = ax.set_yticks(args[0])
        labels = ax.get_yticklabels()
    elif len(args)==2:
        locs = ax.set_yticks(args[0])
        labels = ax.set_yticklabels(args[1], **kwargs)
    else: raise TypeError('Illegal number of arguments to yticks')
    if len(kwargs):
        for l in labels:
            l.update(kwargs)

    draw_if_interactive()

    return (locs,
             silent_list('Text yticklabel', labels)
             )

@script_log(module_name)
def xlim(*args, **kwargs):
    """
    Get or set the *x* limits of the current axes.

    ::

      xmin, xmax = xlim()   # return the current xlim
      xlim( (xmin, xmax) )  # set the xlim to xmin, xmax
      xlim( xmin, xmax )    # set the xlim to xmin, xmax

    If you do not specify args, you can pass the xmin and xmax as
    kwargs, e.g.::

      xlim(xmax=3) # adjust the max leaving min unchanged
      xlim(xmin=1) # adjust the min leaving max unchanged

    Setting limits turns autoscaling off for the x-axis.

    The new axis limits are returned as a length 2 tuple.

    """
    ax = gca()
    if not args and not kwargs:
        return ax.get_xlim()
    ret = ax.set_xlim(*args, **kwargs)
    draw_if_interactive()
    return ret


@script_log(module_name)
def ylim(*args, **kwargs):
    """
    Get or set the *y*-limits of the current axes.

    ::

      ymin, ymax = ylim()   # return the current ylim
      ylim( (ymin, ymax) )  # set the ylim to ymin, ymax
      ylim( ymin, ymax )    # set the ylim to ymin, ymax

    If you do not specify args, you can pass the *ymin* and *ymax* as
    kwargs, e.g.::

      ylim(ymax=3) # adjust the max leaving min unchanged
      ylim(ymin=1) # adjust the min leaving max unchanged

    Setting limits turns autoscaling off for the y-axis.

    The new axis limits are returned as a length 2 tuple.
    """
    ax = gca()
    if not args and not kwargs:
        return ax.get_ylim()
    ret = ax.set_ylim(*args, **kwargs)
    draw_if_interactive()
    return ret

@script_log(module_name)
@activate_category("2d")
@draw_colorbar
def pcolormesh(*args, **kwargs):
    ax = gca()
    # allow callers to override the hold state by passing hold=True|False
    washold = ax.ishold()
    hold = kwargs.pop('hold', None)
    if hold is not None:
        ax.hold(hold)
    try:
        ret = ax.pcolormesh(*args, **kwargs)
    finally:
        ax.hold(washold)
    sci(ret)
    return ret


if __name__ == '__main__':
    from PyQt4.QtGui import QApplication
    qapp = QApplication([])
    imshow([[1,2],[3,4]])
    plot([1,2,3,4])
    figure(1)
    xlabel('hi')
    qapp.exec_()