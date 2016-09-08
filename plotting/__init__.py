import os


def get_figure_class():
    """Check envrionment variable MSLICE_PLOT_WINDOWS to see where to get figure class from """

    env_variable_name = 'MSLICE_PLOT_WINDOWS'
    if env_variable_name in os.environ.keys():
        key = os.environ[env_variable_name]
        if key == 'TEST':
            return None

        elif key == 'MSLICE_DEFAULT':
            from plotting.plot_window.plot_figure import PlotFigureManager
            return PlotFigureManager

        else:
            print "Warning :: Using MSlice Figures because The %s environment vairable was not set to an unrecognized " \
                  "value '%s'" % (env_variable_name,key)

    else:
        print "Warning :: Using MSlice Figures because The %s environment variable was not set" % env_variable_name
        from plotting.plot_window.plot_figure import PlotFigureManager
        return PlotFigureManager
