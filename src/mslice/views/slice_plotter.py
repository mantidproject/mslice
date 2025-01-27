import mslice.plotting.pyplot as plt

PICKER_TOL_PTS = 5


def plot_cached_slice(slice_cache, slice_workspace):
    _show_plot(slice_cache, slice_workspace)


@plt.set_category(plt.CATEGORY_SLICE)
def create_slice_figure(workspace_name, presenter):
    fig_canvas = plt.gcf().canvas
    fig_canvas.manager.set_window_title(workspace_name)
    fig_canvas.manager.add_slice_plot(presenter, workspace_name)
    fig_canvas.manager.update_grid()
    plt.draw_all()


@plt.set_category(plt.CATEGORY_SLICE)
def _show_plot(slice_cache, workspace):
    cur_fig = plt.gcf()
    cur_fig.clf()
    ax = cur_fig.add_subplot(111, projection="mslice")
    image = ax.pcolormesh(workspace, cmap=slice_cache.colourmap, norm=slice_cache.norm)

    cb = plt.colorbar(image, ax=ax)
    cb.set_label(
        "Intensity (arb. units)", labelpad=20, rotation=270, picker=PICKER_TOL_PTS
    )

    plt_handler = cur_fig.canvas.manager.plot_handler

    plt_handler._update_lines()

    cur_fig.canvas.manager.plot_handler._update_lines()

    if plt_handler.icut is not None:
        # Because the axis is cleared, RectangleSelector needs to use the new axis
        plt_handler.icut.refresh_rect_selector(ax)

    cur_fig.canvas.draw_idle()
    cur_fig.show()

    # This ensures that another slice plotted in the same window saves the plot options
    # as the plot window's showEvent is called only once. The equivalent command is left
    # in the showEvent for use by the CLI.
    if plt_handler.default_options is None:
        plt_handler.save_default_options()


def set_colorbar_label(label):
    plt.gcf().get_axes()[1].set_ylabel(label, rotation=270, labelpad=20)
