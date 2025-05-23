import mslice.plotting.pyplot as plt
from mslice.plotting.globalfiguremanager import GlobalFigureManager

PICKER_TOL_PTS = 3


def draw_interactive_cut(workspace):
    cur_fig = plt.gcf()
    cur_canvas = cur_fig.canvas
    ax = plt.gca()

    # disconnect picking in interactive cut
    cur_canvas.manager.picking_connected(False)
    cur_canvas.manager.button_pressed_connected(False)

    if not cur_canvas.manager.has_plot_handler():
        cur_canvas.restore_region(cur_canvas.manager.get_cut_background())
        _create_cut(workspace)
    try:
        children = cur_fig.get_children()
        for artist in children:
            ax.draw_artist(artist)
        cur_canvas.blit(ax.clipbox)
    except AttributeError:
        cur_canvas.draw_idle()
    plt.show()


@plt.set_category(plt.CATEGORY_CUT)
def plot_cut_impl(
    workspace, intensity_range, plot_over=False, legend=None, en_conversion=True
):
    cur_fig = plt.gcf()
    axes = cur_fig.axes
    if len(axes) == 0:
        ax = cur_fig.add_subplot(111, projection="mslice")
    else:
        ax = axes[0]
        if not plot_over:
            ax.cla()

    legend = workspace.name if legend is None else legend
    ax.errorbar(
        workspace,
        "o-",
        label=legend,
        picker=PICKER_TOL_PTS,
        intensity_range=intensity_range,
        plot_over=plot_over,
        en_conversion=en_conversion,
    )
    if plot_over:
        cur_fig.canvas.manager.plot_handler.ws_list.append(workspace.name)
    else:
        cur_fig.canvas.manager.plot_handler.ws_list = [workspace.name]

    if cur_fig.canvas.manager.plot_handler.default_options is None:
        cur_fig.canvas.manager.plot_handler.save_default_options()

    if cur_fig.canvas.manager.plot_handler.is_icut():
        cur_fig.canvas.manager.plot_handler.update_bragg_peaks()

    return ax.lines


def _create_cut():
    canvas = plt.gcf().canvas
    # don't include axis ticks in the saved background
    canvas.figure.gca().xaxis.set_visible(False)
    canvas.figure.gca().yaxis.set_visible(False)
    canvas.draw()
    canvas.manager.set_cut_background(
        canvas.copy_from_bbox(plt.gcf().canvas.figure.bbox)
    )

    canvas.figure.gca().xaxis.set_visible(True)
    canvas.figure.gca().yaxis.set_visible(True)
    canvas.draw()


def cut_figure_exists():
    return GlobalFigureManager.active_cut_figure_exists()


def get_current_plot():
    return plt.gcf().canvas.manager.plot_handler
