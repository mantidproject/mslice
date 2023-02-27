import abc
from six import add_metaclass

@add_metaclass(abc.ABCMeta)
class PlotSelectorPresenterInterface(object):

    @abc.abstractmethod
    def register_master(self, main_view):
        pass

    # @abc.abstractmethod
    # def load_workspace(self, file_paths, merge):
    #     pass
    #
    # @abc.abstractmethod
    # def _report_load_errors(self, ws_names, not_opened, not_loaded):
    #     pass

    # @abc.abstractmethod
    # def workspace_selection_changed(self):
    #     pass
