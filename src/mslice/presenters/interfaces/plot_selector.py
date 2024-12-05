from abc import ABCMeta, abstractmethod


class PlotSelectorPresenterInterface(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def register_master(self, main_view):
        pass

    # @abstractmethod
    # def load_workspace(self, file_paths, merge):
    #     pass
    #
    # @abstractmethod
    # def _report_load_errors(self, ws_names, not_opened, not_loaded):
    #     pass

    # @abstractmethod
    # def workspace_selection_changed(self):
    #     pass
