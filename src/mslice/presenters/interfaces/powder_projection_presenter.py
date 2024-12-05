from abc import ABCMeta, abstractmethod


class PowderProjectionPresenterInterface(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def register_master(self, main_view):
        pass

    @abstractmethod
    def notify(self, command):
        pass

    @abstractmethod
    def workspace_selection_changed(self):
        pass
