from abc import ABCMeta, abstractmethod


class WorkspaceBase(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_coordinates(self):
        return

    @abstractmethod
    def get_signal(self):
        return

    @abstractmethod
    def get_error(self):
        return

    @abstractmethod
    def get_variance(self):
        return

    @abstractmethod
    def rewrap(self, ws):
        return

    @abstractmethod
    def __neg__(self):
        return

    @abstractmethod
    def save_attributes(self):
        return

    @abstractmethod
    def remove_saved_attributes(self):
        return
