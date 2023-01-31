import abc
from six import add_metaclass


@add_metaclass(abc.ABCMeta)
class WorkspaceBase(object):

    @abc.abstractmethod
    def get_coordinates(self):
        return

    @abc.abstractmethod
    def get_signal(self):
        return

    @abc.abstractmethod
    def get_error(self):
        return

    @abc.abstractmethod
    def get_variance(self):
        return

    @abc.abstractmethod
    def rewrap(self, ws):
        return

    @abc.abstractmethod
    def __neg__(self):
        return

    @abc.abstractmethod
    def save_attributes(self):
        return

    @abc.abstractmethod
    def remove_saved_attributes(self):
        return
