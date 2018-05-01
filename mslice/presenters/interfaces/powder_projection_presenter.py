import abc
from six import add_metaclass


@add_metaclass(abc.ABCMeta)
class PowderProjectionPresenterInterface(object):

    @abc.abstractmethod
    def register_master(self, main_view):
        pass

    @abc.abstractmethod
    def notify(self, command):
        pass

    @abc.abstractmethod
    def workspace_selection_changed(self):
        pass
