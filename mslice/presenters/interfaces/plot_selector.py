import abc
from six import add_metaclass

@add_metaclass(abc.ABCMeta)
class PlotSelectorPresenterInterface(object):

    @abc.abstractmethod
    def register_master(self, main_view):
        pass
