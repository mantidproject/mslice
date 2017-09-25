import abc


class WorkspaceBase(object):
    __metaclass__ = abc.ABCMeta

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
    def __add__(self, other):
        return

    @abc.abstractmethod
    def __sub__(self, other):
        return

    @abc.abstractmethod
    def __mul__(self, other):
        return

    @abc.abstractmethod
    def __div__(self, other):
        return

    @abc.abstractmethod
    def __pow__(self, other):
        return

    @abc.abstractmethod
    def __neg__(self):
        return
