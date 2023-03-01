import abc
from six import add_metaclass

@add_metaclass(abc.ABCMeta)
class IPlot(object):

    @abc.abstractmethod
    def window_closing(self):
        pass

    @abc.abstractmethod
    def plot_options(self):
        pass

    @abc.abstractmethod
    def plot_clicked(self, x, y):
        pass

    @abc.abstractmethod
    def object_clicked(self, target):
        pass

    @abc.abstractmethod
    def update_legend(self):
        pass

    @abc.abstractmethod
    def get_line_options(self, line):
        pass

    @abc.abstractmethod
    def set_line_options(self, line, line_options):
        pass
