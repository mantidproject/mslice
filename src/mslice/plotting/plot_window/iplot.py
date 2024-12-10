from abc import ABCMeta, abstractmethod


class IPlot(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def window_closing(self):
        pass

    @abstractmethod
    def plot_options(self):
        pass

    @abstractmethod
    def plot_clicked(self, x, y):
        pass

    @abstractmethod
    def object_clicked(self, target):
        pass

    @abstractmethod
    def update_legend(self):
        pass

    @abstractmethod
    def get_line_options(self, line):
        pass

    @abstractmethod
    def set_line_options(self, line, line_options):
        pass
