from __future__ import (absolute_import, division, print_function)
import abc


class ProjectionCalculator(object):
    @abc.abstractmethod
    def available_axes(self):
        pass

    @abc.abstractmethod
    def calculate_projection(self, input_workspace, axis1, axis2):
        pass
