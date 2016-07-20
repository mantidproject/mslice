import abc

class ProjectionCalculator(object):
    @abc.abstractmethod
    def CalculateProjections(self,workspace,axis1,axis2):
        pass

