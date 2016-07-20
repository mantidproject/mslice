import abc

class ProjectionCalculator(object):
    @abc.abstractmethod
    def CalculateProjections(self,input_workspace,output_workspace,qbinning,axis1,axis2):
        pass
    def CalculateSuggestedBinning(self,input_workspace):
        pass
