from powderprojection.ProjectionCalculator import ProjectionCalculator
from mantid.simpleapi import SofQWNormalisedPolygon

class MantidProjectionCalculator(ProjectionCalculator):
    def CalculateProjections(self,input_workspace,output_workspace,qbinning,axis1,axis2):
        if axis1 == 'Energy' and axis2 == '|Q|':
            #TODO is EMode always direct ?
            SofQWNormalisedPolygon(InputWorkspace=input_workspace, OutputWorkspace=output_workspace,
                                                                QAxisBinning=qbinning,EMode='Direct')
