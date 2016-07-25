from powderprojection.ProjectionCalculator import ProjectionCalculator
from mantid.simpleapi import SofQWNormalisedPolygon,mtd
from math import sqrt,cos


class MantidProjectionCalculator(ProjectionCalculator):
    def calculate_projection(self, input_workspace, output_workspace, qbinning, axis1, axis2):
        if axis1 == 'Energy' and axis2 == '|Q|':
            #TODO is EMode always direct ?
            SofQWNormalisedPolygon(InputWorkspace=input_workspace, OutputWorkspace=output_workspace,
                                                                QAxisBinning=qbinning,EMode='Direct')

    def calculate_suggested_binning(self, input_workspace):
        return '0,.1,10'
        #TODO AskInstrument scientist about calculating scattering angles
        input_workspace = mtd[input_workspace]
        ei = float(input_workspace.getRun().getLogData('Ei').value)
        ki = sqrt(ei/2.07)
        kf = sqrt((ei-0.0)/2.07)
        Qx=ki*1-cos(minTheta)*kf