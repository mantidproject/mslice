"""
Defines an algorithm to rescale a workspace from one temperature to another using the Bose factor
"""

import numpy as np
from scipy import constants
from mantid.api import PythonAlgorithm, WorkspaceProperty, WorkspaceUnitValidator
from mantid.kernel import Direction, FloatMandatoryValidator
from mantid.dataobjects import Workspace2D
from mantid.simpleapi import CloneWorkspace

# Defines some conversion factors
KB_MEV = constants.value('Boltzmann constant in eV/K') * 1000


class Rebose(PythonAlgorithm):

    def PyInit(self):
        self.declareProperty(WorkspaceProperty('InputWorkspace', '', direction=Direction.Input,
                                               validator=WorkspaceUnitValidator('DeltaE')))
        self.declareProperty('CurrentTemperature', 300., FloatMandatoryValidator())
        self.declareProperty('TargetTemperature', 5., FloatMandatoryValidator())
        self.declareProperty(WorkspaceProperty('OutputWorkspace', '', direction=Direction.Output))

    def PyExec(self):
        workspace = self.getProperty('InputWorkspace').value
        if not isinstance(workspace, Workspace2D):
            raise RuntimeError('Invalid workspace type - must be Workspace2D')
        from_temp = self.getProperty('CurrentTemperature').value
        to_temp = self.getProperty('TargetTemperature').value
        y = workspace.extractY()
        e = workspace.extractE()
        en = workspace.getAxis(0).extractValues()
        en = (en[1:] + en[:-1]) / 2
        sg = np.sign(en)
        sg[sg == 0] = 1
        bose_old = (sg - sg * np.exp(-en/(KB_MEV * from_temp)))
        bose_new = (sg - sg * np.exp(-en/(KB_MEV * to_temp)))
        y = y * bose_old / bose_new
        e = e * bose_old / bose_new
        result = CloneWorkspace(workspace, StoreInADS=False, EnableLogging=False)
        for j in range(len(y)):
            result.setY(j, y[j,:])
            result.setE(j, e[j,:])
        self.setProperty('OutputWorkspace', result)

    def category(self):
        return 'MSlice'
