"""
Sets up mslice specific algorithms. This code executes when the module is imported (and should be imported first) as
it can affect other imports.
"""

from mantid.api import AlgorithmFactory
import mantid.simpleapi as s_api
from mslice.models.cut.cut_algorithm import Cut
from mslice.models.projection.powder.make_projection import MakeProjection
from mslice.models.slice.slice_algorithm import Slice
from mslice.models.workspacemanager.rebose_algorithm import Rebose

AlgorithmFactory.subscribe(MakeProjection)
AlgorithmFactory.subscribe(Slice)
AlgorithmFactory.subscribe(Cut)
AlgorithmFactory.subscribe(Rebose)
s_api._create_algorithm_function('MakeProjection', 1, MakeProjection())
s_api._create_algorithm_function('Slice', 1, Slice())
s_api._create_algorithm_function('Cut', 1, Cut())
s_api._create_algorithm_function('Rebose', 1, Rebose())
