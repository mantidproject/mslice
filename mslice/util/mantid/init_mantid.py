from mantid.api import AlgorithmFactory
import mantid.simpleapi as s_api
from mslice.models.cut.cut import Cut
from mslice.models.projection.powder.make_projection import MakeProjection
from mslice.models.slice.slice import Slice


def initialize_mantid():
    AlgorithmFactory.subscribe(MakeProjection)
    AlgorithmFactory.subscribe(Slice)
    AlgorithmFactory.subscribe(Cut)
    s_api._create_algorithm_function('MakeProjection', 1, MakeProjection())
    s_api._create_algorithm_function('Slice', 1, Slice())
    s_api._create_algorithm_function('Cut', 1, Cut())