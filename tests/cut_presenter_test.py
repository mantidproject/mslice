import unittest
from presenters.cut_presenter import CutPresenter
from views.cut_view import CutView
class CutPresenterTest(unittest.TestCase):
    def test_stub(self):
        # This unit test just forces travis CI to include the cut presenter in coverage a fail the build
        CutPresenter(CutView(), None, None)