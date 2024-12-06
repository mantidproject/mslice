from __future__ import (absolute_import, division, print_function)
import unittest

import mock
from mock import MagicMock

from mantid.api import AnalysisDataService
from mantid.simpleapi import RenameWorkspace

from mslice.models.mslice_ads_observer import MSliceADSObserver
from mslice.presenters.interfaces.main_presenter import MainPresenterInterface
from mslice.presenters.workspace_manager_presenter import WorkspaceManagerPresenter
from mslice.views.interfaces.mainview import MainView
from mslice.views.interfaces.workspace_view import WorkspaceView
from mslice.util.mantid.mantid_algorithms import CreateSampleWorkspace


class WorkspaceManagerPresenterTest(unittest.TestCase):

    def setUp(self):
        self.view = mock.create_autospec(spec=WorkspaceView)
        self.mainview = mock.create_autospec(MainView)
        self.main_presenter = mock.create_autospec(MainPresenterInterface)
        self.mainview.get_presenter = mock.Mock(return_value=self.main_presenter)

    def test_ensure_that_the_ads_observer_calls_clear_handle(self):
        presenter = WorkspaceManagerPresenter(self.view)
        presenter.clear_handle = MagicMock()
        self.assertTrue(isinstance(presenter._ads_observer, MSliceADSObserver))
        presenter._ads_observer = MSliceADSObserver(
          presenter.delete_handle, presenter.clear_handle, presenter.rename_handle
        )

        AnalysisDataService.addOrReplace("ws", CreateSampleWorkspace(OutputWorkspace="ws"))
        AnalysisDataService.clear(True)

        presenter.clear_handle.assert_called_once()

    def test_ensure_that_the_ads_observer_calls_delete_handle(self):
        presenter = WorkspaceManagerPresenter(self.view)
        presenter.delete_handle = MagicMock()
        self.assertTrue(isinstance(presenter._ads_observer, MSliceADSObserver))
        presenter._ads_observer = MSliceADSObserver(
            presenter.delete_handle, presenter.clear_handle, presenter.rename_handle
        )

        AnalysisDataService.addOrReplace("ws", CreateSampleWorkspace(OutputWorkspace="ws"))
        AnalysisDataService.remove("ws")

        presenter.delete_handle.assert_called_once_with("ws")

    def test_ensure_that_the_ads_observer_calls_rename_handle(self):
        presenter = WorkspaceManagerPresenter(self.view)
        presenter.rename_handle = MagicMock()
        self.assertTrue(isinstance(presenter._ads_observer, MSliceADSObserver))
        presenter._ads_observer = MSliceADSObserver(
            presenter.delete_handle, presenter.clear_handle, presenter.rename_handle
        )

        AnalysisDataService.addOrReplace("ws", CreateSampleWorkspace(OutputWorkspace="ws"))
        RenameWorkspace(InputWorkspace="ws", OutputWorkspace="ws1")

        presenter.rename_handle.assert_called_once_with("ws", "ws1")


if __name__ == '__main__':
    unittest.main()
