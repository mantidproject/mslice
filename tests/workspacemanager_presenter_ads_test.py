from __future__ import (absolute_import, division, print_function)
import unittest

from mock import MagicMock

from mantid.api import AnalysisDataService
from mantid.simpleapi import CreateSampleWorkspace, RenameWorkspace

from mslice.models.mslice_ads_observer import MSliceADSObserver
from mslice.presenters.workspace_manager_presenter import WorkspaceManagerPresenter


class WorkspaceManagerPresenterTest(unittest.TestCase):

    def setUp(self):
        AnalysisDataService.clear(True)

    def test_ensure_that_the_ads_observer_calls_delete_handle(self):
        presenter = WorkspaceManagerPresenter(MagicMock())
        presenter.delete_handle = MagicMock()
        self.assertTrue(isinstance(presenter._ads_observer, MSliceADSObserver))
        presenter._ads_observer = MSliceADSObserver(
            presenter.delete_handle, presenter.clear_handle, presenter.rename_handle
        )

        CreateSampleWorkspace(OutputWorkspace="ws", StoreInADS=True)
        AnalysisDataService.remove("ws")

        presenter.delete_handle.assert_called_once_with("ws")

    def test_ensure_that_the_ads_observer_calls_rename_handle(self):
        presenter = WorkspaceManagerPresenter(MagicMock())
        presenter.rename_handle = MagicMock()
        self.assertTrue(isinstance(presenter._ads_observer, MSliceADSObserver))
        presenter._ads_observer = MSliceADSObserver(
            presenter.delete_handle, presenter.clear_handle, presenter.rename_handle
        )

        CreateSampleWorkspace(OutputWorkspace="ws", StoreInADS=True)
        RenameWorkspace(InputWorkspace="ws", OutputWorkspace="ws1")

        presenter.rename_handle.assert_called_once_with("ws", "ws1")

    def test_ensure_that_the_ads_observer_calls_clear_handle(self):
        presenter = WorkspaceManagerPresenter(MagicMock())
        presenter.clear_handle = MagicMock()
        self.assertTrue(isinstance(presenter._ads_observer, MSliceADSObserver))
        presenter._ads_observer = MSliceADSObserver(
          presenter.delete_handle, presenter.clear_handle, presenter.rename_handle
        )

        CreateSampleWorkspace(OutputWorkspace="ws", StoreInADS=True)
        AnalysisDataService.clear(True)

        presenter.clear_handle.assert_called_once()


if __name__ == '__main__':
    unittest.main()
