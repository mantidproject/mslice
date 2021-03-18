from __future__ import (absolute_import, division, print_function)
import unittest
from unittest import mock
from unittest.mock import call

from mslice.plotting.globalfiguremanager import GlobalFigureManager, set_category

# Class/function locations for mock
PLOTFIGUREMANAGER_CLS = 'mslice.plotting.plot_window.plot_figure_manager.PlotFigureManagerQT'
FORCE_METHOD_CALLS_TO_QAPP_THREAD = 'mslice.plotting.plot_window.plot_figure_manager.force_method_calls_to_qapp_thread'


@mock.patch(PLOTFIGUREMANAGER_CLS, autospec=True)
class CurrentFigureTest(unittest.TestCase):
    # In the case of more plot figures being created than expected then a StopIteration Exception will be raised
    # by the mock module. The number of expected plots is defined by the list at the beginning of each unit test

    def setUp(self):
        GlobalFigureManager.reset()
        # The getattr_static call in the real implementation does not play well
        # with mocks
        self.mock_force_qapp = mock.patch(
            FORCE_METHOD_CALLS_TO_QAPP_THREAD).start()
        # make it a noop
        self.mock_force_qapp.side_effect = lambda arg: arg

    def tearDown(self):
        self.mock_force_qapp.stop()

    def test_create_single_unclassified_plot_success(self, mock_figure_class):
        mock_figures = [mock.Mock()]
        mock_figure_class.side_effect = mock_figures

        GlobalFigureManager.get_figure_number()
        self.assertTrue(1 in GlobalFigureManager.all_figure_numbers()
                        )  #Check that a new figure with number=1 was created
        self.assertRaises(KeyError, GlobalFigureManager.get_category,
                          1)  #Check that figure has no category
        self.assertTrue(
            GlobalFigureManager.get_active_figure() == mock_figures[0])

    def test_create_multiple_unclassified_figures(self, mock_figure_class):
        """Test that n calls to GlobalFigureManager create n unclassified _figures numbered 1 to n """
        n = 10  # number of unclassfied _figures to be created
        mock_figures = [mock.Mock() for i in range(n)]
        mock_figure_class.side_effect = mock_figures

        for i in range(n):
            GlobalFigureManager.get_figure_number()  # Create a new figure
        for i in range(1, n + 1):
            self.assertTrue(i in GlobalFigureManager.all_figure_numbers(
            ))  #Check that a new figure with number=i was created
            self.assertRaises(KeyError, GlobalFigureManager.get_category,
                              i)  #Check that figure has no category

    def test_create_single_categorised_figure(self, mock_figure_class):
        mock_figures = [mock.Mock()]
        mock_figure_class.side_effect = mock_figures
        category = '1d'
        # The following line is equivalent to applying the decorator setcategory with the parameter category
        # to function GlobalFigureManager.get_active_figure
        categorised_get_active_figure = set_category(category)(
            GlobalFigureManager.get_active_figure)
        fig = categorised_get_active_figure()
        # Assert Figure object came from right place
        self.assertTrue(fig == mock_figures[0])
        self.assertTrue(GlobalFigureManager.get_category(1) == category)
        # Check that it is set as the active figure
        self.assertTrue(GlobalFigureManager.get_active_figure() == mock_figures[0])

    def test_create_categorised_figure_then_uncategorised_figure(
            self, mock_figure_class):
        mock_figures = [mock.Mock(), mock.Mock()]
        mock_figure_class.side_effect = mock_figures
        category = '1d'
        categorised_get_active_figure = set_category(category)(
            GlobalFigureManager.get_active_figure)

        fig1 = categorised_get_active_figure()
        fig2 = GlobalFigureManager.get_figure_number()
        fig1_number = GlobalFigureManager.number_of_figure(fig1)
        fig2_number = GlobalFigureManager.number_of_figure(fig2)
        self.assertTrue(GlobalFigureManager.get_active_figure() == fig2)
        self.assertTrue(
            GlobalFigureManager.get_category(fig1_number) == category)
        self.assertRaises(KeyError, GlobalFigureManager.get_category,
                          fig2_number)
        self.assertTrue(fig1_number == 1 and fig2_number == 2)

    def test_category_switching(self, mock_figure_class):
        mock_figures = [mock.Mock(), mock.Mock()]
        mock_figure_class.side_effect = mock_figures
        cat1 = '1d'
        cat2 = '2d'
        cat1_get_active_figure = set_category(cat1)(
            GlobalFigureManager.get_active_figure)
        cat2_get_active_figure = set_category(cat2)(
            GlobalFigureManager.get_active_figure)
        # test is an arbitrary method just to make sure the correct figures are returned
        cat1_get_active_figure().test(1)
        cat2_get_active_figure().test(2)
        cat1_get_active_figure().test(3)

        mock_figures[0].test.assert_has_calls([call(1), call(3)])
        mock_figures[1].test.assert_has_calls([call(2)])
        self.assertTrue(GlobalFigureManager._active_figure == 1)

    def test_close_only_window(self, mock_figure_class):
        mock_figures = [mock.Mock(), mock.Mock()]
        mock_figure_class.side_effect = mock_figures
        # Get a figure
        fig1 = GlobalFigureManager.get_active_figure()
        # check that getting the active window doesnt bring up a new one
        self.assertTrue(GlobalFigureManager.get_active_figure() == fig1)
        GlobalFigureManager.figure_closed(1)
        fig2 = GlobalFigureManager.get_active_figure()

        self.assertTrue(fig1 == mock_figures[0])
        self.assertTrue(fig2 == mock_figures[1])

    def test_close_non_existant_window_fail(self, mock_figure_class):
        mock_figures = [mock.Mock()]
        mock_figure_class.side_effect = mock_figures
        # Get a figure
        fig1 = GlobalFigureManager.get_active_figure()
        # check that getting the active window doesnt bring up a new one
        self.assertTrue(GlobalFigureManager.get_active_figure() == fig1)
        self.assertRaises(KeyError, GlobalFigureManager.figure_closed, 2)
        fig2 = GlobalFigureManager.get_active_figure()

        self.assertTrue(fig1 == mock_figures[0])
        self.assertTrue(fig2 == mock_figures[0])

    def test_destroy_all(self, mock_figure_class):
        mock_figures = [mock.Mock(), mock.Mock()]
        mock_figure_class.side_effect = mock_figures
        # Get first figure
        fig1 = GlobalFigureManager.get_active_figure()
        self.assertTrue(GlobalFigureManager.get_active_figure() == fig1)
        # Get second figure
        fig2 = GlobalFigureManager.get_active_figure()
        self.assertTrue(GlobalFigureManager.get_active_figure() == fig2)
        GlobalFigureManager.destroy_all()
        
        # check that both figures were destroyed
        self.assertRaises(KeyError, GlobalFigureManager.figure_closed, 1)
        self.assertRaises(KeyError, GlobalFigureManager.figure_closed, 2)

    def test_categorizing_of_uncategorized_plot(self, mock_figure_class):
        mock_figures = [mock.Mock(), mock.Mock(), mock.Mock()]
        fig1_mock_manager = mock.Mock()
        # This manager is used to compare the relative order of calls of two differebc functions
        fig1_mock_manager.attach_mock(mock_figures[0].flag_as_kept,
                                      'fig1_kept')
        fig1_mock_manager.attach_mock(mock_figures[0].flag_as_current,
                                      'fig1_current')
        mock_figure_class.side_effect = mock_figures
        cat1 = '1d'
        cat2 = '2d'
        cat1_get_active_figure = set_category(cat1)(
            GlobalFigureManager.get_active_figure)
        cat2_get_active_figure = set_category(cat2)(
            GlobalFigureManager.get_active_figure)

        # test is an arbitrary method just to make sure the correct figures are returned

        cat1_get_active_figure().test(1)  # create a figure of category 1
        cat2_get_active_figure().test(2)  # create a figure of category 2
        GlobalFigureManager.set_figure_as_kept(
            2)  # now there is no active figure

        GlobalFigureManager.get_active_figure().test(
            3)  # create an uncategorized figure
        cat1_get_active_figure().test(
            4
        )  # the previously uncategorized figure should now be categorized as cat1

        mock_figures[0].test.assert_has_calls([call(1)])
        mock_figures[1].test.assert_has_calls([call(2)])
        mock_figures[2].test.assert_has_calls([call(3), call(4)])

        # assert final status of fig1 is kept
        self.assertTrue(
            fig1_mock_manager.mock_calls[-1] == call.fig1_kept())
        self.assertTrue(GlobalFigureManager._active_figure == 3)

    def test_make_current_with_single_category(self, mock_figure_class):
        mock_figures = [mock.Mock(), mock.Mock()]
        # These manager is used to compare the relative order of calls of two different functions
        mock_managers = [mock.Mock(), mock.Mock()]

        for i in range(len(mock_figures)):
            mock_managers[i].attach_mock(mock_figures[i].flag_as_kept,
                                         'fig_kept')
            mock_managers[i].attach_mock(mock_figures[i].flag_as_current,
                                         'fig_current')
        mock_figure_class.side_effect = mock_figures
        cat1 = '1d'
        cat1_get_active_figure = set_category(cat1)(
            GlobalFigureManager.get_active_figure)
        # test is an arbitrary method just to make sure the correct figures are returned

        cat1_get_active_figure().test(1)  # create a figure of category 1
        GlobalFigureManager.set_figure_as_kept(
            1)  # now there is no active figure
        cat1_get_active_figure().test(
            2)  # this command should go to a new figure

        # assert fig1 now displays kept
        self.assertTrue(mock_managers[0].mock_calls[-1] == call.fig_kept())
        # assert fig2 now displays current
        self.assertTrue(mock_managers[1].mock_calls[-1] == call.fig_current())

        GlobalFigureManager.set_figure_as_current(1)
        self.assertTrue(mock_managers[0].mock_calls[-1] == call.fig_current())
        self.assertTrue(mock_managers[1].mock_calls[-1] == call.fig_kept())

        cat1_get_active_figure().test(3)  # This should go to fig1
        GlobalFigureManager.get_active_figure().test(4) # so should this

        mock_figures[0].test.assert_has_calls([call(1), call(3), call(4)])
        mock_figures[1].test.assert_has_calls([call(2)])

        self.assertTrue(GlobalFigureManager._active_figure == 1)
