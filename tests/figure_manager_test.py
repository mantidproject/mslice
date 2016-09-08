from mock import call
import mock
import unittest
import plotting
#TODO use method to specify figure class when that is decided
plotting.get_figure_class = lambda : None #This stops the Figure manager from importing the windows which import pyqt
from plotting.figuremanager import FigureManager
from plotting.figuremanager import activate_category


class FigureManagerTest(unittest.TestCase):
    # In the case of more plot figures being created than expected then a StopIteration Exception will be raised
    # by the mock module. The number of expected plots is defined by the list at the beginning of each unit test

    def setUp(self):
        FigureManager.reset()

    def test_constructor_fails(self):
        """The figure manager class is a singleton static class and any attempts to instantiate it should fail"""
        self.assertRaises(Exception, FigureManager)

    plot_window_class = 'plotting.figuremanager.PlotFigureManager'

    @mock.patch(plot_window_class)
    def test_create_single_unclassified_plot_success(self, mock_figure_class):
        mock_figures = [mock.Mock()]
        mock_figure_class.side_effect = mock_figures

        FigureManager.get_figure_number()
        self.assert_(1 in FigureManager.all_figure_numbers()) #Check that a new figure with number=1 was created
        self.assertRaises(KeyError,FigureManager.get_category, 1) #Check that figure has no category
        self.assert_(FigureManager.get_active_figure() == mock_figures[0]) # Check that it is set as the active figure

    @mock.patch(plot_window_class)
    def test_create_multiple_unclassified_figures(self, mock_figure_class):
        """Test that n calls to figureManager create n unclassified _figures numbered 1 to n """
        n = 10  # number of unclassfied _figures to be created
        mock_figures = [mock.Mock() for i in range(n)]
        mock_figure_class.side_effect = mock_figures

        for i in range(n):
            FigureManager.get_figure_number() # Create a new figure
        for i in range(1, n+1):
            self.assert_(i in FigureManager.all_figure_numbers()) #Check that a new figure with number=i was created
            self.assertRaises(KeyError,FigureManager.get_category, i) #Check that figure has no category

    @mock.patch(plot_window_class)
    def test_create_single_categorised_figure(self,mock_figure_class):
        mock_figures = [mock.Mock()]
        mock_figure_class.side_effect = mock_figures
        category = '1d'
        # The following line is equivalent to applying the decorator activate_category with the parameter category
        # to function FigureManager.get_active_figure
        categorised_get_active_figure = activate_category(category)(FigureManager.get_active_figure)
        fig = categorised_get_active_figure()
        self.assert_(fig == mock_figures[0]) # Assert Figure object came from right place
        self.assert_(FigureManager.get_category(1) == category)
        self.assert_(FigureManager.get_active_figure() == mock_figures[0]) # Check that it is set as the active figure

    @mock.patch(plot_window_class)
    def test_create_categorised_figure_then_uncategorised_figure(self,mock_figure_class):
        mock_figures = [mock.Mock(), mock.Mock()]
        mock_figure_class.side_effect = mock_figures
        category = '1d'
        categorised_get_active_figure = activate_category(category)(FigureManager.get_active_figure)

        fig1 = categorised_get_active_figure()
        fig2 = FigureManager.get_figure_number()
        fig1_number = FigureManager.number_of_figure(fig1)
        fig2_number = FigureManager.number_of_figure(fig2)
        self.assert_(FigureManager.get_active_figure() == fig2)
        self.assert_(FigureManager.get_category(fig1_number) == category)
        self.assertRaises(KeyError,FigureManager.get_category,fig2_number)
        self.assert_( fig1_number == 1 and fig2_number == 2)

    @mock.patch(plot_window_class)
    def test_category_switching(self,mock_figure_class):
        mock_figures = [mock.Mock(),mock.Mock()]
        mock_figure_class.side_effect = mock_figures
        cat1 = '1d'
        cat2 = '2d'
        cat1_get_active_figure = activate_category(cat1)(FigureManager.get_active_figure)
        cat2_get_active_figure = activate_category(cat2)(FigureManager.get_active_figure)
        # test is an arbitrary method just to make sure the correct figures are returned
        cat1_get_active_figure().test(1)
        cat2_get_active_figure().test(2)
        cat1_get_active_figure().test(3)

        mock_figures[0].test.assert_has_calls([call(1), call(3)])
        mock_figures[1].test.assert_has_calls([call(2)])
        self.assert_(FigureManager._active_figure == 1)

    @mock.patch(plot_window_class)
    def test_close_only_window(self,mock_figure_class):
        mock_figures = [mock.Mock(), mock.Mock()]
        mock_figure_class.side_effect = mock_figures
        # Get a figure
        fig1 = FigureManager.get_active_figure()
        # check that getting the active window doesnt bring up a new one
        self.assert_(FigureManager.get_active_figure() == fig1)
        FigureManager.figure_closed(1)
        fig2 = FigureManager.get_active_figure()

        self.assert_(fig1 == mock_figures[0])
        self.assert_(fig2 == mock_figures[1])

    @mock.patch(plot_window_class)
    def test_close_non_existant_window_fail(self,mock_figure_class):
        mock_figures = [mock.Mock()]
        mock_figure_class.side_effect = mock_figures
        # Get a figure
        fig1 = FigureManager.get_active_figure()
        # check that getting the active window doesnt bring up a new one
        self.assert_(FigureManager.get_active_figure() == fig1)
        self.assertRaises(KeyError, FigureManager.figure_closed, 2)
        fig2 = FigureManager.get_active_figure()

        self.assert_(fig1 == mock_figures[0])
        self.assert_(fig2 == mock_figures[0])

    @mock.patch(plot_window_class)
    def test_categorizing_of_uncategorized_plot(self, mock_figure_class):
        mock_figures = [mock.Mock(), mock.Mock(), mock.Mock()]
        fig1_mock_manger = mock.Mock()
        # This manager is used to compare the relative order of calls of two differebc functions
        fig1_mock_manger.attach_mock(mock_figures[0].set_as_kept, 'fig1_kept')
        fig1_mock_manger.attach_mock(mock_figures[0].set_as_current, 'fig1_current')
        mock_figure_class.side_effect = mock_figures
        cat1 = '1d'
        cat2 = '2d'
        cat1_get_active_figure = activate_category(cat1)(FigureManager.get_active_figure)
        cat2_get_active_figure = activate_category(cat2)(FigureManager.get_active_figure)

        # test is an arbitrary method just to make sure the correct figures are returned

        cat1_get_active_figure().test(1)  # create a figure of category 1
        cat2_get_active_figure().test(2)  # create a figure of category 2
        FigureManager.set_figure_as_kept(2) # now there is no active figure

        FigureManager.get_active_figure().test(3) # create an uncategorized figure
        cat1_get_active_figure().test(4) # the previously uncategorized figure should now be categorized as cat1

        mock_figures[0].test.assert_has_calls([call(1)])
        mock_figures[1].test.assert_has_calls([call(2)])
        mock_figures[2].test.assert_has_calls([call(3), call(4)])

        self.assert_(fig1_mock_manger.mock_calls[-1] == call.fig1_kept())  # assert final status of fig1 is kept
        self.assert_(FigureManager._active_figure == 3)

    @mock.patch(plot_window_class)
    def test_make_current_with_single_category(self, mock_figure_class):
        mock_figures = [mock.Mock(), mock.Mock()]
        # These manager is used to compare the relative order of calls of two different functions
        mock_managers = [mock.Mock(), mock.Mock()]

        for i in range(len(mock_figures)):
            mock_managers[i].attach_mock(mock_figures[i].set_as_kept, 'fig_kept')
            mock_managers[i].attach_mock(mock_figures[i].set_as_current, 'fig_current')
        mock_figure_class.side_effect = mock_figures
        cat1 = '1d'
        cat1_get_active_figure = activate_category(cat1)(FigureManager.get_active_figure)
        # test is an arbitrary method just to make sure the correct figures are returned

        cat1_get_active_figure().test(1)  # create a figure of category 1
        FigureManager.set_figure_as_kept(1)  # now there is no active figure
        cat1_get_active_figure().test(2)  # this command should go to a new figure

        self.assert_(mock_managers[0].mock_calls[-1] == call.fig_kept())     # assert fig1 now displays kept
        self.assert_(mock_managers[1].mock_calls[-1] == call.fig_current())  # assert fig2 now displays current

        FigureManager.set_figure_as_current(1)
        self.assert_(mock_managers[0].mock_calls[-1] == call.fig_current())     # assert fig1 now displays current
        self.assert_(mock_managers[1].mock_calls[-1] == call.fig_kept())        # assert fig2 now displays kept

        cat1_get_active_figure().test(3)                # This should go to fig1
        FigureManager.get_active_figure().test(4)       # This should go to fig1 as well

        mock_figures[0].test.assert_has_calls([call(1), call(3), call(4)])
        mock_figures[1].test.assert_has_calls([call(2)])

        self.assert_(FigureManager._active_figure == 1)

