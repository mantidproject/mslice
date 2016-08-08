from mock import call
import mock
import unittest
from plotting.FigureManager import FigureManager, activate_category
import plotting.FigureManager


class FigureManagerTest(unittest.TestCase):
    # In the case of more plot figures being created than expected then a StopIteration Exception will be raised
    # by the mock module. The number of expected plots is defined by the list at the beginning of each unit test

    def setUp(self):
        FigureManager.reset()

    def test_constructor_fails(self):
        """The figure manager class is a singleton static class and any attempts to instantiate it should fail"""
        self.assertRaises(Exception, FigureManager)

    @mock.patch('plotting.FigureManager.PlotFigure')
    def test_create_single_unclassified_plot_success(self, mock_figure_class):
        mock_figures = [mock.Mock()]
        mock_figure_class.side_effect = mock_figures

        FigureManager.get_figure_number()
        self.assert_(1 in FigureManager.all_figure_numbers()) #Check that a new figure with number=1 was created
        self.assertRaises(KeyError,FigureManager.get_category, 1) #Check that figure has no category
        self.assert_(FigureManager.get_active_figure() ==mock_figures[0]) # Check that it is set as the active figure

    @mock.patch('plotting.FigureManager.PlotFigure')
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

    @mock.patch('plotting.FigureManager.PlotFigure')
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

    @mock.patch('plotting.FigureManager.PlotFigure')
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
        self.assert_(FigureManager)
        self.assert_(FigureManager.get_category(fig1_number) == category)
        self.assertRaises(KeyError,FigureManager.get_category,fig2_number)
        self.assert_( fig1_number ==1 and fig2_number == 2)

    @mock.patch('plotting.FigureManager.PlotFigure')
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

