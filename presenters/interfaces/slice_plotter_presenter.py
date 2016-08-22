class SlicePlotterPresenterInterface(object):
    def __init__(self, slice_view, slice_plotter):
        raise Exception("This interface class must not be instantiated")

    def register_master(self, main_view):
        raise NotImplementedError("This method must be overriden in implementation")

    def notify(self,command):
        raise NotImplementedError("This method must be overriden in implementation")

    def workspace_selection_changed(self):
        raise NotImplementedError("This method must be overriden in implementation")