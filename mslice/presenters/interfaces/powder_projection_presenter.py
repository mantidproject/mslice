
class PowderProjectionPresenterInterface(object):

    def __init__(self, _powder_view, _projection_calculator):
        raise Exception("This interface must not be instantiated")

    def register_master(self, main_view):
        raise NotImplementedError("This method must be overriden in implementation")

    def notify(self, command):
        raise NotImplementedError("This method must be overriden in implementation")

    def set_workspace_provider(self, workspace_provider):
        raise NotImplementedError("This method must be overriden in implementation")
