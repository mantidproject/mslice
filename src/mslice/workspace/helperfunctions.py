import pickle
import codecs
from mantid.simpleapi import DeleteWorkspace, RenameWorkspace


def _attribute_from_string(ws, comstr):
    if comstr:
        try:
            attrdict = pickle.loads(codecs.decode(comstr.encode(), 'base64'))
        except ValueError:
            pass
        else:
            for (k, v) in list(attrdict.items()):
                if hasattr(ws, k):
                    setattr(ws, k, v)


def attribute_from_comment(ws, raw_ws):
    try:
        comstr = raw_ws.getComment()
    except AttributeError:
        return
    else:
        _attribute_from_string(ws, comstr)


def attribute_from_log(ws, raw_ws):
    try:
        runinfo = raw_ws.run()
    except AttributeError:
        try:
            runinfo = raw_ws.getExperimentInfo(0).run()
        except ValueError:
            attribute_from_comment(ws, raw_ws)
            return
    try:
        comstr = runinfo.getProperty('MSlice').value
    except RuntimeError:
        return
    else:
        _attribute_from_string(ws, comstr)


def attribute_to_comment(attrdict, raw_ws, append=False):
    if append:
        try:
            comstr = raw_ws.getComment()
        except AttributeError:
            pass
        else:
            prevdict = pickle.loads(codecs.decode(comstr.encode(), 'base64'))
            for (k, v) in list(prevdict.items()):
                if k not in attrdict:
                    attrdict[k] = v
    try:
        raw_ws.setComment(str(codecs.encode(pickle.dumps(attrdict), 'base64').decode()))
    except AttributeError:
        pass


def attribute_to_log(attrdict, raw_ws, append=False):
    try:
        runinfo = raw_ws.run()
    except AttributeError:
        try:
            runinfo = raw_ws.getExperimentInfo(0).run()
        except ValueError:
            attribute_to_comment(attrdict, raw_ws, append)
            return
    if not append:
        runinfo.addProperty('MSlice', str(codecs.encode(pickle.dumps(attrdict), 'base64').decode()), True)
    else:
        try:
            comstr = runinfo.getProperty('MSlice').value
        except RuntimeError:
            pass
        else:
            prevdict = pickle.loads(codecs.decode(comstr.encode(), 'base64'))
            for (k, v) in list(prevdict.items()):
                if k not in attrdict:
                    attrdict[k] = v
        runinfo.addProperty('MSlice', str(codecs.encode(pickle.dumps(attrdict), 'base64').decode()), True)


def delete_workspace(workspace, ws):
    try:
        if hasattr(workspace, str(ws)) and ws is not None and WorkspaceNameHandler(ws.name()).isHiddenFromMsl():
            DeleteWorkspace(ws)
            ws = None
    except RuntimeError:
        # On exit the workspace can get deleted before __del__ is called
        # where you receive a RuntimeError: Variable invalidated, data has been deleted.
        # error
        pass


def rename_workspace(old_name: str, new_name: str) -> None:
    """Rename a workspace stored in the ADS."""
    if new_name != old_name:
        RenameWorkspace(InputWorkspace=old_name, OutputWorkspace=new_name)


class WrapWorkspaceAttribute(object):

    def __init__(self, workspace):
        self.workspace = workspace if (hasattr(workspace, 'save_attributes')
                                       and hasattr(workspace, 'remove_saved_attributes')) else None

    def __enter__(self):
        if self.workspace:
            self.workspace.save_attributes()
        return self.workspace

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.workspace:
            self.workspace.remove_saved_attributes()
        return True


class WorkspaceNameHandler:

    def __init__(self, ws_name: str):
        self.ws_name = ws_name

    def add_prefix(self, prefix) -> str:
        return prefix + self.ws_name

    def add_sufix(self, sufix) -> str:
        return self.ws_name + sufix

    def scaled(self, scaling_factor: float) -> str:
        return self.add_sufix("_ssf_" + f"{scaling_factor:.2f}".replace('.', '_'))

    def subtracted(self, scaling_factor: float) -> str:
        return self.add_sufix("_minus_ssf_" + f"{scaling_factor:.2f}".replace('.', '_'))

    def summed(self) -> str:
        return self.add_sufix("_sum")

    def rebosed(self) -> str:
        return self.add_sufix('_bosed')

    def combined(self) -> str:
        return self.add_sufix('_combined')

    def merged(self) -> str:
        return self.add_sufix('_merged')
    
    def isHiddenFromMsl(self) -> bool:
        return '_HIDDEN' in self.ws_name

    def hideFromMsl(self) -> str:
        return self.add_sufix('_HIDDEN')

    def removeHideFlags(self) -> str:
        return self.ws_name.replace('__MSL', '').replace('_HIDDEN', '')

    def hideMslInAds(self) -> str:
        return self.add_prefix('__MSL')

    def isMslHiddenInAds(self) -> bool:
        return '__MSL' in self.ws_name

    def hideTmpMslInAds(self) -> str:
        return self.add_prefix('__MSLTMP')

    def hideInAds(self) -> str:
        return self.add_prefix('__')

    def isHiddenInAds(self) -> bool:
        return self.ws_name.startswith('__')

    def makeVisibleInAds(self) -> str:
        return self.ws_name.lstrip('__')
