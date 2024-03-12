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
        if hasattr(workspace, str(ws)) and ws is not None and WorkspaceNameHandler(ws.name()).assert_name(is_hidden_from_mslice=True):
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

    def _add_prefix(self, prefix) -> str:
        self.ws_name = prefix + self.ws_name

    def _add_sufix(self, sufix) -> str:
        self.ws_name = self.ws_name + sufix

    def get_name(
            self,
            scaling_factor=None,
            scaled=False,
            subtracted=False,
            summed=False,
            rebosed=False,
            combined=False,
            merged=False,
            hide_from_mslice=False,
            hide_from_ADS=False,
            mslice_signature=False,
            temporary_signature=False,
            make_ws_visible_in_ADS=False,
            make_ws_visible_in_mslice=False):

        singular_arguments = [scaled, subtracted, summed, rebosed, combined, merged]
        assert sum(singular_arguments) <= 1, "Two or more incompatible arguments were set to True."

        if scaled:
            assert scaling_factor is not None, "Scaling factor should be provided to build name of workspace"
            self._add_sufix("_ssf_" + f"{scaling_factor:.2f}".replace('.', '_'))

        if subtracted:
            assert scaling_factor is not None, "Scaling factor should be provided to build name of workspace"
            self._add_sufix("_minus_ssf_" + f"{scaling_factor:.2f}".replace('.', '_'))

        if summed:
            self._add_sufix("_sum")

        if rebosed:
            self._add_sufix("_bosed")

        if combined:
            self._add_sufix("_combined")

        if merged:
            self._add_sufix("_merged")

        if hide_from_mslice:
            self._add_sufix("_HIDDEN")

        if temporary_signature:
            self._add_prefix("TMP")

        if mslice_signature:
            self._add_prefix("MSL")

        if hide_from_ADS:
            self._add_prefix("__")

        if make_ws_visible_in_ADS:
            self.ws_name = self.ws_name.replace('__MSL', '').replace('__', '')

        if make_ws_visible_in_mslice:
            self.ws_name = self.ws_name.replace('_HIDDEN', '')

        return self.ws_name

    def assert_name(
            self,
            is_hidden_from_mslice=False,
            is_hidden_from_ADS=False,
            has_mslice_signature=False):

        result_flag = True

        if is_hidden_from_mslice:
            result_flag = result_flag and (self.ws_name.endswith("_HIDDEN"))

        if is_hidden_from_ADS:
            result_flag = result_flag and (self.ws_name.startswith("__"))

        if has_mslice_signature:
            result_flag = result_flag and (self.ws_name.startswith("__MSL") or self.ws_name.startswith("MSL"))

        return result_flag
