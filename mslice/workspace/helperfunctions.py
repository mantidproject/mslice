import pickle
import codecs

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
