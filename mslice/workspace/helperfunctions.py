import pickle
import codecs

def attribute_from_comment(ws, raw_ws):
    try:
        comstr = raw_ws.getComment()
    except AttributeError:
        comstr = ''
    if comstr:
        try:
            attrdict = pickle.loads(codecs.decode(comstr.encode(), 'base64'))
        except ValueError:
            pass
        else:
            raw_ws.setComment(attrdict.pop('comment', ''))
            for (k, v) in list(attrdict.items()):
                if hasattr(ws, k):
                    setattr(ws, k, v)

def attribute_to_comment(attrdict, raw_ws):
    try:
        comstr = raw_ws.getComment()
        if 'comment' not in attrdict.keys() and comstr:
            attrdict['comment'] = comstr
        raw_ws.setComment(str(codecs.encode(pickle.dumps(attrdict), 'base64').decode()))
    except AttributeError:
        pass

class WrapWorkspaceAttribute(object):

    def __init__(self, workspace):
        self.workspace = workspace if (hasattr(workspace, 'save_attributes')
                                       and hasattr(workspace, 'remove_comment_attributes')) else None

    def __enter__(self):
        if self.workspace:
            self.workspace.save_attributes()
        return self.workspace

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.workspace:
            self.workspace.remove_comment_attributes()
        return True
