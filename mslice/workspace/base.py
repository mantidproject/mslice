import abc
import pickle
import codecs
from six import add_metaclass

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


@add_metaclass(abc.ABCMeta)
class WorkspaceBase(object):

    @abc.abstractmethod
    def get_coordinates(self):
        return

    @abc.abstractmethod
    def get_signal(self):
        return

    @abc.abstractmethod
    def get_error(self):
        return

    @abc.abstractmethod
    def get_variance(self):
        return

    @abc.abstractmethod
    def rewrap(self, ws):
        return

    @abc.abstractmethod
    def __add__(self, other):
        return

    @abc.abstractmethod
    def __sub__(self, other):
        return

    @abc.abstractmethod
    def __mul__(self, other):
        return

    @abc.abstractmethod
    def __truediv__(self, other):
        return

    @abc.abstractmethod
    def __pow__(self, other):
        return

    @abc.abstractmethod
    def __neg__(self):
        return
