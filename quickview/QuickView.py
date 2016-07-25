#TODO MOVE TO TOOLS
import re
from ui_mock import GetInputFromUser,display_message
from PyQt4 import QtGui


class QuickView(QtGui.QWidget):
    """QuickView is a base class for dynamically generated view for use while developing MVP applications.

    The constructor takes a'command' class or object. The command class should contain only constants that will be sent to Presenter via
    Presenter.notify(). Function are redirected upon call time to the handler
    It is your responsibility to Create a QApplication before and call QApplication.exec_() after creating this view
    self._presenter should be supplied in constructor in child class """

    def __init__(self,commands):
        super(QtGui.QWidget,self).__init__()
        #Supply two default _handlers
        #Starting variable names with _ breaks the whole tool due to __getattr__ override
        self._handlers = {re.compile("get.*"): self._get, re.compile('populate.*'): self._display}
        self._default_handler = self._display
        self._commands = commands
        self._setupCommandCenter()


    def __getattr__(self, item):
        class_methods = ['get_presenter','_default_handler']
        if item.startswith('_') or item in dir(QuickView) or item in class_methods:
            try:
                return object.__getattribute__(self,item)
            except AttributeError:
                print item, 'Call handle failed'
                raise AttributeError('Attribute %s not found'%item)

        self._title = item
        for regex,function in self._handlers.items():
            if regex.match(item):
                return function
        return self._default_handler

    def add_handler(self,regex,handler_function):
        """Add a function handler_function to handle all function calls that match the regex string supplied"""
        self._handlers[re.compile(regex)] = handler_function

    def set_default_handler(self,handler):
        """Set Function to handle calls that match none of the available regular expressions"""
        self._default_handler = handler

    def _get(self):
        getter = GetInputFromUser(self._title)
        return getter._data

    def _supply_filler(self,length = 2):
        filler = []
        lorem_ipsum = ('lorem','ipsum')
        for i in range(length):
            filler.append(lorem_ipsum[i%2])

    def _display(self, *args):
        display_message(self._title, *args)

    def _setupCommandCenter(self):

        self._window = QtGui.QWidget()
        self._layout = QtGui.QGridLayout()
        self._window.setWindowTitle('Send Notifications')
        self._buttons = {}
        for i,command in enumerate(dir(self._commands)):
            if command[:2] == '__':
                continue
            self._buttons[command] = getattr(self._commands, command)
            btn = QtGui.QPushButton(parent=self._window, text=command)
            self._layout.addWidget(btn, i % 4, i // 4)
            btn.clicked.connect(self._notify_presenter)
        self._window.setLayout(self._layout)
        self._window.show()

    def _notify_presenter(self):
        sender = self.sender()
        self._presenter.notify(self._buttons[str(sender.text())])

if __name__ == '__main__':

    app = QtGui.QApplication([])
    m = QuickView()
    m.get_data_from_user()
    m.say_hi('A',1,(1,1,1,1,1,1,1,1))

