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
        #Supply two default handlers
        self._handlers = {re.compile("get.*"):self._get, re.compile('populate.*'):self._display}
        self._default_handler = self._display
        self._commands = commands
        self._setupCommandCenter()

    def get_presenter(self):
        return self._presenter

    def __getattr__(self, item):
        print 'intercepted call of '+item
        if item.startswith('_') or item in dir(self):
            print 'released item',item
            try:
                return object.__getattribute__(self,item)
            except AttributeError:
                setattr(self,item,None)
                return getattr(self,item)
        self.title = item
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
        getter = GetInputFromUser(self.title)
        return getter._data

    def _supply_filler(self,length = 2):
        filler = []
        lorem_ipsum = ('lorem','ipsum')
        for i in range(length):
            filler.append(lorem_ipsum[i%2])

    def _display(self, *args):
        display_message(self.title, *args)

    def _setupCommandCenter(self):

        self.window = QtGui.QWidget()
        self.layout = QtGui.QGridLayout()
        self.window.setWindowTitle('Send Notifications')
        self.buttons = {}
        for i,command in enumerate(dir(self._commands)):
            if command[:2]=='__':
                continue
            self.buttons[command] = getattr(self._commands, command)
            btn = QtGui.QPushButton(parent = self.window,text=command)
            self.layout.addWidget(btn,i%4,i//4)
            btn.clicked.connect(self._notify_presenter)
        self.window.setLayout(self.layout)
        self.window.show()

    def _notify_presenter(self):
        sender = self.sender()
        self._presenter.notify(self.buttons[str(sender.text())])

if __name__ == '__main__':

    app = QtGui.QApplication([])
    m = QuickView()
    m.get_data_from_user()
    m.say_hi('A',1,(1,1,1,1,1,1,1,1))

