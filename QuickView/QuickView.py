#TODO MOVE TO TOOLS
import re
from ui_mock import Getter,displayMessage
from PyQt4 import QtGui

class QuickView(QtGui.QWidget):
    def __init__(self,commands):
        """"Your responsibilty to start a QApplication and call QApplication.exec_() after creating this view"""
        print commands
        super(QtGui.QWidget,self).__init__()
        self.handlers = {re.compile("get.*"):self.get,re.compile('populate.*'):lambda *args:('lorem','ipsum')}
        self.default_handler = self.display
        self.commands = commands
        self.setupCommandCenter()

    def get_presenter(self):
        return self._presenter

    def __getattr__(self, item):
        print 'intercepted call of '+item
        if item in ('__methods__','__members__') or item[:2] == '__' or item in dir(self):
            return object.__getattribute__(self,item)
        self.title = item
        for regex,function in self.handlers.items():
            if regex.match(item):
                return function
        return self.default_handler

    def add_handler(self,regex,handler_function):
        """Add a function handler_function to handle all function calls that match the regex"""
        self.handlers[re.compile(regex)] = handler_function
    def set_default_handler(self,handler):
        """Set Function to handle calls that match none of the available regular expressions"""
        self.default_handler = handler
    def get(self):
        getter = Getter(self.title)
        return getter.data

    def display(self,*args):
        displayMessage(self.title,*args)

    def setupCommandCenter(self):

        self.window = QtGui.QWidget()
        self.layout = QtGui.QGridLayout()
        self.window.setWindowTitle('Send Notifications')
        self.buttons = {}
        for i,command in enumerate(dir(self.commands)):
            if command[:2]=='__':
                continue
            self.buttons[command] = getattr(self.commands,command)
            btn = QtGui.QPushButton(parent = self.window,text=command)
            self.layout.addWidget(btn,i%4,i//4)
            btn.clicked.connect(self.notify_presenter)
        self.window.setLayout(self.layout)
        self.window.show()

    def notify_presenter(self):
        sender = self.sender()
        self._presenter.notify(self.buttons[str(sender.text())])

if __name__ == '__main__':

    app = QtGui.QApplication([])
    m = QuickView()
    m.get_data_from_user()
    m.say_hi('A',1,(1,1,1,1,1,1,1,1))

