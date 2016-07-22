from getter_ui import Ui_Dialog
from PyQt4 import QtGui


class GetInputFromUser(Ui_Dialog, QtGui.QDialog):
    """This is a generic _window to get input from user.

     Type input in the input box and click a button to return the contents of the input field explicitly typecasted to
     the type shown in the button. Alternatively hit enter to return the evaluation of the python expression
      in the input field. Clicking on filepath shows a graphical interface to return a string containing to a
      filepath"""

    def __init__(self,title):
        super(QtGui.QDialog,self).__init__()
        self.setupUi(self)
        self.setWindowTitle(title)
        self.btnInt.clicked.connect(self.on_click)
        self.btnFloat.clicked.connect(self.on_click)
        self.btnFilepath.clicked.connect(self.on_click)
        self.btnBool.clicked.connect(self.on_click)
        self.btnString.clicked.connect(self.on_click)
        self.lineEdit.returnPressed.connect(self.on_click)
        self._done = False
        self._data = None
        self.exec_()

    def is_done(self):
        return self._done

    def get_data(self):
        return self._data

    def on_click(self):
        """If user hits enter the data is set to be the python evaluation of the expression in the input box
        If the user clicks a button the input is explicitly typecasted to the type shown on the button label"""
        sender = self.sender()
        text = str(self.lineEdit.text())
        if sender == self.btnInt:
            self._data = int(text)
        if sender == self.btnString:
            self._data = text
        if sender == self.btnFloat:
            self._data = float(text)
        if sender == self.btnBool:
            self._data = bool(int(text))
        if sender == self.lineEdit:
            self._data = eval(text)
        if sender == self.btnFilepath:
            self._data = str(QtGui.QFileDialog.getSaveFileName())
        self._done = True
        self.accept()


def display_message(title, *args):
    args_and_types = [(str(arg) + ' ' + str(type(arg))) for arg in args]
    if not args_and_types:
        args_and_types=['']
    #Pad message box so whole title is visible
    padding = len(title) * '   '
    args_and_types[0] += padding
    messageBox = QtGui.QMessageBox()
    messageBox.setWindowTitle(title)
    messageBox.setText('\n'.join(args_and_types))
    messageBox.setGeometry(100,100,4000,200)
    messageBox.exec_()
