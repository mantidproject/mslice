from getter_ui import Ui_Dialog
from PyQt4 import QtGui

const = -1213124234534
class Getter(Ui_Dialog,QtGui.QDialog):
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
        self.data = const
        self.exec_()

    def is_done(self):
        return self.data != const

    def on_click(self):
        sender = self.sender()
        text = str(self.lineEdit.text())
        if sender == self.btnInt:
            self.data =  int(text)
        if sender == self.btnString:
            self.data =  text
        if sender == self.btnFloat:
            self.data = float(text)
        if sender == self.btnBool:
            self.data = bool(int(text))
        if sender ==self.lineEdit:
            self.data =  eval(text)
        if sender == self.btnFilepath:
            self.data = str(QtGui.QFileDialog.getOpenFileName())
        self.accept()
def displayMessage(title,*args):
    args_and_types = [(str(arg)+' '+str(type(arg)) ) for arg in args]
    messageBox = QtGui.QMessageBox()
    messageBox.setWindowTitle(title)
    messageBox.setText('\n'.join(args_and_types))
    messageBox.exec_()
