# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\OMi32458\Workspace\MSlice3.0\QuickView\ui_mock_getter.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName(_fromUtf8("Dialog"))
        Dialog.resize(390, 132)
        self.gridLayout = QtGui.QGridLayout(Dialog)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.lineEdit = QtGui.QLineEdit(Dialog)
        self.lineEdit.setObjectName(_fromUtf8("lineEdit"))
        self.gridLayout.addWidget(self.lineEdit, 0, 0, 1, 4)
        self.btnString = QtGui.QPushButton(Dialog)
        self.btnString.setObjectName(_fromUtf8("btnString"))
        self.gridLayout.addWidget(self.btnString, 1, 0, 1, 1)
        self.btnInt = QtGui.QPushButton(Dialog)
        self.btnInt.setObjectName(_fromUtf8("btnInt"))
        self.gridLayout.addWidget(self.btnInt, 1, 1, 1, 1)
        self.btnFloat = QtGui.QPushButton(Dialog)
        self.btnFloat.setObjectName(_fromUtf8("btnFloat"))
        self.gridLayout.addWidget(self.btnFloat, 1, 2, 1, 1)
        self.btnBool = QtGui.QPushButton(Dialog)
        self.btnBool.setObjectName(_fromUtf8("btnBool"))
        self.gridLayout.addWidget(self.btnBool, 1, 3, 1, 1)
        self.btnFilepath = QtGui.QPushButton(Dialog)
        self.btnFilepath.setObjectName(_fromUtf8("btnFilepath"))
        self.gridLayout.addWidget(self.btnFilepath, 2, 0, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(_translate("Dialog", "Dialog", None))
        self.btnString.setText(_translate("Dialog", "String", None))
        self.btnInt.setText(_translate("Dialog", "Int", None))
        self.btnFloat.setText(_translate("Dialog", "Float", None))
        self.btnBool.setText(_translate("Dialog", "Bool", None))
        self.btnFilepath.setText(_translate("Dialog", "FilePath", None))

