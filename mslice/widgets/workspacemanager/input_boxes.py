from mslice.util.qt import load_ui
from mslice.util.qt.QtWidgets import QDialog, QFormLayout, QLabel, QLineEdit, QPushButton
from mslice.util.qt.validator_helper import double_validator_without_separator


class SubtractInputBox(QDialog):

    def __init__(self, ws_list, parent=None):
        QDialog.__init__(self, parent)
        load_ui(__file__, 'subtract_input_box.ui', self)
        for i in range(ws_list.count()):
            item = ws_list.item(i).clone()
            self.listWidget.insertItem(i, item)
        self.listWidget.setCurrentRow(0)

    def user_input(self):
        background_ws = self.listWidget.currentItem().text()
        return background_ws, self.ssf.value()


class ScaleInputBox(QDialog):

    def __init__(self, is_bose=False, parent=None):
        super(ScaleInputBox, self).__init__(parent)
        self.setWindowTitle('Scale parameters')
        layout = QFormLayout(self)
        self.text1 = QLabel(self)
        self.text2 = QLabel(self)

        double_validator = double_validator_without_separator()
        self.edit1 = QLineEdit(self)
        self.edit1.setValidator(double_validator)
        self.edit1.setText('1.0')
        self.edit2 = QLineEdit(self)
        self.edit2.setValidator(double_validator)

        if is_bose:
            self.text1.setText('Current temperature (K)')
            self.text2.setText('Target temperature (K)')
            layout.addRow(self.text1, self.text2)
            layout.addRow(self.edit1, self.edit2)
        else:
            self.text1.setText('Scale Factor')
            layout.addRow(self.text1)
            layout.addRow(self.edit1)
            self.text2.hide()
            self.edit2.hide()
        okbtn = QPushButton('OK')
        okbtn.clicked.connect(self.accept)
        layout.addRow(okbtn)

    def user_input(self):
        v1 = self.edit1.text()
        v2 = self.edit2.text()
        return float(v1) if v1 else None, float(v2) if v2 else None
