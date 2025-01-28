from qtpy.QtWidgets import (
    QDialog,
    QFormLayout,
    QLabel,
    QDoubleSpinBox,
    QPushButton,
    QCheckBox,
)


class EfInputDialog(QDialog):
    def __init__(self, parent=None):
        super(EfInputDialog, self).__init__(parent)
        self.setWindowTitle("Indirect Ef")
        layout = QFormLayout(self)
        self.text = QLabel(self)
        layout.addRow(self.text)
        self.efspin = QDoubleSpinBox(self)
        self.efspin.setMinimum(0.0001)
        self.efspin.setDecimals(4)
        layout.addRow(self.efspin)
        okbtn = QPushButton("OK")
        okbtn.clicked.connect(self.accept)
        self.allchk = QCheckBox("Apply to all workspaces", self)
        self.allchk.hide()
        layout.addRow(self.allchk, okbtn)

    def showCheck(self):
        self.allchk.show()

    def value(self):
        return self.efspin.value(), self.allchk.isChecked()

    @staticmethod
    def getEf(workspace, hasMultipleWS=False, default_value=None, parent=None):
        """Static method to get an Ef"""
        dialog = EfInputDialog(parent)
        dialog.text.setText("Input Fixed Final Energy in meV for %s:" % (workspace))
        if default_value:
            dialog.efspin.setValue(default_value)
        if hasMultipleWS:
            dialog.showCheck()
        result = dialog.exec_()
        Ef, allChecked = dialog.value()
        return Ef, allChecked, result == QDialog.Accepted
