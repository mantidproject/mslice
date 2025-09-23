from PyQt5.QtWidgets import QComboBox, QVBoxLayout, QFormLayout, QLabel, QPushButton
from qtpy.QtWidgets import QDialog, QCheckBox

class QCacheableInputDialog(QDialog):

    def __init__(self, parent=None):
        super(QCacheableInputDialog, self).__init__(parent)

        layout = QFormLayout(self)
        self.text = QLabel(self)
        layout.addRow(self.text)
        self.combo = QComboBox(self)
        self.combo.setEditable(True)
        layout.addRow(self.combo)
        self.check_text = QLabel(self)
        self.check_text.setText("Overwrite Cached Value?")
        self.cache_checkbox = QCheckBox(self)
        layout.addRow(self.check_text, self.cache_checkbox)
        self.okay_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")
        self.okay_button.clicked.connect(self.accept)
        layout.addRow(self.cancel_button, self.okay_button)

    def value(self):
        return self.combo.currentText(), self.cache_checkbox.isChecked()

    @staticmethod
    def ask_for_input(parent, title, label_text, options):
        dialog = QCacheableInputDialog(parent)
        dialog.setWindowTitle(title)
        dialog.text.setText(label_text)
        dialog.combo.addItems(options)

        dialog_result = dialog.exec_()
        chosen_option, is_cached = dialog.value()
        return chosen_option, is_cached, dialog_result == QDialog.Accepted