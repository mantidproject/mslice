from PyQt5.QtWidgets import QComboBox, QFormLayout, QLabel, QPushButton, QWidget, QSpacerItem, QSizePolicy, QVBoxLayout, \
    QHBoxLayout
from qtpy.QtWidgets import QDialog, QCheckBox

class QCacheableInputDialog(QDialog):
    """
    An input dialog that allows the user to select an option from a list or
    input a custom one of their own.

    The dialog also contains a checkbox, which allows the user to indicate that
    they wish to cache the entered value to be used later.
    """

    def __init__(self, parent=None):
        super(QCacheableInputDialog, self).__init__(parent)

        layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        self.text = QLabel(self)
        form_layout.addRow(self.text)
        self.combo = QComboBox(self)
        self.combo.setEditable(True)
        form_layout.addRow(self.combo)
        self.check_text = QLabel(self)
        self.check_text.setText("Overwrite Cached Value?")
        self.cache_checkbox = QCheckBox(self)
        form_layout.addRow(self.check_text, self.cache_checkbox)
        layout.addLayout(form_layout)
        button_layout = QHBoxLayout()
        self.okay_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")
        self.okay_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        spacer = QSpacerItem(20, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        button_layout.addSpacerItem(spacer)
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.okay_button)
        layout.addLayout(button_layout)

    def value(self):
        return self.combo.currentText(), self.cache_checkbox.isChecked()

    @staticmethod
    def ask_for_input(parent: QWidget, title: str, label_text: str, options: list) -> tuple[str, bool, bool]:
        """
        :param parent: Parent QWidget.
        :param title: Dialog title.
        :param label_text: Text indicating what the combobox selects.
        :param options: Options the combobox will be populated with.
        :return: Chosen option, true if option should be cached, true if OK was clicked.
        """
        dialog = QCacheableInputDialog(parent)
        dialog.setWindowTitle(title)
        dialog.text.setText(label_text)
        dialog.combo.addItems(options)

        dialog_result = dialog.exec_()
        chosen_option, is_cached = dialog.value()
        return chosen_option, is_cached, dialog_result == QDialog.Accepted