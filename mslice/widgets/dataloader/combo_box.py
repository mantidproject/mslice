
from mslice.util.qt.QtWidgets import QComboBox

class ComboBox(QComboBox):
    '''workaround for AttributeError when loading QComboBox with load_ui'''

    def setCurrentText(self, text):
        pass