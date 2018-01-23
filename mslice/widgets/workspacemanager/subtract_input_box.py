from mslice.util.qt import load_ui
from mslice.util.qt.QtWidgets import QDialog


class SubtractInputBox(QDialog):

    def __init__(self, ws_list, parent=None):
        QDialog.__init__(self, parent)
        load_ui(__file__, 'subtract_input_box.ui', self)
        for i in range(ws_list.count()):
            item = ws_list.item(i).clone()
            self.listWidget.insertItem(i, item)

    def user_input(self):
        background_ws = self.listWidget.selectedItems()[0].text()
        return background_ws, self.ssf.value()