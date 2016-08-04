from cut_ui import Ui_Form
from PyQt4.QtGui import QWidget


class CutWidget(QWidget,Ui_Form):
    def __init__(self,*args, **kwargs):
        super(CutWidget,self).__init__(*args, **kwargs)
        self.setupUi(self)
