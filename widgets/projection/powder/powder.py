from powder_ui import Ui_Form
from PyQt4.QtGui import QWidget


class PowderWidget(QWidget,Ui_Form):
    def __init__(self,*args, **kwargs):
        super(PowderWidget,self).__init__(*args, **kwargs)
        self.setupUi(self)
