from singlecrystal_ui import Ui_Form
from PyQt4.QtGui import QWidget


class SingleCrystalWidget(QWidget,Ui_Form):
    def __init__(self,*args, **kwargs):
        super(SingleCrystalWidget,self).__init__(*args, **kwargs)
        self.setupUi(self)
