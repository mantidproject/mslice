from slice_ui import Ui_Form
from PyQt4.QtGui import QWidget


class SliceWidget(QWidget,Ui_Form):
    def __init__(self,*args, **kwargs):
        super(SliceWidget,self).__init__(*args, **kwargs)
        self.setupUi(self)
