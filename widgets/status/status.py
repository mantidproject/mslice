from status_ui import Ui_Form
from PyQt4.QtGui import QWidget


class StatusWidget(QWidget,Ui_Form):
    def __init__(self,*args, **kwargs):
        super(StatusWidget,self).__init__(*args, **kwargs)
        self.setupUi(self)
