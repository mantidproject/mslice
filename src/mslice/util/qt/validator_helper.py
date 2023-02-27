from qtpy.QtCore import QLocale
from qtpy.QtGui import QDoubleValidator


def double_validator_without_separator():
    """Returns a QDoubleValidator which doesn't accept a comma separator as input."""
    locale = QLocale(QLocale.C)
    locale.setNumberOptions(QLocale.RejectGroupSeparator)

    double_validator = QDoubleValidator()
    double_validator.setLocale(locale)
    return double_validator
