from ui_mock import Getter, displayMessage


def get(title):
    getter = Getter(title)
    while not getter.is_done():
        pass
    return getter.data

def display(*args):
    displayMessage(*args)