#This file defines generic functions to get and display variables from users
from ui_mock import Getter, displayMessage


def get(title):
    getter = Getter(title)
    while not getter.is_done():
        pass
    return getter.get_data()

def display(*args):
    displayMessage(*args)