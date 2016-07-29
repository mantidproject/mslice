#This file defines generic functions to get and display variables from users
from ui_mock import GetInputFromUser, display_message


def get(title):
    getter = GetInputFromUser(title)
    while not getter.is_done():
        pass
    return getter.get_data()

def display(*args):
    display_message(*args)