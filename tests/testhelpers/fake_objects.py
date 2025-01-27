"""
Fake object classes used for the purposes of testing.
"""


class FakeClipboard:
    Clipboard = None

    def __init__(self):
        self.text = ""

    def setText(self, text, mode=None):
        self.text = text


class FakeFile:
    def __init__(self):
        self.text = ""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def writelines(self, lines):
        self.text = "".join(lines)
