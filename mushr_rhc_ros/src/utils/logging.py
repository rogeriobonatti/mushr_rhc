"""
Logging utilities.
"""

import sys

class Tee(object):
    """ Tee class to overwrite stdout. """
    # Ref: https://stackoverflow.com/questions/616645/how-to-duplicate-sys-stdout-to-a-log-file
    def __init__(self, path, mode):
        self.file = open(path, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()