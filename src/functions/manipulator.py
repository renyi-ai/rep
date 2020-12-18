import numpy as np

class Manipulator:

    def get(self, str_func):
        return getattr(self, str_func)

    # ====================================================================
    # DEFINE FUNCTIONS HERE
    # ====================================================================

    def identity(self, x):
        return x