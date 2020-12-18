import numpy as np

class Comparator:

    def get(self, str_func):
        return getattr(self, str_func)

    # ====================================================================
    # DEFINE FUNCTIONS HERE
    # ====================================================================

    def l1_diff(self, x, y):
        return np.mean(np.abs(x-y))

    def l1_max(self, x, y):
        return np.max(np.abs(x-y))