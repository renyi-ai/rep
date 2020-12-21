import numpy as np

class Comparator:

    def get(self, str_func):
        return getattr(self, str_func)

    # ====================================================================
    # DEFINE FUNCTIONS HERE
    # ====================================================================

    def l1_diff(self, true_y, pre_y, post_y):
        return np.mean(np.abs(pre_y-post_y))

    def l1_max(self, true_y, pre_y, post_y):
        return np.max(np.abs(pre_y-post_y))

    def pre_acc(self, true_y, pre_y, post_y):
        pred = pre_y.argmax(axis=1).flatten()
        return (pred==true_y).sum() / true_y.size

    def post_acc(self, true_y, pre_y, post_y):
        pred = post_y.argmax(axis=1).flatten()
        return (pred==true_y).sum() / true_y.size