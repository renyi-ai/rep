
from src.functions.logit_comparison import Comparator
from src.functions.manipulator import  Manipulator

def get_comparator(name):
    return Comparator().get(name)

def get_manipulator(name):
    return Manipulator().get(name)

