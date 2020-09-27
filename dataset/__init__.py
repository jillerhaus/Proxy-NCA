
from .cars import Cars
from .cub import CUBirds
from .sop import SOProducts
from .food import Food
from .food_test import Food_test
from . import utils
from .base import BaseDataset


_type = {
    'cars': Cars,
    'cub': CUBirds,
    'sop': SOProducts,
    'food': Food,
    'food_test': Food_test
}


def load(name, root, classes, transform = None):
    return _type[name](root = root, classes = classes, transform = transform)
