# For python version compatibility

import importlib

def reload(module):
    return importlib.reload(module)
