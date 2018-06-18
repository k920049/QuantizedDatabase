"""
from __future__ import print_function

try:
    import numpy
    print(numpy.get_include(), end='\n')
except:
    pass
"""

a = 16
b = 32

c = a | b

print(c)
