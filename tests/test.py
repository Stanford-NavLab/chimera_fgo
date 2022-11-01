"""Test short description

Longer description (if necessary).

Notes
-----

"""

# <General import statements>
import math

# <Project specific import statements>
from projectname.pkg2.module2 import add_and_mult_function


def test_add_and_mult_function():
    a = 2
    b = 3
    c = 4
    print("4 * (2 + 3) = ", add_and_mult_function(a, b, c))

if __name__ == '__main__':
    test_add_and_mult_function()