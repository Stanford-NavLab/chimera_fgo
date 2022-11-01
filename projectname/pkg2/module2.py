"""Module short description

Longer description (if necessary).

Notes
-----

References
----------

"""

# <General import statements>
import math

# <Project specific import statements>
from projectname.pkg1.module1 import add_function


# <Example function and docstring formatting>
def add_and_mult_function(a, b, c):
    """Adds two numbers then multiplies by third
    
    Parameters
    ----------
    a : float
        First number to add
    b : float
        Second number to add
    c : float
        Number to multiply by

    Returns
    -------
    float
        Sum of first two numbers multiplied by third

    """
    return c * add_function(a, b)
