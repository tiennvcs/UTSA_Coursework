from matplotlib import pyplot as plt


def function(x: float):
    """
    - Parameters:
        + x (float): the real value 
    - Return: value of f at x
    """
    return 2 * (x - 2) ** 3 - x ** 2


def first_derivative(x: float):
    """
   - Parameters:
       + x (float): the real value
   - Return: value of f' at x
   """
    return 6 * (x - 2) ** 2 - 2 * x


def second_derivative(x: float):
    """
   - Parameters:
       + x (float): the real value
   - Return: value of f'' at x
   """
    return 12 * (x - 2) - 2


if __name__ == '__main__':
    x = 3
    f = function(x)
    f1 = first_derivative(x)
    f2 = second_derivative(x)
    print("Value of f at x = {} is {}".format(x, f))
    print("First-order derivative of f at x = {} is {}".format(x, f1))
    print("Second-order derivative of f at x = {} is {}".format(x, f2))
