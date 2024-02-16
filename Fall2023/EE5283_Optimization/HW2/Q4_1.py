from matplotlib import pyplot as plt


def function(x: float):
    """
    - Parameters:
        + x (float): the real value 
    - Return: value of f at x
    """
    return 2*(x-2)**3 - x**2


def first_taylor_approx(x: float)-> float:
     """
    - Parameters:
        + x (float): the real value 
    - Return: value of g(x) at x with g(x) is first-order series expansion of f(x), defined as g(x) = -7
    """
     return -7


def second_taylor_approx(x: float)->float:
     """
    - Parameters:
        + x (float): the real value 
    - Return: value of h(x) at x with g(x) is second-order series expansion of f(x), defined as h(x) = 5(x-3)^2 - 7
    """
     return 5*(x-3)**2 - 7

    

if __name__ == '__main__':
    x = 2
    f = function(x)
    g = first_taylor_approx(x)
    h = second_taylor_approx(x)
    print("Value of f({}) is {}".format(x, f))
    print("Value of g({}) is {}".format(x, g))
    print("Value of h({}) is {}".format(x, h))