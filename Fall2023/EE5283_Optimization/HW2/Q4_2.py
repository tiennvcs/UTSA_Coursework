from matplotlib import pyplot as plt
import numpy as np


def function(x: float):
    """
    - Parameters:
        + x (float): the real value 
    - Return: value of f at x
    """
    return 2 * (x - 2) ** 3 - x ** 2


def first_taylor_approx(x: float) -> float:
    """
   - Parameters:
       + x (float): the real value
   - Return: value of g(x) at x with g(x) is first-order series expansion of f(x), defined as g(x) = -7
   """
    return -7


def second_taylor_approx(x: float) -> float:
    """
   - Parameters:
       + x (float): the real value
   - Return: value of h(x) at x with g(x) is second-order series expansion of f(x), defined as h(x) = 5(x-3)^2 - 7
   """
    return 5 * (x - 3) ** 2 - 7


def plot_graph():
    # Generate data points
    X = np.linspace(-20, 20, 1000)  # list of x values
    F = np.array([function(x) for x in X])  # list of f values
    G = np.array([first_taylor_approx(x) for x in X])  # list of 1st-order derivatives
    H = np.array([second_taylor_approx(x) for x in X])  # list of 2nd-order derivatives

    # Configuration
    plt.style.use("seaborn-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot F, G, H functions
    ax.plot(X, F, color='b', label=r"$f(x) = 2(x-2)^3-x^2$")
    ax.plot(X, G, color='r', label=r"$g(x) = -7$")
    ax.plot(X, H, color='y', label=r"$h(x) = 5(x-3)^2 - 7$")

    # Plot the point x0 = (3, -7)
    plt.plot(3, -7, marker="o", markersize=8, markeredgecolor="red", markerfacecolor="green")
    y = [t for t in range(-20, -7 + 1, 1)]
    plt.plot([3] * len(y), y, color='black', markersize=1, linestyle='dashed')
    x = [t for t in range(-20, 3 + 1, 1)]
    plt.plot(x, [-7] * len(x), color='black', markersize=1, linestyle='dashed')

    # Make up for plot
    ax.set_xticks([x for x in range(-20, 20, 1)])
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title("1st-order g(x), and 2nd-order h(x) Taylor approximation graph", fontsize=14)
    ax.set_xticks([i for i in range(-3, 7, 1)])
    ax.set_yticks([i for i in range(-17, 20, 2)])
    plt.xlim(-3, 7)
    plt.ylim(-20, 20)
    ax.legend()

    plt.show()
    fig.savefig("./HW2/plot.pdf")


if __name__ == '__main__':
    plot_graph()
