import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle


def rotate(x: np.array, theta) -> np.array:
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([
        [c, s],
        [-s, c]
    ])
    return np.dot(R, x.T)


def plot_point_and_operation(original_point: np.array, theta_list: list):
    fig, ax = plt.subplots(figsize=(10, 10))
    # Draw points into polar coordinates
    ax.plot([0] * 20, range(-10, 10, 1), color='black')
    ax.plot(range(-10, 10, 1), [0] * 20, color='black')
    for theta in theta_list:
        new_x = rotate(original_point, theta)
        point_color = 'red'
        line_color = 'green'
        if theta == np.radians(90):
            point_color = 'brown'
            line_color = "pink"
            ax.text(new_x[0] + 0.2, new_x[1], r"$90^{\circ}$")
        if theta == np.radians(180):
            point_color = 'brown'
            line_color = "pink"
            ax.text(new_x[0] + 0.2, new_x[1], r"$180^{\circ}$")
        ax.plot([0, new_x[0]], [0, new_x[1]], color=line_color, linestyle='dashed')
        ax.scatter(new_x[0], new_x[1], color=point_color)

    ax.scatter(original_point[0], original_point[1], color='b', s=20, label=r"Original point $x_0 = (3, 4)$")
    ax.plot([0, original_point[0]], [0, original_point[1]], color='b', linestyle='dashed')

    # Plot the circle with R = ||original_point||_2^2, I=(0,0)
    ellipse = Circle(xy=(0, 0),
                     radius=np.linalg.norm(original_point, 2), alpha=0.3)
    ax.add_artist(ellipse)

    ax.set_xticks(range(-10, 10, 1))
    ax.set_yticks(range(-10, 10, 1))
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.legend()
    plt.show()
    fig.savefig('./Q2_plot.pdf')


if __name__ == '__main__':
    origin = np.array([3, 4])
    theta_list = np.radians(range(0, 360, 30))
    plot_point_and_operation(origin, theta_list)
