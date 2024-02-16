from matplotlib import pyplot as plt


def plot(data, solution=None, output_file="./data_plot.pdf"):
    if solution is None:
        fig, ax = plt.subplots(figsize=(9, 6))
        X = [coord[0] for coord in data]
        y = [coord[1] for coord in data]
        for i in range(len(X)):
            ax.text(X[i], y[i], '$v_{0}$'.format(i+1))
        # for i in range(len(data)-1):
        #     for j in range(i+1, len(data)):
        #         x_list = [data[i][0], data[j][0]]
        #         y_list = [data[i][1], data[j][1]]
        #         plt.plot(x_list, y_list, 'r-')
        ax.scatter(X, y)
        fig.savefig(output_file)
        plt.show()

# def plot_solution()
#     xy_cords = np.zeros((n,2))
#     for i in range(0, n):
#         xy_cords[i,0] = distance.distance((points[0][1],0), (points[i][1],0)).km
#         xy_cords[i,1] = distance.distance((0,points[0][0]), (0,points[i][0])).km
#     # Plotting the points
#     fig, ax = plt.subplots(figsize=(14,7))
#     for i in range(n):
#         ax.annotate(str(i), xy=(xy_cords[i,0], xy_cords[i,1]+0.1))
#     ax.scatter(xy_cords[:,0],xy_cords[:,1])
#     ax.plot(xy_cords[orden,0], xy_cords[orden,1])
#     ax.set_title(' => '.join(map(str, orden)), fontsize=14)
#     fig.savefig("visualize_sol.pdf")
#     plt.show()