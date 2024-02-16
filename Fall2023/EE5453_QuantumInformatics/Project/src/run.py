import json
from utils.read_data_lr_utils import read_data_lr_from_path
from utils.plot_solution import plot_regressor
from lr_classical_solver import ClassicalSolver
from lr_quantum_solver import QuantumSolver
from utils.metric import mean_square_error


def run(args):
    print("1. Loading data from {}".format(args['data_path']))
    X, y = read_data_lr_from_path(data_path=args['data_path'])
    
    print("2. Initilize solvers")
    c_solver = ClassicalSolver()
    q_solver = QuantumSolver()
    
    print("3. Geting the closed-form solution ...")
    c_w, c_time = c_solver.solve(X, y)
    print("\tClassical solution: y = {}x + {}".format(c_w[1], c_w[0]))
    
    print("4. Getting the quantum approximation solution ...")
    q_w, q_time = q_solver.solve(X, y)
    q_w = q_w*1e4/2
    print("\tQuantum solution: y = {}x + {}".format(q_w[1], q_w[0]))
    
    print("5. Generating the plot result at {}...".format(args['output_plot_path']))
    plot_regressor(X, y, [c_w, q_w], output_file=args['output_plot_path'])

    print("7. Calculating the Estimated MSE ...")
    c_mse = mean_square_error(X, y, c_w)
    print("\tClassical MSE: {}".format(c_mse))

    q_mse = mean_square_error(X, y, q_w)
    print("\tQuantum MSE: {}".format(q_mse))

    print("6. Storing solution into {}...".format(args['solution_path']))
    storing_data = {
        'q_sol': q_w.tolist(),
        'c_sol': c_w.tolist(),
        'q_time': q_time,
        'c_time': c_time,
        'q_mse': q_mse.tolist(),
        'c_mse': c_mse.tolist(),
    }
    with open(args['solution_path'], 'w') as f:
        json.dump(storing_data, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    args = {
        "data_path": "./data/data1/data.csv",
        "output_plot_path": "./output/quantum_classical_compare.pdf",
        "solution_path": "./output/solution.json"
    }
    
    run(args)
