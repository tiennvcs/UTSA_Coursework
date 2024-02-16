import json
import os
import pandas as pd
from utils.data_reader_writer import load_data_from_file, np_encoder, load_weight_matrix
from utils.plot import plot
from utils.distance import build_distance_matrix
from cvx_solver import CVXSolver

    
def run(args: dict):
    print("1. Loading data from {} ...".format(args['data_path']))
    data = load_data_from_file(args['data_path'])
    D = build_distance_matrix(data)
    print("\t\t -> {}".format(data.shape))
    df = pd.DataFrame(D, index=None)
    df.to_csv(args['distance_path'])
    print("\t\t -> Saving distance matrix to {}".format(args['distance_path']))
    print("2. Ploting data points ...")
    plot(data=data, output_file=args['visualize_path'])

    print("3. Create solver ...")
    solver = CVXSolver()

    print("4. Solving to find optimal routines ...")
    solver.solve(distance_matrix=D, m=args['m'])

    print("5. Gathering the path from found solution")
    solver.gather_path()

    print("6. Write solution into {} ...".format(args['output_file']))
    args.update(solver.get_data())
    args.update(solver.get_path())
    with open(args['output_file'], 'w') as f:
        json.dump(args, f, indent=4, ensure_ascii=False, default=np_encoder)

if __name__ == '__main__':
    args = {
        'data_path': "/home/tiennv/Github/EE5283EngineeringOptimization/FinalProject/data/numerical_example/data.csv",
        # 'matrix_path': "/home/tiennv/Github/EE5283EngineeringOptimization/FinalProject/data/test_tsp/gr21.distance.csv",
        "visualize_path": "/home/tiennv/Github/EE5283EngineeringOptimization/FinalProject/data/numerical_example/data_visualize.pdf",
        "distance_path": "/home/tiennv/Github/EE5283EngineeringOptimization/FinalProject/data/numerical_example/data.distance.csv" ,
        "output_file": "",
        "m": 2,
    }
    args["output_file"] = os.path.join("/home/tiennv/Github/EE5283EngineeringOptimization/FinalProject/output/", 
                                       "output_{}_".format(args['m'])+os.path.basename(args["data_path"]).split(".")[0]+".json")

    run(args)

