# Multiple Traveling Salesmen Problem

This work try to solve the MTSP using CVXPY library

## Installation
- Python 3.11+
- CVXPY 1.4.1

```bash
    pip install -r requirements.txt
```

## Usage
To run the MTSP with specific value $m$, adjust value $m$ (line 43) in file ./src/cvxpy_mtsp.py in advance and run the following command:

```bash
    python ./src/cvxpy_mtsp.py
```

Output:
```bash
    1. Loading data from /home/tiennv/Github/EE5283EngineeringOptimization/FinalProject/data/numerical_example/data.csv ...
                    -> (9, 2)
                    -> Saving distance matrix to /home/tiennv/Github/EE5283EngineeringOptimization/FinalProject/data/numerical_example/data.distance.csv
    2. Ploting data points ...
    3. Create solver ...
    4. Solving to find optimal routines ...
    ===============================================================================
                                        CVXPY                                     
                                        v1.4.1                                    
    ===============================================================================
    (CVXPY) Dec 07 07:58:45 PM: Your problem has 90 variables, 64 constraints, and 0 parameters.
    (CVXPY) Dec 07 07:58:45 PM: It is compliant with the following grammars: DCP, DQCP
    (CVXPY) Dec 07 07:58:45 PM: (If you need to solve this problem multiple times, but with different data, consider using parameters.)
    (CVXPY) Dec 07 07:58:45 PM: CVXPY will first compile your problem; then, it will invoke a numerical solver to obtain a solution.
    (CVXPY) Dec 07 07:58:45 PM: Your problem is compiled with the CPP canonicalization backend.
    -------------------------------------------------------------------------------
                                    Compilation                                  
    -------------------------------------------------------------------------------
    (CVXPY) Dec 07 07:58:45 PM: Compiling problem (target solver=SCIPY).
    (CVXPY) Dec 07 07:58:45 PM: Reduction chain: Dcp2Cone -> CvxAttr2Constr -> ConeMatrixStuffing -> SCIPY
    (CVXPY) Dec 07 07:58:45 PM: Applying reduction Dcp2Cone
    (CVXPY) Dec 07 07:58:45 PM: Applying reduction CvxAttr2Constr
    (CVXPY) Dec 07 07:58:45 PM: Applying reduction ConeMatrixStuffing
    (CVXPY) Dec 07 07:58:45 PM: Applying reduction SCIPY
    (CVXPY) Dec 07 07:58:45 PM: Finished problem compilation (took 3.007e-02 seconds).
    -------------------------------------------------------------------------------
                                    Numerical solver                               
    -------------------------------------------------------------------------------
    (CVXPY) Dec 07 07:58:45 PM: Invoking solver SCIPY  to obtain a solution.
    Solver terminated with message: Optimization terminated successfully. (HiGHS Status 7: Optimal)
    -------------------------------------------------------------------------------
                                        Summary                                    
    -------------------------------------------------------------------------------
    (CVXPY) Dec 07 07:58:45 PM: Problem status: optimal
    (CVXPY) Dec 07 07:58:45 PM: Optimal value: 4.255e+01
    (CVXPY) Dec 07 07:58:45 PM: Compilation took 3.007e-02 seconds
    (CVXPY) Dec 07 07:58:45 PM: Solver (including time spent in interface) took 1.740e-02 seconds
    5. Gathering the path from found solution
            The path of Salesman_1 is: 1 => 5 => 4 => 2 => 3 => 1
            The path of Salesman_2 is: 1 => 7 => 8 => 9 => 6 => 1
    6. Write solution into /home/tiennv/Github/EE5283EngineeringOptimization/FinalProject/output/output_2_data.json ...
```
### To plot the curve showing the relationship between the number of salesmen and total cost
```bash
    python ./src./plot_relationship.py
``` 
The output is stored in ./output/objective_vs_salesmen.pdf


## Other information
- The synthesis data for showing numerical study can be found in ./data/numerical_example/ which contains visualization file, figure file.
- The input of running various number of salesmen can be found in ./output/numerical_result including 8 experimental results.
- Each output file run from specific value $m$ will be stored under folder ./output/numerical_result/. It contains all output: optimal value, optimal point X, Tours of salesmen, Traveling order vector u.  

