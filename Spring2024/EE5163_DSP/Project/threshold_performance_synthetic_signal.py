import json
import os
import numpy as np
from tqdm import tqdm
from src import FourierTransformDenoiser, TrigonometricSignal, PolynomialSignal
from matplotlib import pyplot as plt
from src import mean_square_error, signal2noise
import argparse
from typing import List
from src import plot_res


def get_arguments():
    parser = argparse.ArgumentParser(description="Thresholding performance on synthetic signal")
    parser.add_argument("--signal_type", choices=["trig", "poly"], required=True)
    parser.add_argument("--amps", nargs="*", type=float, default=[1/3, 1/2])
    parser.add_argument("--degree", type=int, default=5)
    parser.add_argument("--coeffs", nargs="*", type=float, default=[1, 0, 12, -60, 120, -70])
    parser.add_argument("--freqs", nargs="*", type=int, default=[19, 29])
    parser.add_argument("--f_names", nargs="*", type=str, default=['cos', 'sin'])
    parser.add_argument("--sampling_rate", type=int, required=True)
    parser.add_argument("--mean", type=float, required=False, default=0)
    parser.add_argument("--var", type=float, required=True, default=1)
    parser.add_argument("--realization", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="./output/synthetic/threshold/")
    return vars(parser.parse_args())


def run(
        signal_type: str,
        amplitudes: np.ndarray, 
        frequencies: np.ndarray, 
        function_names: List[str], 
        degree: int,
        coeffs: np.ndarray,
        mean: float, 
        var: float, 
        sampling_rate: int, 
        realization: int,
        visualize: bool=True,
        output_dir: str='./output',
    ):
    
    mse_before = {}
    mse_after = {}

    s2n_before = {}
    s2n_after = {}

    if signal_type == 'trig':
        dataset_generator = TrigonometricSignal(amplitudes=amplitudes, frequencies=frequencies, function_names=function_names, mean=mean, var=var)
    elif signal_type == 'poly':
        dataset_generator = PolynomialSignal(degree=degree, coefficents=coeffs, mean=mean, var=var)
    else:
        print("Invalid signal type")
        exit(0)


    os.makedirs(args['output_dir'], exist_ok=True)

    threshold_ratio_lst = np.arange(0, 1+1e-6, 0.05)
    for threshold_ratio in threshold_ratio_lst:
        print("Threshold = {:.3}".format(threshold_ratio))
        denoiser = FourierTransformDenoiser(threshold_ratio=threshold_ratio)
        mse_before_lst = []
        s2n_before_lst  = []
        mse_after_lst = []
        s2n_after_lst = []
        for _ in range(0, realization):
            data = dataset_generator.generate(sampling_rate=sampling_rate)
            mse_before_lst.append(mean_square_error(data['true_xt'], data['x_t']))
            s2n_before_lst.append(signal2noise(data['x_t'].real))
            output = denoiser.run(input_signal=data['x_t'], sampling_rate=sampling_rate)
            mse_after_lst.append(mean_square_error(data['true_xt'], output['clean_xt'].real))
            s2n_after_lst.append(signal2noise(output['clean_xt'].real))
        
        mse_before[threshold_ratio] = np.mean(mse_before_lst).tolist()
        mse_after[threshold_ratio] = np.mean(mse_after_lst).tolist()
        s2n_before[threshold_ratio] = np.mean(s2n_before_lst).tolist()
        s2n_after[threshold_ratio] = np.mean(s2n_after_lst).tolist()
    
    output_result_file = os.path.join(output_dir, 'res.json')
    with open(output_result_file, 'w') as f:
        json.dump({
            'threshold_ratio': threshold_ratio_lst.tolist(),
            'mse_before': list(mse_before.values()),
            's2n_before': list(s2n_before.values()),
            'mse_after': list(mse_after.values()),
            's2n_after': list(s2n_after.values())
        },
            f, indent=4, ensure_ascii=False
        )
        
    if visualize:
        c = np.mean(list(mse_before.values()))
        output_file = os.path.join(output_dir, 'mse.pdf')
        plot_res(X=threshold_ratio_lst, y=mse_after.values(), c=c, x_label='Threshold', y_label=r"$E\{\frac{1}{N}||f(t)-f_{true}(t)||_2^2\}$", output_file=output_file)

if __name__ == '__main__':
    args = get_arguments()

    assert len(args['amps']) == len(args['freqs']) and len(args['amps']) == len(args['freqs']), "The input lengths are incompatible"
    
    run(signal_type=args['signal_type'], amplitudes=np.array(args['amps']), frequencies=np.array(args['freqs']), function_names=args['f_names'], degree=args['degree'], coeffs=args['coeffs'],
        mean=args['mean'], var=args['var'], 
        sampling_rate=args['sampling_rate'], realization=args['realization'], output_dir=args['output_dir'])