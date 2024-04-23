import json
import os
import numpy as np
from tqdm import tqdm
from src import FourierTransformDenoiser, IBMDataset
from src import signal2noise
import argparse
from typing import List
from src import plot_res


def get_arguments():
    parser = argparse.ArgumentParser(description="Thresholding performance on synthetic signal")
    parser.add_argument("--input_data", type=str, required=True, default='/home/tiennv/Github/UTSA_Coursework/Spring2024/EE5163_DPS/Project/data/ibm_stock_2022_Jan2March.csv')
    parser.add_argument("--num_samples", type=int, required=True, default=2000)
    parser.add_argument("--sampling_rate", type=int, required=True, default=1000)
    parser.add_argument("--output_dir", type=str, default="./output/ibm_dataset/")
    return vars(parser.parse_args())


def run(
        input_data: str,
        num_samples: int,
        sampling_rate: int,
        visualize: bool=True,
        output_dir: str='./output',
    ):
    
    s2n_before = {}
    s2n_after = {}

    dataset_generator = IBMDataset(data_file=input_data)

    os.makedirs(args['output_dir'], exist_ok=True)
 
    threshold_ratio_lst = np.arange(0, 5e-5, 1e-6)
    
    for threshold_ratio in threshold_ratio_lst:
        print("Threshold = {:.3}".format(threshold_ratio))
        data = dataset_generator.generate(num_samples=num_samples)
        denoiser = FourierTransformDenoiser(threshold_ratio=threshold_ratio)
        output = denoiser.run(input_signal=data['x_t'], sampling_rate=sampling_rate)        
        s2n_before[threshold_ratio] = signal2noise(frequency_signal=output['X_f'].real)
        s2n_after[threshold_ratio] = signal2noise(frequency_signal=output['clean_Xf'].real)
    
    max_idx = -1
    max_value = -1e6
    for k, v in s2n_after.items():
        if v >= max_value:
            max_idx = k
            max_value = v
    print("Max idx: ", max_idx)
    print("Max value: ", v)
    output_result_file = os.path.join(output_dir, 'res.json')
    with open(output_result_file, 'w') as f:
        json.dump({
            'threshold_ratio': threshold_ratio_lst.tolist(),
            's2n_before': list(s2n_before.values()),
            's2n_after': list(s2n_after.values()),
        },
            f, indent=4, ensure_ascii=False
        )
        
    if visualize:
        c = np.mean(list(s2n_before.values()))
        output_file = os.path.join(output_dir, 'signal2noise.pdf')
        plot_res(X=threshold_ratio_lst, y=s2n_after.values(), c=c, x_label='Threshold', y_label="SNR (db)", output_file=output_file)


if __name__ == '__main__':
    args = get_arguments()    
    run(input_data=args['input_data'], num_samples=args['num_samples'], sampling_rate=args['sampling_rate'], output_dir=args['output_dir'])