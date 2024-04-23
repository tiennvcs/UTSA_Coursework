import os
import numpy as np
from src import FourierTransformDenoiser, TrigonometricSignal, PolynomialSignal, IBMDataset
from matplotlib import pyplot as plt
from src import mean_square_error


if __name__ == '__main__':

    # For the trigonometric signals
    amplitudes = np.array([1/3, 1/2])
    frequencies = np.array([19, 29])
    function_names = ['cos', 'sin']
    mean = 0
    var = 1
    sampling_rate = 1000
    threshold_ratio_range = [0.2, 0.4]

    # Dataset generation
    dataset_generator = TrigonometricSignal(amplitudes=amplitudes, frequencies=frequencies, function_names=function_names, mean=mean, var=var)
    data = dataset_generator.generate(sampling_rate=sampling_rate)
    output_dir = './output/signal_reconstruction/trig'

    ## For the polynomial signals
    # degree = 5
    # coeffs = np.array([1, 0, 12, -60, 120, -70])
    # mean = 0
    # var = 1
    # sampling_rate = 1000
    # dataset_generator = PolynomialSignal(degree=degree, coefficents=coeffs, mean=mean, var=var)
    # data = dataset_generator.generate(sampling_rate=sampling_rate)
    # output_dir = "./output/signal_reconstruction/poly"
    # threshold_ratio_range = [0.2, 0.4]

    ## For IBM dataset
    # sampling_rate = 1000
    # input_data = '/home/tiennv/Github/UTSA_Coursework/Spring2024/EE5163_DPS/Project/data/ibm_stock_2022_Jan2March.csv'
    # num_samples = 2000
    # dataset_generator = IBMDataset(data_file=input_data)
    # data = dataset_generator.generate(num_samples=num_samples)
    # output_dir = "./output/signal_reconstruction/ibm_dataset"
    # threshold_ratio_range = [1e-3, 0.01]

    colormark = ['r-', 'c-']
    alpha_values = [0.4, 0.8]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    ax.plot(data['t'], data['x_t'], 'g-', label='Noise data', alpha=1)
    ax.plot(data['t'], data['true_xt'], 'b-', label='Clean data', alpha=0.9)
    ax.set_xlabel(r'$t$ (seconds)')
    ax.set_ylabel(r'$\hat{f}(t)$')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for i, threshold_ratio in enumerate(threshold_ratio_range):
        denoiser = FourierTransformDenoiser(threshold_ratio=threshold_ratio)
        output = denoiser.run(input_signal=data['x_t'], sampling_rate=sampling_rate)
        ax.plot(data['t'], output['clean_xt'].real, colormark[i], label="Threshold = {}".format(threshold_ratio), alpha=alpha_values[i])
        # mse = mean_square_error(data['true_xt'], output['clean_xt'].real)
        # print("Threshold ratio: {} - Mean square error: {}".format(threshold_ratio, mse))

    ax.legend(loc='best')
    plt.show()
    output_file = os.path.join(output_dir, 'ifft_output_and_noise_signal_and_clean_signal.pdf')
    fig.savefig(output_file, bbox_inches='tight')

        

