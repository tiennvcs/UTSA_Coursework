import os
import numpy as np
from src import FourierTransformDenoiser, TrigonometricSignal, PolynomialSignal
from matplotlib import pyplot as plt
from src import mean_square_error


if __name__ == '__main__':

    ## For the trigonometric signals
    amplitudes = np.array([1/3, 1/2])
    frequencies = np.array([19, 29])
    function_names = ['cos', 'sin']
    mean = 0
    var = 1
    sampling_rate = 1000

    # Dataset generation
    dataset_generator = TrigonometricSignal(amplitudes=amplitudes, frequencies=frequencies, function_names=function_names, mean=mean, var=var)
    data = dataset_generator.generate(sampling_rate=sampling_rate)
    output_dir = './output/demo/trig'

    ## For the polynomial signals
    # degree = 5
    # coeffs = np.array([1, 0, 12, -60, 120, -70])
    # mean = 0
    # var = 1
    # sampling_rate = 1000
    # dataset_generator = PolynomialSignal(degree=degree, coefficents=coeffs, mean=mean, var=var)
    # data = dataset_generator.generate(sampling_rate=sampling_rate)
    # output_dir = "./output/demo/poly"
    
    if __debug__:
        # Plot the dataset
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
        ax.plot(data['t'], data['x_t'], 'g-', label='Noise data')
        ax.plot(data['t'], data['true_xt'], 'b-', label='Clean data')
        ax.set_xlabel("t (seconds)")
        ax.set_ylabel("f(t)")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.legend(loc='best')
        plt.show()
        fig.savefig(os.path.join(output_dir, 'noise_signal_versus_true_signal.pdf'), bbox_inches='tight')

    threshold_ratio = 0.9
    print("Threshold: {}".format(threshold_ratio))
    denoiser = FourierTransformDenoiser(threshold_ratio=threshold_ratio)
    output = denoiser.run(input_signal=data['x_t'], sampling_rate=sampling_rate)

    if __debug__:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
        ax.stem(output['freq_oneside'], abs(output['Xf_oneside'].real)/np.abs(np.max(output['Xf_oneside'].real)), 'b', markerfmt=" ", basefmt="-b") 
        ax.axhline(y=threshold_ratio, color = 'r', linestyle = '--')
        ax.set_xlabel(r'$\omega$ (Hz)')
        ax.set_ylabel(r'$|g(\omega)|$')
        ax.set_yscale('log')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.show()
        fig.savefig(os.path.join(output_dir, 'fft_output.pdf'), bbox_inches='tight')

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
        ax.stem(output['freq_oneside'],  abs(output['clean_Xf_oneside'].real), 'b', markerfmt=" ", basefmt="-b")
        ax.set_xlabel(r'$\omega$ (Hz)')
        ax.set_ylabel(r'$|\hat{g}(\omega)$|')
        ax.set_yscale('log')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.show()
        fig.savefig(os.path.join(output_dir, 'thresholding_output.pdf'), bbox_inches='tight')

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
        ax.plot(data['t'], output['clean_xt'].real, 'r-', label="Denoising signal")
        ax.set_xlabel(r'$t (seconds)$')
        ax.set_ylabel(r'$\hat{f}(t)$')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.show()
        fig.savefig(os.path.join(output_dir, 'ifft_output.pdf'), bbox_inches='tight')

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
        ax.plot(data['t'], data['x_t'], 'g-', label='Noise data', alpha=0.4)
        ax.plot(data['t'], output['clean_xt'].real, 'r-', label="Removal noise signal")
        ax.plot(data['t'], data['true_xt'], 'b-', label='Clean data', alpha=0.6)
        ax.set_xlabel(r'$t$ (seconds)')
        ax.set_ylabel(r'$\hat{f}(t)$')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.legend(loc='best')
        plt.show()
        fig.savefig(os.path.join(output_dir, 'ifft_output_and_noise_signal_and_clean_signal.pdf'), bbox_inches='tight')

    mse = mean_square_error(data['true_xt'], output['clean_xt'].real)
    print("Mean square error: {}".format(mse))
