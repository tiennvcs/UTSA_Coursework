import numpy as np
from src import FourierTransformDenoiser, IBMDataset
from matplotlib import pyplot as plt
from src import signal2noise


if __name__ == '__main__':

    # amplitudes = np.array([1/3, 1/2])
    # frequencies = np.array([19, 29])
    # function_names = ['cos', 'sin']
    # mean = 0
    # var = 1
    # sampling_rate = 1000

    # # Dataset generation
    # dataset_generator = TrigonometricSignal(amplitudes=amplitudes, frequencies=frequencies, function_names=function_names, mean=mean, var=var)
    # data = dataset.generate(sampling_rate=sampling_rate)
    
    sampling_rate = 1000
    input_data = '/home/tiennv/Github/UTSA_Coursework/Spring2024/EE5163_DPS/Project/data/ibm_stock_2022_Jan2March.csv'
    num_samples = 2000
    dataset_generator = IBMDataset(data_file=input_data)
    data = dataset_generator.generate(num_samples=num_samples)
    
    if __debug__:
        # Plot the dataset
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
        ax.plot(data['t'], data['x_t'], 'g-', label='Noise data')
        ax.set_xlabel("t (seconds)")
        ax.set_ylabel("f(t)")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.legend(loc='best')
        plt.show()
        fig.savefig('./output/demo/ibm_dataset/noise_signal.pdf', bbox_inches='tight')

    threshold_ratio = 5e-4
    print("Threshold: {}".format(threshold_ratio))
    denoiser = FourierTransformDenoiser(threshold_ratio=threshold_ratio)
    output = denoiser.run(input_signal=data['x_t'], sampling_rate=sampling_rate)
    org_s2n = signal2noise(frequency_signal=output['X_f'].real)
    print("Original signal to noise ratio: {}".format(org_s2n))
    clean_s2n = signal2noise(frequency_signal=output['clean_Xf'].real)
    print("Clean signal to noise ratio: {}".format(clean_s2n))
    improvement_percent = (org_s2n-clean_s2n)/org_s2n*100
    print("Percentage of improvement: {}".format(improvement_percent))

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
        fig.savefig('./output/demo/ibm_dataset/fft_output.pdf', bbox_inches='tight')

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
        ax.stem(output['freq_oneside'],  abs(output['clean_Xf_oneside'].real)/np.abs(np.max(output['clean_Xf_oneside'].real)), 'b', markerfmt=" ", basefmt="-b")
        ax.set_xlabel(r'$\omega$ (Hz)')
        ax.set_ylabel(r'$|\hat{g}(\omega)$|')
        ax.set_yscale('log')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.show()
        fig.savefig('./output/demo/ibm_dataset/thresholding_output.pdf', bbox_inches='tight')

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
        ax.plot(data['t'], output['clean_xt'].real, 'r-', label="Denoising signal")
        ax.set_xlabel(r'$t (seconds)$')
        ax.set_ylabel(r'$\hat{f}(t)$')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.show()
        fig.savefig('./output/demo/ibm_dataset/ifft_output.pdf', bbox_inches='tight')

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
        ax.plot(data['t'], data['x_t'], 'g-', label='Noise data', alpha=0.4)
        ax.plot(data['t'], output['clean_xt'].real, 'r-', label="Removal noise signal")
        ax.set_xlabel(r'$t$ (seconds)')
        ax.set_ylabel(r'$\hat{f}(t)$')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.legend(loc='best')
        plt.show()
        fig.savefig('./output/demo/ibm_dataset/ifft_output_and_noise_signal_and_clean_signal.pdf', bbox_inches='tight')