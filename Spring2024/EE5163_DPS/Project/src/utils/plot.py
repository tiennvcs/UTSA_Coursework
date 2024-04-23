import os
from matplotlib import pyplot as plt
import numpy as np


def plot_res(X: np.ndarray, y: np.ndarray, c: float, x_label: str, y_label: str, output_file):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
    ax.plot(X, y, 'go-', label="After denoising")
    ax.axhline(y=c, color = 'r', linestyle = '-', label="Before denoising") 
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    plt.show()
    fig.savefig(output_file, bbox_inches='tight')


def plot_progress(data, output, output_dir):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    ax.plot(data['t'], data['x_t'], 'g-', label='Noise data')
    ax.plot(data['t'], data['true_xt'], 'b-', label='Clean data')
    ax.set_xlabel("t (seconds)")
    ax.set_ylabel("f(t)")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.legend(loc='best')
    plt.show()
    fig.savefig(os.path.join(output, 'noise_signal_versus_true_signal.pdf'), bbox_inches='tight')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    ax.stem(output['freq_oneside'], abs(output['Xf_oneside']), 'b', markerfmt=" ", basefmt="-b")
    # ax.axhline(y=threshold_ratio*np.max(output['freq_oneside']), color = 'r', linestyle = '--')
    ax.set_xlabel(r'$\omega$ (Hz)')
    ax.set_ylabel(r'$|g(\omega)|$')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()
    fig.savefig(os.path.join(output_dir, 'fft_output.pdf'), bbox_inches='tight')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    ax.stem(output['freq_oneside'], abs(output['clean_Xf_oneside']), 'b', markerfmt=" ", basefmt="-b")
    ax.set_xlabel(r'$\omega$ (Hz)')
    ax.set_ylabel(r'$|\hat{g}(\omega)$|')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.show()
    fig.savefig(os.path.join(output_dir, './output/thresholding_output.pdf'), bbox_inches='tight')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    ax.plot(data['t'], output['clean_xt'].real, 'r-', label="Denoising signal")
    ax.set_xlabel(r'$t (seconds)$')
    ax.set_ylabel(r'$\hat{f}(t)$')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()
    fig.savefig(os.path.join(output_dir, './output/ifft_output.pdf'), bbox_inches='tight')

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
    fig.savefig(os.path.join(output_dir, './output/ifft_output_and_noise_signal_and_clean_signal.pdf'), bbox_inches='tight')