import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy import stats
from sklearn.decomposition import PCA
import scipy as sp

# Settings - Set Dataset Path for normal and anomaly samples
dataset_path = 'handDatasets'
normal_op_list = ['ycorrect']
anomaly_op_list = ['yfalse']

sample_rate = 200  # Hz
sample_time = 0.5  # Time (sec) length of each sample
NUM_SAMPLES = 5
samples_per_file = 200  # Expected number of measurements in each file
max_measurements = int(sample_time * sample_rate)

print('Max measurements per file:', max_measurements)


def create_filename_list(op_list):
    """Extract paths and filenames in each directory."""
    op_filenames = []
    for target in op_list:
        samples_in_dir = os.listdir(os.path.join(dataset_path, target))
        samples_in_dir = [os.path.join(dataset_path, target, sample) for sample in samples_in_dir]
        op_filenames.extend(samples_in_dir)
    return op_filenames


def plot_time_series_sample(normal_sample, anomaly_sample):
    """Plot normal vs anomaly samples side-by-side."""
    fig, axs = plt.subplots(2, 1, figsize=(6, 6))
    fig.tight_layout(pad=3.0)
    axs[0].plot(normal_sample.T[0], label='x')
    axs[0].plot(normal_sample.T[1], label='y')
    axs[0].plot(normal_sample.T[2], label='z')
    axs[0].set_title('Normal sample')
    axs[0].set_xlabel('Sample')
    axs[0].set_ylabel('G-force')
    axs[0].legend()
    axs[1].plot(anomaly_sample.T[0], label='x')
    axs[1].plot(anomaly_sample.T[1], label='y')
    axs[1].plot(anomaly_sample.T[2], label='z')
    axs[1].set_title('Anomaly sample')
    axs[1].set_xlabel('Sample')
    axs[1].set_ylabel('G-force')
    axs[1].legend()
    plt.show()  # Display the plot interactively


def plot_scatter_samples(normal_samples, anomaly_samples, num_samples, title=''):
    """Plot 3D scatterplot of normal and anomaly samples."""
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for i in range(num_samples):
        ax.scatter(normal_samples[i].T[0], normal_samples[i].T[1], normal_samples[i].T[2], c='b', label='Normal' if i == 0 else "")
        ax.scatter(anomaly_samples[i].T[0], anomaly_samples[i].T[1], anomaly_samples[i].T[2], c='r', label='Anomaly' if i == 0 else "")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title)
    ax.legend()
    plt.show()  # Display the plot interactively


def extract_fft_features(sample, max_measurements):
    # Ensure sample has the desired length
    if sample.shape[0] > max_measurements:
        sample = sample[:max_measurements]  # Trim to max_measurements if longer
    elif sample.shape[0] < max_measurements:
        padding = max_measurements - sample.shape[0]  # Pad with zeros if shorter
        sample = np.pad(sample, ((0, padding), (0, 0)), mode='constant')

    # Create a window (Hann window to reduce spectral leakage)
    hann_window = np.hanning(sample.shape[0])

    # Compute FFT for each axis in the sample (exclude DC component)
    out_sample = np.zeros((max_measurements // 2, sample.shape[1]))  # Store FFT results
    for i, axis in enumerate(sample.T):  # Iterate over each axis (x, y, z)
        fft = abs(np.fft.rfft(axis * hann_window))  # Apply FFT and take magnitude
        out_sample[:, i] = fft[1:]  # Exclude the DC component (index 0)

    return out_sample


def save_fft_plots(normal_fft_avg, anomaly_fft_avg):
    """Plot FFTs and display them interactively."""
    start_bin = 1
    fig, axs = plt.subplots(3, 1, figsize=(8, 6))
    fig.tight_layout(pad=3.0)

    axs[0].plot(normal_fft_avg[start_bin:, 0], label='normal', color='blue')
    axs[0].plot(anomaly_fft_avg[start_bin:, 0], label='anomaly', color='red')
    axs[0].set_title('X')
    axs[0].set_xlabel('Bin')
    axs[0].set_ylabel('G-force')
    axs[0].legend()

    axs[1].plot(normal_fft_avg[start_bin:, 1], label='normal', color='blue')
    axs[1].plot(anomaly_fft_avg[start_bin:, 1], label='anomaly', color='red')
    axs[1].set_title('Y')
    axs[1].set_xlabel('Bin')
    axs[1].set_ylabel('G-force')
    axs[1].legend()

    axs[2].plot(normal_fft_avg[start_bin:, 2], label='normal', color='blue')
    axs[2].plot(anomaly_fft_avg[start_bin:, 2], label='anomaly', color='red')
    axs[2].set_title('Z')
    axs[2].set_xlabel('Bin')
    axs[2].set_ylabel('G-force')
    axs[2].legend()

    plt.show()  # Display the plot interactively


def main():
    # Create filename lists
    normal_op_filenames = create_filename_list(normal_op_list)
    anomaly_op_filenames = create_filename_list(anomaly_op_list)

    print('Number of normal samples:', len(normal_op_filenames))
    print('Number of anomaly samples:', len(anomaly_op_filenames))

    # Examine a normal sample vs anomalous sample
    normal_sample = np.genfromtxt(normal_op_filenames[0], delimiter=',')
    anomaly_sample = np.genfromtxt(anomaly_op_filenames[0], delimiter=',')

    # Plot time series and display
    plot_time_series_sample(normal_sample, anomaly_sample)
    plt.show()

    # Shuffle samples
    random.shuffle(normal_op_filenames)
    random.shuffle(anomaly_op_filenames)

    # Make a 3D scatterplot
    num_samples = NUM_SAMPLES
    normal_samples = []
    anomaly_samples = []
    for i in range(num_samples):
        normal_samples.append(np.genfromtxt(normal_op_filenames[i], delimiter=','))
        anomaly_samples.append(np.genfromtxt(anomaly_op_filenames[i], delimiter=','))

    plot_scatter_samples(normal_samples, anomaly_samples, num_samples, title="3D Scatterplot")
    plt.show()

    # Compute FFTs for normal and anomaly samples
    normal_ffts = []
    anomaly_ffts = []
    for i in range(NUM_SAMPLES):
        normal_sample = np.genfromtxt(normal_op_filenames[i], delimiter=',', max_rows=max_measurements)
        anomaly_sample = np.genfromtxt(anomaly_op_filenames[i], delimiter=',', max_rows=max_measurements)
        normal_fft = extract_fft_features(normal_sample, max_measurements)
        anomaly_fft = extract_fft_features(anomaly_sample, max_measurements)
        normal_ffts.append(normal_fft)
        anomaly_ffts.append(anomaly_fft)

    # Convert lists to numpy arrays
    normal_ffts = np.array(normal_ffts)
    anomaly_ffts = np.array(anomaly_ffts)

    # Average the FFTs
    normal_fft_avg = np.average(normal_ffts, axis=0)
    anomaly_fft_avg = np.average(anomaly_ffts, axis=0)

    # Save and display FFT plots
    save_fft_plots(normal_fft_avg, anomaly_fft_avg)  # Replace save_fft_plots with the plotting function
    plt.show()

    # Make a 3D scatterplot of means
    normal_means = []
    anomaly_means = []
    for i in range(NUM_SAMPLES):
        normal_sample = np.genfromtxt(normal_op_filenames[i], delimiter=',')
        anomaly_sample = np.genfromtxt(anomaly_op_filenames[i], delimiter=',')
        normal_sample = normal_sample - np.mean(normal_sample, axis=0)
        anomaly_sample = anomaly_sample - np.mean(anomaly_sample, axis=0)
        normal_means.append(np.mean(normal_sample, axis=0))
        anomaly_means.append(np.mean(anomaly_sample, axis=0))

    plot_scatter_samples(normal_means, anomaly_means, NUM_SAMPLES, title='Means')
    plt.show()

    # Make a 3D scatterplot of variances
    normal_variances = []
    anomaly_variances = []
    for i in range(NUM_SAMPLES):
        normal_sample = np.genfromtxt(normal_op_filenames[i], delimiter=',')
        anomaly_sample = np.genfromtxt(anomaly_op_filenames[i], delimiter=',')
        normal_variances.append(np.var(normal_sample, axis=0))
        anomaly_variances.append(np.var(anomaly_sample, axis=0))

    plot_scatter_samples(normal_variances, anomaly_variances, NUM_SAMPLES, title='Variances')
    plt.show()

    # Make a 3D scatterplot of kurtosis
    normal_kurtosis = []
    anomaly_kurtosis = []
    for i in range(NUM_SAMPLES):
        normal_sample = np.genfromtxt(normal_op_filenames[i], delimiter=',')
        anomaly_sample = np.genfromtxt(anomaly_op_filenames[i], delimiter=',')
        normal_kurtosis.append(stats.kurtosis(normal_sample))
        anomaly_kurtosis.append(stats.kurtosis(anomaly_sample))

    plot_scatter_samples(normal_kurtosis, anomaly_kurtosis, NUM_SAMPLES, title='Kurtosis')
    plt.show()

    # Make a 3D scatterplot of skew
    normal_skew = []
    anomaly_skew = []
    for i in range(NUM_SAMPLES):
        normal_sample = np.genfromtxt(normal_op_filenames[i], delimiter=',')
        anomaly_sample = np.genfromtxt(anomaly_op_filenames[i], delimiter=',')
        normal_skew.append(stats.skew(normal_sample))
        anomaly_skew.append(stats.skew(anomaly_sample))

    plot_scatter_samples(normal_skew, anomaly_skew, NUM_SAMPLES, title='Skew')
    plt.show()

    # Make a 3D scatterplot of MAD
    normal_mad = []
    anomaly_mad = []
    for i in range(NUM_SAMPLES):
        normal_sample = np.genfromtxt(normal_op_filenames[i], delimiter=',')
        anomaly_sample = np.genfromtxt(anomaly_op_filenames[i], delimiter=',')
        normal_mad.append(stats.median_abs_deviation(normal_sample))
        anomaly_mad.append(stats.median_abs_deviation(anomaly_sample))

    plot_scatter_samples(normal_mad, anomaly_mad, NUM_SAMPLES, title='MAD')
    plt.show()

    # Plot histograms of correlation coefficients
    n_bins = 20
    normal_corr = []
    anomaly_corr = []
    for i in range(NUM_SAMPLES):
        normal_sample = np.genfromtxt(normal_op_filenames[i], delimiter=',')
        anomaly_sample = np.genfromtxt(anomaly_op_filenames[i], delimiter=',')
        normal_sample = normal_sample - np.mean(normal_sample, axis=0)
        anomaly_sample = anomaly_sample - np.mean(anomaly_sample, axis=0)
        normal_corr.append(np.corrcoef(normal_sample.T))
        anomaly_corr.append(np.corrcoef(anomaly_sample.T))

    normal_corr = np.array(normal_corr)
    anomaly_corr = np.array(anomaly_corr)

    fig, axs = plt.subplots(3, 3, figsize=(12, 10))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    axs[0, 1].hist(normal_corr[:, 0, 1], bins=n_bins, color='blue', alpha=0.7, label='Normal')
    axs[0, 1].hist(anomaly_corr[:, 0, 1], bins=n_bins, color='red', alpha=0.7, label='Anomaly')
    axs[0, 2].hist(normal_corr[:, 0, 2], bins=n_bins, color='blue', alpha=0.7)
    axs[0, 2].hist(anomaly_corr[:, 0, 2], bins=n_bins, color='red', alpha=0.7)
    axs[1, 2].hist(normal_corr[:, 1, 2], bins=n_bins, color='blue', alpha=0.7)
    axs[1, 2].hist(anomaly_corr[:, 1, 2], bins=n_bins, color='red', alpha=0.7)
    axs[0, 1].set_title('Correlation Coefficient: X-Y')
    axs[0, 2].set_title('Correlation Coefficient: X-Z')
    axs[1, 2].set_title('Correlation Coefficient: Y-Z')
    fig.suptitle('Histograms of Correlation Coefficients')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
