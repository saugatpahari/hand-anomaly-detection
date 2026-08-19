"""
Hand Anomaly Detection — Exploratory Data Analysis (EDA) script.

WHAT THIS SCRIPT IS: a data-exploration/visualization tool. It loads the raw
IMU recordings, plots what "normal" vs "anomaly" movement looks like, and
computes summary statistics to see whether the two classes are actually
distinguishable before you go build a classifier.

SIGNAL CORRECTION (2026-08-17): the three logged channels are per-sample
roll/pitch/tilt angle in degrees (computed from the accelerometer via
atan(), see hand_imu_logger.ino in this repo) -- not raw X/Y/Z
acceleration as earlier comments in this file described. Confirmed
empirically against the real data: values fall in roughly -90..90, the
signature of degrees, not m/s^2. Doesn't affect any plot or statistic
below -- they all just treat this as a 3-channel signal -- only the
physical description of what's being plotted changes.

WHAT THIS SCRIPT IS NOT: a trained model. Nothing here fits a classifier,
computes accuracy, or makes a prediction. That's a separate script (still
missing) that would import something like RandomForestClassifier,
LogisticRegression, and GroupKFold from sklearn.

HOW THE PLOTS ARE ORGANIZED, and why: each figure below has one specific job.
  1. plot_raw_signal_grid - "what does one example recording of each class
     look like, over time, on each axis?" Illustrative only.
  2. save_fft_plots - "what does the frequency content look like, averaged
     across recordings, on each axis?"
  3. plot_statistics_grid - "across ALL recordings, does any single summary
     statistic (mean, variance, kurtosis, skew, MAD, correlation) separate
     the two classes?" One consolidated grid so every statistic can be
     compared side by side instead of paging through separate figures.
  4. plot_pca_feature_space - "combining everything at once (all 15
     features together, the way a real classifier would see them), does
     the combined feature space look separable?" The capstone plot.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy as sp

# --------------------------------------------------------------------------
# SETTINGS
# --------------------------------------------------------------------------
dataset_path = 'handDatasets'
normal_op_list = ['ycorrect']
anomaly_op_list = ['yfalse']
# sample_rate/sample_time below are the nominal design values used to derive
# the 100-sample FFT window size (kept as-is so this matches the same
# 100-sample window train_classifier.py uses). The firmware's REAL rate is
# not 200Hz -- see the correction note above and hand_imu_logger.ino's
# explicit 10ms delay -- realistically ~55-80Hz, making each 100-sample
# window closer to ~1.25-1.8s of real time than the "0.5s" these two
# numbers imply.
sample_rate = 200
sample_time = 0.5

NUM_SAMPLES_ILLUSTRATIVE = 5
RANDOM_SEED = 42

# Every figure this script produces is also saved here automatically (in
# addition to being shown on screen), so there's one consistent, named copy
# ready to drop into the GitHub repo without having to manually save each
# matplotlib window.
FIGURES_DIR = 'figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

samples_per_file = 200
max_measurements = int(sample_time * sample_rate)
print('Max measurements per file:', max_measurements)


def create_filename_list(op_list):
    op_filenames = []
    for target in op_list:
        samples_in_dir = os.listdir(os.path.join(dataset_path, target))
        samples_in_dir = [os.path.join(dataset_path, target, sample) for sample in samples_in_dir]
        op_filenames.extend(samples_in_dir)
    return op_filenames


def plot_raw_signal_grid(normal_samples, anomaly_samples, num_to_show=1, window_samples=400,
                          title='Raw Signal, by Axis and Class'):
    """
    Shows what one representative recording of each class looks like, axis
    by axis, so a reader can see the shape of the movement before looking
    at any summary statistic.

    Layout: rows = class (Normal / Anomaly), columns = axis (X / Y / Z).
    Splitting by axis gives each column its own y-scale, so an axis with
    small amplitude (e.g. Y) isn't visually flattened by an axis with large
    amplitude (e.g. Z) sharing the same scale. Splitting by class as well
    means a panel only ever has one color's line in it.

    Only one recording is drawn per class by default. Showing two
    recordings in the same class color doesn't let you tell them apart as
    separate recordings - it just reads as a thicker/noisier single line.
    Recording-to-recording spread within a class is handled properly by
    plot_statistics_grid instead, where every recording is its own
    distinguishable point.

    Only the first `window_samples` samples of the recording are plotted
    (default 400), not the full file. Real recordings here run to several
    thousand raw samples - squeezing that many points into a chart a few
    hundred pixels wide doesn't render as a clean line, it renders as a
    dense blur, simply because there are more data points than pixels to
    place them at. The title always states exactly how many samples are
    shown out of how many exist, so this is a visible choice, not a silent
    truncation. Every actual statistic elsewhere in this script (mean,
    variance, FFT, etc.) is still computed from the complete, untruncated
    file - only this one illustrative plot is windowed.
    """
    axis_labels = ['X', 'Y', 'Z']
    class_rows = [('Normal', normal_samples, 'blue'), ('Anomaly', anomaly_samples, 'red')]
    n_show = min(num_to_show, len(normal_samples), len(anomaly_samples))
    full_length = normal_samples[0].shape[0]
    n_window = min(window_samples, full_length)

    fig, axs = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey='col')
    recording_note = 'one representative recording per class' if n_show == 1 \
        else f'{n_show} of {len(normal_samples)} recordings per class'
    fig.suptitle(f'{title}\n(first {n_window} of {full_length} samples shown, '
                 f'{recording_note})', fontsize=11)

    for row_idx, (class_name, samples, color) in enumerate(class_rows):
        for col_idx, axis_label in enumerate(axis_labels):
            ax = axs[row_idx, col_idx]
            for i in range(n_show):
                windowed = samples[i].T[col_idx][:n_window]
                ax.plot(windowed, color=color, linewidth=1.3, alpha=0.85)
            if row_idx == 0:
                ax.set_title(axis_label)
            if col_idx == 0:
                ax.set_ylabel(f'{class_name}\nG-force')
            if row_idx == 1:
                ax.set_xlabel('Sample')

    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    fig.savefig(os.path.join(FIGURES_DIR, 'raw_signal_by_axis_and_class.png'), dpi=150, bbox_inches='tight')
    plt.show()


def extract_fft_features(sample, max_measurements):
    if sample.shape[0] > max_measurements:
        sample = sample[:max_measurements]
    elif sample.shape[0] < max_measurements:
        padding = max_measurements - sample.shape[0]
        sample = np.pad(sample, ((0, padding), (0, 0)), mode='constant')
    hann_window = np.hanning(sample.shape[0])
    out_sample = np.zeros((max_measurements // 2, sample.shape[1]))
    for i, axis in enumerate(sample.T):
        fft = abs(np.fft.rfft(axis * hann_window))
        out_sample[:, i] = fft[1:]
    return out_sample


def save_fft_plots(normal_fft_avg, anomaly_fft_avg):
    """Frequency content (via FFT, Hann-windowed to reduce spectral leakage),
    averaged across recordings, one panel per axis."""
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
    fig.savefig(os.path.join(FIGURES_DIR, 'fft_spectrum_by_axis.png'), dpi=150, bbox_inches='tight')
    plt.show()


def load_all_samples(filenames):
    return [np.genfromtxt(f, delimiter=',') for f in filenames]


def plot_statistics_grid(stats_to_plot):
    """
    One consolidated figure covering every per-recording summary statistic
    (Mean, Variance, Kurtosis, Skew, MAD, Correlation) instead of a separate
    pop-up per statistic. Rows are statistics, columns are axes (or
    axis-pairs for Correlation) - scan down a column to see whether a given
    axis separates the classes on any statistic, or across a row to see
    whether a given statistic separates the classes on any axis.

    Each individual panel is a strip plot: every real recording shown as
    its own point at its true value on the x-axis, with a small vertical
    jitter purely so overlapping points stay visible (the y-position itself
    carries no meaning - only the jitter band, Normal vs Anomaly, does).
    With only 5 recordings per class, a strip plot shows every real data
    point directly rather than summarizing into a histogram or density
    curve that would imply more data than actually exists.

    `stats_to_plot` is a list of (stat_name, normal_values, anomaly_values,
    axis_labels) tuples, one entry per row of the grid.
    """
    n_rows = len(stats_to_plot)
    fig, axs = plt.subplots(n_rows, 3, figsize=(12, 2.5 * n_rows))
    fig.suptitle('Per-Recording Statistics, Normal vs Anomaly', fontsize=14)

    for row_idx, (stat_name, normal_values, anomaly_values, axis_labels) in enumerate(stats_to_plot):
        normal_arr = np.array(normal_values)
        anomaly_arr = np.array(anomaly_values)
        for col_idx, axis_label in enumerate(axis_labels):
            ax = axs[row_idx, col_idx]
            rng = np.random.default_rng(RANDOM_SEED + row_idx * 3 + col_idx)
            ax.scatter(normal_arr[:, col_idx], rng.normal(0, 0.02, len(normal_arr)),
                       color='blue', label='Normal', zorder=3, s=45, alpha=0.8)
            ax.scatter(anomaly_arr[:, col_idx], rng.normal(0.3, 0.02, len(anomaly_arr)),
                       color='red', label='Anomaly', zorder=3, s=45, alpha=0.8)
            ax.set_yticks([])
            ax.axhline(0.15, color='gray', linewidth=0.5, zorder=1)
            ax.set_title(axis_label, fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(stat_name, fontsize=11)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.985), ncol=1)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIGURES_DIR, 'per_recording_statistics_grid.png'), dpi=150, bbox_inches='tight')
    plt.show()


def build_feature_vector(s):
    """Combine every per-recording statistic computed elsewhere in this
    script (mean, variance, kurtosis, skew, MAD - each logged channel) into ONE
    15-number feature vector. This is the same kind of feature engineering
    a real classifier would use - it just gathers what's already calculated
    into the shape a model would actually see it in."""
    mean = np.mean(s, axis=0)
    var = np.var(s, axis=0)
    kurt = stats.kurtosis(s)
    skew = stats.skew(s)
    mad = stats.median_abs_deviation(s)
    return np.concatenate([mean, var, kurt, skew, mad])  # length 15


def plot_pca_feature_space(normal_full, anomaly_full):
    """
    The capstone EDA plot. Combines every recording's full feature vector
    (all 15 numbers from build_feature_vector) and projects it down to 2
    dimensions with PCA, so you can see - in one picture - roughly how
    separable normal and anomaly recordings are in the SAME kind of
    combined feature space a real classifier (RF/LogReg) would actually be
    trained on. Every earlier plot in this script looks at one statistic at
    a time; this is the one that previews "does this classification problem
    look learnable" using all of them together.

    Honest caveat to state alongside this plot: with only ~10 total
    recordings, 2 principal components summarizing 15 features is a
    significant compression - treat this as a rough sanity check on
    separability, not proof of it. The real answer comes from the actual
    classifier's cross-validated accuracy, not from this picture.
    """
    normal_features = np.array([build_feature_vector(s) for s in normal_full])
    anomaly_features = np.array([build_feature_vector(s) for s in anomaly_full])
    all_features = np.vstack([normal_features, anomaly_features])
    labels = np.array(['Normal'] * len(normal_features) + ['Anomaly'] * len(anomaly_features))

    # Standardize BEFORE PCA - essential here, since raw variance is in the
    # hundreds/thousands while raw skew is a small decimal. Without this,
    # PCA would just be "which feature has the biggest numbers," not a fair
    # combination of all 15 features.
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(all_features)

    # PCA can't produce more components than you have samples - with n=10
    # total recordings, 2 components is safely within range, but this guard
    # keeps the code from breaking if the dataset ever gets even smaller.
    n_components = min(2, len(all_features) - 1)
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(scaled_features)
    explained = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    for label, color in [('Normal', 'blue'), ('Anomaly', 'red')]:
        mask = labels == label
        y_vals = projected[mask, 1] if n_components > 1 else np.zeros(mask.sum())
        ax.scatter(projected[mask, 0], y_vals, color=color, label=label, s=90, alpha=0.85, zorder=3)

    ax.axhline(0, color='gray', linewidth=0.5, zorder=1)
    ax.axvline(0, color='gray', linewidth=0.5, zorder=1)
    ax.set_xlabel(f'PC1 ({explained[0]:.1f}% of variance)')
    ax.set_ylabel(f'PC2 ({explained[1]:.1f}% of variance)' if n_components > 1 else '(too few samples for a 2nd component)')
    ax.set_title(f'PCA of Combined Feature Space (n={len(all_features)} recordings total)')
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'pca_feature_space.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print(f"PCA explained variance: PC1={explained[0]:.1f}%"
          + (f", PC2={explained[1]:.1f}%" if n_components > 1 else "")
          + f"  (n={len(all_features)} total recordings - treat as a rough separability check, not proof)")


def main():
    random.seed(RANDOM_SEED)

    normal_op_filenames = create_filename_list(normal_op_list)
    anomaly_op_filenames = create_filename_list(anomaly_op_list)
    print('Number of normal samples:', len(normal_op_filenames))
    print('Number of anomaly samples:', len(anomaly_op_filenames))

    random.shuffle(normal_op_filenames)
    random.shuffle(anomaly_op_filenames)

    num_samples = NUM_SAMPLES_ILLUSTRATIVE
    normal_samples = []
    anomaly_samples = []
    for i in range(num_samples):
        normal_samples.append(np.genfromtxt(normal_op_filenames[i], delimiter=','))
        anomaly_samples.append(np.genfromtxt(anomaly_op_filenames[i], delimiter=','))
    plot_raw_signal_grid(normal_samples, anomaly_samples)
    plt.show()

    normal_ffts = []
    anomaly_ffts = []
    for i in range(NUM_SAMPLES_ILLUSTRATIVE):
        normal_sample = np.genfromtxt(normal_op_filenames[i], delimiter=',', max_rows=max_measurements)
        anomaly_sample = np.genfromtxt(anomaly_op_filenames[i], delimiter=',', max_rows=max_measurements)
        normal_fft = extract_fft_features(normal_sample, max_measurements)
        anomaly_fft = extract_fft_features(anomaly_sample, max_measurements)
        normal_ffts.append(normal_fft)
        anomaly_ffts.append(anomaly_fft)
    normal_ffts = np.array(normal_ffts)
    anomaly_ffts = np.array(anomaly_ffts)
    normal_fft_avg = np.average(normal_ffts, axis=0)
    anomaly_fft_avg = np.average(anomaly_ffts, axis=0)
    save_fft_plots(normal_fft_avg, anomaly_fft_avg)
    plt.show()

    n_stat = min(len(normal_op_filenames), len(anomaly_op_filenames))
    print(f"Using {n_stat} samples per class for the statistical comparisons below "
          f"(full available data, paired by index after shuffle).")

    normal_full = load_all_samples(normal_op_filenames[:n_stat])
    anomaly_full = load_all_samples(anomaly_op_filenames[:n_stat])

    normal_means = [np.mean(s, axis=0) for s in normal_full]
    anomaly_means = [np.mean(s, axis=0) for s in anomaly_full]

    normal_variances = [np.var(s, axis=0) for s in normal_full]
    anomaly_variances = [np.var(s, axis=0) for s in anomaly_full]

    normal_kurtosis = [stats.kurtosis(s) for s in normal_full]
    anomaly_kurtosis = [stats.kurtosis(s) for s in anomaly_full]

    normal_skew = [stats.skew(s) for s in normal_full]
    anomaly_skew = [stats.skew(s) for s in anomaly_full]

    normal_mad = [stats.median_abs_deviation(s) for s in normal_full]
    anomaly_mad = [stats.median_abs_deviation(s) for s in anomaly_full]

    normal_corr_pairs = []
    for s in normal_full:
        centered = s - np.mean(s, axis=0)
        c = np.corrcoef(centered.T)
        normal_corr_pairs.append([c[0, 1], c[0, 2], c[1, 2]])
    anomaly_corr_pairs = []
    for s in anomaly_full:
        centered = s - np.mean(s, axis=0)
        c = np.corrcoef(centered.T)
        anomaly_corr_pairs.append([c[0, 1], c[0, 2], c[1, 2]])

    plot_statistics_grid([
        ('Mean', normal_means, anomaly_means, ('X', 'Y', 'Z')),
        ('Variance', normal_variances, anomaly_variances, ('X', 'Y', 'Z')),
        ('Kurtosis', normal_kurtosis, anomaly_kurtosis, ('X', 'Y', 'Z')),
        ('Skew', normal_skew, anomaly_skew, ('X', 'Y', 'Z')),
        ('MAD', normal_mad, anomaly_mad, ('X', 'Y', 'Z')),
        ('Correlation', normal_corr_pairs, anomaly_corr_pairs, ('X-Y', 'X-Z', 'Y-Z')),
    ])

    # All features combined into one feature vector per recording, projected to 2D.
    plot_pca_feature_space(normal_full, anomaly_full)


if __name__ == "__main__":
    main()
