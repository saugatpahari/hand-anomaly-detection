"""
Hand Anomaly Detection — the classifier this project was always missing.

WHAT THIS SCRIPT DOES: trains and cross-validates a real classifier on the
actual recordings, for three feature sets (Y-axis, X-axis, Combined), and
reports honest, freshly-computed accuracy and per-fold precision/recall —
replacing the "previously-validated numbers" the write-up has been relying
on with numbers this script actually reproduces, right now.

DATA STRUCTURE (confirmed by inspecting the real files): there are FOUR
recording folders, not two. ycorrect/yfalse (10 recordings) and
xcorrect/xfalse (10 recordings) are two SEPARATE recording campaigns
-- this matches the write-up's "Y-axis" and "X-axis" results rows, which
were never actually about a single axis of one dataset, but about two
different recording sessions. "Combined" = all 20 recordings together.

METHODOLOGY DECISION (this is new, and is the actual answer to the write-up's
open question about windowing): each recording is ~3,000-3,200 raw samples
long. Rather than treat one recording as one training example (only 10-20
examples total is too little for meaningful cross-validation), this script
splits each recording into non-overlapping 100-sample windows (the same
window size already used for FFT features elsewhere in this project), and
extracts one 15-number feature vector per window (mean, variance, kurtosis,
skew, median absolute deviation -- each logged channel, the same features
already used in the EDA script). This is why grouped cross-validation
matters: many windows come from the same recording, so a group
(GroupKFold, grouped by source recording) ensures no recording's windows
appear on both sides of a train/test split -- otherwise the model could
"recognize" a recording it has partially already seen, inflating accuracy.

CORRECTION (2026-08-17): earlier comments in this file described the 3
logged channels as raw X/Y/Z acceleration at 200Hz, and the 100-sample
window as "0.5s". Reading the actual firmware (hand_imu_logger.ino,
included in this repo) and checking it against the real data showed both
were wrong. The firmware logs per-sample roll/pitch/tilt angle in degrees
(computed from the accelerometer via atan(), not raw acceleration) --
confirmed empirically, real values fall in roughly -90..90, the signature
of degrees, not m/s^2. And the firmware's loop has an explicit 10ms delay
plus I2C/serial overhead, making 200Hz physically impossible -- the real
rate is more likely 55-80Hz (no timestamp was logged, so this is an
estimate from the code, not a measurement), putting each 100-sample window
at roughly 1.25-1.8s of real time, not 0.5s. None of this changes anything
below -- the pipeline treats the recording as a 3-channel time series
regardless of what the channels represent or the exact rate -- it only
corrects how the signal should be described.
"""

import os
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

DATASET_ROOT = 'handDatasets'
WINDOW_SIZE = 100  # ~1.25-1.8s at the firmware's real ~55-80Hz rate (see module docstring); matches the FFT window used elsewhere
RANDOM_SEED = 42


def list_files(folder):
    d = os.path.join(DATASET_ROOT, folder)
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith('.csv'))


def windows_from_recording(filepath, window_size):
    """Split one recording into non-overlapping windows, dropping any
    leftover tail shorter than a full window."""
    data = np.genfromtxt(filepath, delimiter=',')
    n_windows = data.shape[0] // window_size
    return [data[i * window_size:(i + 1) * window_size] for i in range(n_windows)]


def build_feature_vector(window):
    """Same feature family as the EDA script's build_feature_vector, applied
    per-window instead of per-whole-recording.

    Real-data edge case found while running this: a handful of windows have
    one axis reading a constant 0.0 for the entire 0.5s window (a sensor
    dropout, not a real "the hand didn't move" moment - the other two axes
    are moving normally in these same windows). Kurtosis/skew are undefined
    (0/0) for a zero-variance signal, which produces NaN. Rather than drop
    those windows and quietly shrink the dataset, NaN is replaced with 0 -
    a flat signal has no meaningful "shape" to its distribution, so 0 is the
    honest value, not a fabricated one."""
    mean = np.mean(window, axis=0)
    var = np.var(window, axis=0)
    kurt = np.nan_to_num(stats.kurtosis(window), nan=0.0)
    skew = np.nan_to_num(stats.skew(window), nan=0.0)
    mad = stats.median_abs_deviation(window)
    return np.concatenate([mean, var, kurt, skew, mad])  # length 15


def build_dataset(normal_folders, anomaly_folders):
    """Returns X (features), y (0=normal/1=anomaly), groups (recording id)."""
    X, y, groups = [], [], []
    recording_id = 0
    for folder in normal_folders:
        for filepath in list_files(folder):
            for w in windows_from_recording(filepath, WINDOW_SIZE):
                X.append(build_feature_vector(w))
                y.append(0)
                groups.append(recording_id)
            recording_id += 1
    for folder in anomaly_folders:
        for filepath in list_files(folder):
            for w in windows_from_recording(filepath, WINDOW_SIZE):
                X.append(build_feature_vector(w))
                y.append(1)
                groups.append(recording_id)
            recording_id += 1
    return np.array(X), np.array(y), np.array(groups), recording_id


def evaluate(X, y, groups, n_splits=5):
    """Grouped k-fold CV for each of 3 models. Returns dict of
    model_name -> (mean_accuracy, mean_precision, mean_recall)."""
    results = {}
    models = {
        'Random Forest': lambda: RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED),
        'Logistic Regression': lambda: LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
        'SVM (RBF)': lambda: SVC(kernel='rbf', random_state=RANDOM_SEED),
    }
    n_splits = min(n_splits, len(np.unique(groups)))
    # StratifiedGroupKFold, not plain GroupKFold: plain GroupKFold doesn't
    # balance classes across folds, and with only 10-20 groups it produced
    # folds that were nearly all one class - high accuracy (easy to predict
    # the majority class right) but misleadingly low precision/recall
    # (undefined/near-zero on folds with almost no positive examples to
    # test against). Stratifying by class while still respecting groups
    # (no recording split across train/test) fixes this and gives numbers
    # that actually reflect real class-balanced performance.
    gkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    for name, make_model in models.items():
        fold_acc, fold_prec, fold_rec = [], [], []
        for train_idx, test_idx in gkf.split(X, y, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model = make_model()
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)

            fold_acc.append(accuracy_score(y_test, preds))
            precision, recall, _, _ = precision_recall_fscore_support(
                y_test, preds, average='binary', zero_division=0)
            fold_prec.append(precision)
            fold_rec.append(recall)

        results[name] = (np.mean(fold_acc), np.mean(fold_prec), np.mean(fold_rec),
                          np.std(fold_acc))
    return results


def run_configuration(config_name, normal_folders, anomaly_folders):
    X, y, groups, n_recordings = build_dataset(normal_folders, anomaly_folders)
    print(f"\n{'=' * 60}")
    print(f"{config_name}")
    print(f"{'=' * 60}")
    print(f"Source recordings: {n_recordings}  |  Windows (training examples): {len(X)}")
    print(f"Window size: {WINDOW_SIZE} samples (~1.25-1.8s @ the firmware's real ~55-80Hz rate)")

    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups)
    results = evaluate(X, y, groups, n_splits=n_splits)

    print(f"Grouped {n_splits}-fold CV (grouped by source recording):")
    best_model, best_acc = None, -1
    for name, (acc, prec, rec, std) in results.items():
        print(f"  {name:22s} accuracy={acc:.3f} (+/-{std:.3f})  precision={prec:.3f}  recall={rec:.3f}")
        if acc > best_acc:
            best_acc, best_model = acc, name
    print(f"Best: {best_model} at {best_acc:.3f}")
    return config_name, best_model, best_acc, results


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)

    summary = []
    summary.append(run_configuration("Y-axis (ycorrect vs yfalse)", ['ycorrect'], ['yfalse']))
    summary.append(run_configuration("X-axis (xcorrect vs xfalse)", ['xcorrect'], ['xfalse']))
    summary.append(run_configuration("Combined (all 4 folders)", ['ycorrect', 'xcorrect'], ['yfalse', 'xfalse']))

    print(f"\n{'=' * 60}")
    print("SUMMARY (compare against the write-up's previously-validated table)")
    print(f"{'=' * 60}")
    for config_name, best_model, best_acc, _ in summary:
        print(f"  {config_name:35s} {best_model:22s} {best_acc:.1%}")
