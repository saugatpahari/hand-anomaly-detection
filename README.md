# Hand Anomaly Detection

Detecting simulated hand-movement anomalies from a wrist-worn IMU, using engineered statistical features over windowed segments and stratified, grouped cross-validation.

## What this is

A small-sample case study: can a classifier tell apart "normal" (free, unscripted) hand movement from "anomaly" (a deliberately performed, rehearsed simulation of impaired movement), using only a wrist-worn IMU?

**Important scope note:** "anomaly" here means a rehearsed simulation of impaired movement, not real clinical impairment. This is a methodology/feasibility study, not a medical device or clinical claim.

## Hardware and what's actually logged

ESP32 + MPU6050 (a 3-axis accelerometer + gyroscope IMU), Bluetooth to Android. Firmware: `hand_imu_logger.ino` (included in this repo).

**Correction (2026-08-17):** earlier drafts of this project's docs described the logged signal as raw 3-axis acceleration at 200 Hz. Reading the actual firmware and checking it against the real data showed both of those were wrong:

- **The three logged channels are roll/pitch/tilt angle in degrees, not raw acceleration.** The firmware computes each angle per-sample from the accelerometer via `atan(...)` (a standard accelerometer-only tilt estimate — no gyroscope fusion, no filtering across samples) and logs *that*, not the raw `a.acceleration.x/y/z` values. Confirmed empirically: real recorded values fall in roughly the -90 to +90 range, the signature of degrees from an arctangent calculation, not m/s² (which would center near ±9.8 at rest on an 8G-range accelerometer).
- **The real sampling rate is well under 200 Hz.** The logging loop has an explicit 10 ms delay plus I2C read and serial-print overhead on top of that, which makes 200 Hz physically impossible with this code — realistically it runs somewhere in the 55-80 Hz range. No timestamp column was logged, so this is an estimate from the firmware's timing, not a measured value.

This doesn't change any of the results below — the classifier and its cross-validation treat the recording as a 3-channel time series regardless of what the channels physically represent, and none of the pipeline logic depends on the sample rate being exactly 200 Hz. It only changes how the signal and window durations should be described (see Method).

## Data

20 total recordings across two separate recording campaigns (10 each: 5 normal / 5 anomaly per campaign). Each recording is ~3,000-3,200 raw samples. At the corrected ~55-80 Hz estimate that's roughly 40-60 seconds per recording, not the ~15-16 seconds stated in earlier drafts (which assumed 200 Hz). Small by design (a personal side project, not a funded data collection effort) — the honest caveat section below exists because of this.

## Method

- **Windowing:** each recording split into non-overlapping 100-sample windows, giving ~590 training examples from the 20 source recordings. At the corrected ~55-80 Hz rate, each window covers roughly 1.25-1.8 seconds of real time, not the 0.5 seconds stated in earlier drafts.
- **Features:** per window — mean, variance, kurtosis, skew, median absolute deviation, each computed per logged channel (roll/pitch/tilt angle, not raw X/Y/Z acceleration — see Hardware section above). 15 numbers per window.
- **Models:** Random Forest, Logistic Regression, and an RBF-kernel SVM, compared per configuration.
- **Validation:** stratified grouped 5-fold cross-validation — grouped by source recording (no recording's windows appear on both sides of a split) and stratified by class (ensures folds aren't accidentally near-single-class, which would make precision/recall misleading even with correct grouping).

## Results

| Configuration | Best model | Accuracy | Precision | Recall |
|---|---|---|---|---|
| Campaign 1 (10 recordings) | Random Forest | 99.7% | 1.000 | 0.993 |
| Campaign 2 (10 recordings) | Random Forest | 94.8% | 0.948 | 0.960 |
| Combined (20 recordings) | SVM (RBF) | 96.3% | 0.950 | 0.980 |

Fully reproducible — run `train_classifier.py` against `handDatasets/` to regenerate these numbers directly.

**Caveat (read this before the table above):** validation is grouped by 20 independent source recordings, even though there are hundreds of individual windows. Every window from one recording shares that recording's calibration and specific rehearsed-gesture instance, so while window-level leakage is correctly prevented, the real ceiling on generalization confidence is set by "20 independent sessions," not "hundreds of examples." These results show the feature set clearly separates rehearsed normal vs. rehearsed anomaly movement *in this sample* — not that this would hold for a new person, or for real (not rehearsed) impairment.

## Repository structure

- `hand_imu_logger.ino` — the actual firmware that ran on the ESP32 to collect this dataset. Adapted from Adafruit's stock MPU6050 example sketch (kept the sensor init/range/filter setup, changed the loop to compute and log per-sample roll/pitch/tilt angle instead of raw acceleration or the example's debug prints). Included as-is, imperfections and all — see the note in Hardware above and the caveats below.
- `hand_anomaly_model.py` — exploratory data analysis: loads recordings, visualizes normal vs. anomaly signal characteristics, computes and visualizes per-recording statistics, and previews class separability via PCA before training. Does not train a classifier.
- `train_classifier.py` — the actual classifier: windowing, feature extraction, and stratified grouped cross-validation across Random Forest / Logistic Regression / SVM. Produces the results table above.

## Known firmware limitations (honest, not hidden)

Found while reviewing `hand_imu_logger.ino` for this README:
- No timestamp is logged per row, so the real sample rate can only be estimated from the code's timing (see Hardware above), not measured directly from the data.
- The angle calculation is accelerometer-only — no gyroscope fusion (e.g. a complementary or Kalman filter), so it's a noisier per-sample tilt estimate, not a smoothed orientation.
- The raw accelerometer readings are stored in an `int16_t` before being used in the angle calculation, truncating their fractional part before the `atan()` call — a real precision loss on small values, not a design choice. Doesn't invalidate the results (the classifier still separates the classes on whatever signal came out), but worth knowing if extending this firmware.

None of this needed fixing retroactively — the firmware included here is exactly what produced the real dataset the results above are computed from, imperfections included.

## Status / what's next

- Original wearable (gloves) was lost mid-project; a natural next step is exploring webcam-based hand-landmark tracking as an additional, hardware-free sensing modality.
- More source recordings would meaningfully shrink the "20 independent sessions" ceiling noted in the caveat above.
- Leave-one-recording-out cross-validation would stress-test the single worst-case held-out recording rather than an averaged fold result.
- A revised firmware logging raw acceleration (or both raw + derived angle) with a timestamp column and gyro-fused orientation would be a meaningful upgrade for any future data collection.
