# Detecting Simulated Hand-Movement Anomalies from Wearable IMU Data: A Small-Sample Case Study

*Technical write-up — AIAF Fellowship independent research artifact*
*Status: all data facts are confirmed — everything, including the results, is now independently reproducible from code in the repo. What remains is writing sections 0, 1, 6, and 7 in your own words.*

---

## 0. One-paragraph summary (write this last, in your own words — draft below to react to)

[NEED — your own 3-4 sentences, but here's a draft to start from: "I built and validated a pipeline that classifies simulated hand-movement anomalies from a wrist-worn ESP32+MPU6050 IMU sensor, using engineered statistical features over 0.5-second windows and grouped, class-stratified cross-validation to avoid leakage across recording sessions. Across two recording campaigns (20 recordings total), a Random Forest classifier reached 99.7% accuracy on one campaign and 94.8% on the other, with a combined-data SVM reaching 96.3% — all with precision and recall in the 0.93-1.00 range, not just aggregate accuracy. I treat this as a strong small-sample result on a task I deliberately kept honest in scope: these are simulated, rehearsed anomalies, not real clinical impairment."]

---

## 1. Why this project (and why it's a fit for you to talk about)

Frame this early, briefly: hardware background (ESP32/MPU6050), interest in working with real signal data rather than toy datasets, and — important for AIAF specifically — the part of this that touches your actual research interest (interpreting *why* a classifier separates two movement classes, not just reporting an accuracy number). The honest hook: this project forced you to distinguish "the model achieves high accuracy" from "the model is reliably measuring the thing I think it's measuring" — which is the same discipline interpretability work demands. Write this section in your own words; I can tighten prose but the framing should be recognizably yours.

## 2. The task and the (important) caveat

- **What "anomaly" means here:** recordings are a *deliberately performed simulation* of impaired hand movement — a rehearsed, repeated gesture — not organic/real impairment. "Normal" is genuinely free, unscripted movement.
- **Why this matters for interpreting the results:** anomaly samples are smoother/more periodic than normal samples *because they're rehearsed*, not because the model is detecting a subtle physiological signal. This is very likely part of why the classifier below performs as well as it does — naming this yourself, up front, is the difference between "this pipeline works" and "this pipeline detects real clinical impairment." Only the first claim is supported.
- **Sample size:** 20 total source recordings, across two separate recording campaigns — 10 recordings (5 normal, 5 anomaly) from one campaign, 10 more (5 normal, 5 anomaly) from a second. Confirmed directly by inspecting the actual data folders. Still a genuinely small number of independent recording *sessions* — see section 6 for why that matters even though the classifier below is trained on hundreds of windowed examples drawn from those 20 recordings.

## 3. Data & hardware

- ESP32 + MPU6050 (a 3-axis accelerometer + gyroscope IMU), Bluetooth to Android. Firmware included in the repo as `hand_imu_logger.ino`.
- **What's actually logged (corrected 2026-08-17):** I originally described this as raw 3-axis acceleration at 200 Hz. Reading the actual firmware and checking it against the real data corrected both parts of that. The three logged channels are roll/pitch/tilt angle in degrees, computed per-sample from the accelerometer via `atan(...)` — not raw acceleration, and not fused with the gyroscope across samples. I confirmed this directly against the real CSVs: values fall in roughly the -90 to +90 range, which is the signature of degrees from an arctangent, not m/s² (which would center near ±9.8 at rest). Separately, the firmware's loop has an explicit 10 ms delay plus I2C-read and serial-print overhead, which makes 200 Hz physically impossible — the real rate is more likely 55-80 Hz. No timestamp was logged, so that's an estimate from the code's timing, not a measured value.
- Each raw recording is a long, continuous file — roughly 3,000-3,200 raw samples per recording (confirmed directly from the data). At the corrected ~55-80 Hz estimate that's roughly 40-60 seconds per session, not the ~15-16 seconds I originally stated (which assumed 200 Hz).
- **Why this correction doesn't change the results:** the classifier and cross-validation in section 4-5 treat the recording as a 3-channel time series regardless of what the channels physically represent or the exact sample rate — nothing in the pipeline depends on either being correct. It only changes how the signal and window durations should be *described*, which is exactly the kind of gap worth catching and naming rather than letting an incorrect technical claim sit in a submitted document.
- **Who recorded:** most of the recordings are mine, with a few friends also contributing recordings to help build out the dataset faster. The exact split wasn't tracked at the time — worth naming as a limitation rather than glossing over, since "grouped" cross-validation groups by recording, not by person, so any variation between different people's movement patterns isn't explicitly isolated in the current validation.

## 4. Method

The classifier described here is real and reproducible — `train_classifier.py` in the repo — built after realizing the original classifier script that produced an earlier set of numbers had been lost. Rather than keep relying on unreproducible historical numbers, this section describes what the current script actually does, and section 5's results come directly from running it.

- **Windowing:** each ~3,000-3,200-sample recording is split into non-overlapping 100-sample windows — the same window size already used for FFT features in the EDA script. At the corrected ~55-80 Hz estimate (see section 3), each window covers roughly 1.25-1.8 seconds of real time, not the 0.5 seconds I originally stated. This turns 20 recordings into ~590 training examples, while keeping the *source recording* as the unit that matters for validation (see below).
- **Features:** per window, 15 numbers — mean, variance, kurtosis, skew, and median absolute deviation, each computed per logged channel (roll/pitch/tilt angle — see section 3 — not raw X/Y/Z acceleration as I originally described it).
- **A real data-quality issue found and handled:** a small number of windows have one channel reading a constant 0.0 for the full window — a sensor dropout, not a real "no movement" moment (the other two channels are moving normally in the same windows). Kurtosis/skew are mathematically undefined for a zero-variance signal (a 0/0 case), which surfaced as NaN values when this was first run. Fixed by treating a flat signal's shape statistics as 0 rather than dropping the affected windows — a flat signal genuinely has no meaningful skew or kurtosis, so 0 is the honest value, not a fabricated one. Worth mentioning in an interview if asked about debugging: this is a real example of a small data-quality bug that would have silently crashed a less-careful pipeline.
- **Models compared:** Random Forest, Logistic Regression, and an RBF-kernel SVM.
- **Validation:** stratified grouped 5-fold cross-validation — grouped by source recording (no recording's windows appear on both sides of a train/test split, preventing the model from partially recognizing a recording it's already seen) *and* stratified by class (an earlier version of this script used plain grouping without stratification, which produced folds that were nearly all one class — technically correct but statistically misleading, since precision/recall on a near-single-class test fold isn't meaningful). This is a real methodological correction made during this work, not a hypothetical — worth mentioning if asked about validation design in an interview.

## 5. Results

Two separate recording campaigns exist in the data (see section 2) — results are reported per campaign, plus a combined-data run.

| Configuration | Best model | Accuracy | Precision | Recall |
|---|---|---|---|---|
| Campaign 1 (10 recordings) | Random Forest | 99.7% | 1.000 | 0.993 |
| Campaign 2 (10 recordings) | Random Forest | 94.8% | 0.948 | 0.960 |
| Combined (20 recordings) | SVM (RBF) | 96.3% | 0.950 | 0.980 |

Every number above comes directly from running `train_classifier.py` against the real dataset — fully reproducible, not a historical figure being trusted secondhand. Per-fold accuracy standard deviation ranged from ±0.007 (Campaign 1, Random Forest) up to ±0.11 (Combined, Logistic Regression) — some folds are notably more stable than others, which is expected and worth naming rather than hiding (see section 6).

## 6. The honest caveat (this section is what makes it a *research* artifact, not a portfolio flex)

Precision and recall now track accuracy closely (0.93-1.00 across the board) — a real improvement over an earlier, less careful validation setup that produced misleadingly low precision/recall (~0.79) due to a stratification bug (see section 4). That fix is itself worth stating plainly in an interview: catching that the *validation methodology* was the problem, not the classifier, is exactly the kind of debugging judgment worth demonstrating.

That said, the honest limitation hasn't gone away, it's just more precise now: cross-validation here is grouped by only 20 independent source recordings, even though there are hundreds of individual windows. Every window from one recording shares that recording's calibration, arm position, and specific rehearsed-gesture instance — so while the *validation protocol* correctly prevents window-level leakage, the *ceiling* on how confidently these numbers generalize is still set by "20 independent sessions," not "590 independent examples." Say this directly: these results show the feature set clearly separates rehearsed normal vs. rehearsed anomaly movement *in this sample*, with more confidence than the earlier caveat implied — but they are still not evidence this would hold for a genuinely new person, or for real (not rehearsed) impairment.

## 7. What I'd do with more time / data (shows research judgment, not just execution)

- Collect more source recordings — now that a real, working classifier exists, this is the highest-leverage next step: more independent sessions directly shrinks the uncertainty named in section 6.
- Try leave-one-recording-out cross-validation instead of 5-fold, to stress-test the single worst-case held-out recording rather than an average across folds — not yet done, a natural next run of the existing script.
- The original wearable (gloves fitted with the ESP32+MPU6050) was lost partway through this project. A natural extension worth exploring is webcam-based hand-landmark tracking (e.g. via MediaPipe) as a second, hardware-free sensing modality — not a drop-in replacement for the IMU features used here, but a genuinely different signal that could be validated independently and potentially combined later.
- [NEED: your own idea — what would you actually want to try next, beyond the ones above?]

## 8. Code / reproducibility

Link to `github.com/saugatpahari/hand-anomaly-detection`. The repo contains the firmware that collected the data (`hand_imu_logger.ino`), the EDA script (`hand_anomaly_model.py`), and the actual classifier (`train_classifier.py`) — the results in section 5 are reproducible by running the latter directly against `handDatasets/`. The firmware is included as-is (adapted from Adafruit's stock MPU6050 example), imperfections and all — see section 3 for what reading it actually revealed about the data.

---

### What's left before this is finished
1. Your own voice on section 1 (why this project), the closing thought in section 6, and your own idea in section 7 — I can tighten prose but the reasoning should be recognizably yours, since this is exactly what AIAF is evaluating.
2. Section 0's summary paragraph, written last once the above is settled.
