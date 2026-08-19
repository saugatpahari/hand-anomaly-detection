/*
 * Hand IMU Logger — the actual firmware used to collect the recordings in
 * this dataset (handDatasets/). Adapted from Adafruit's stock MPU6050
 * example sketch: kept the sensor init / accelerometer-range / gyro-range /
 * filter-bandwidth setup as-is, and changed the loop to compute and log
 * per-sample roll/pitch/tilt angle instead of the example's raw-value debug
 * prints.
 *
 * WHAT THIS ACTUALLY LOGS (confirmed against the real data, 2026-08-17):
 * the three CSV columns are roll/pitch/tilt angle in DEGREES, computed
 * per-sample from the accelerometer via atan() — not raw acceleration, and
 * not fused with the gyroscope across samples (no complementary/Kalman
 * filter). Real logged values fall in roughly the -90..90 range, which is
 * the signature of degrees from this calculation, not m/s^2.
 *
 * KNOWN LIMITATIONS (left as-is, documented rather than silently fixed,
 * since this exact code produced the real dataset the project's results
 * are built on):
 *   1. No timestamp column — the real sample rate can only be estimated
 *      from this code's timing, not measured from the data. The loop below
 *      has an explicit 10ms delay() plus I2C-read and Serial-print
 *      overhead on top of that, which makes anything near 200Hz physically
 *      impossible — the real rate is more likely in the 55-80Hz range.
 *   2. `ax`/`ay`/`az` are declared as int16_t but assigned the float
 *      accelerometer readings (in m/s^2) — this truncates the fractional
 *      part before it's used in the angle calculation below, a real (if
 *      minor) precision loss on small values, not a deliberate design
 *      choice.
 *   3. Accelerometer-only tilt estimate — no gyroscope fusion, so it's
 *      noisier than a proper filtered orientation would be.
 *
 * A revised version logging raw acceleration (or both raw + derived angle)
 * with a timestamp column and gyro-fused orientation would be a meaningful
 * upgrade for any future data collection — noted as future work, not done
 * here, since this file documents what was actually used.
 */

#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

Adafruit_MPU6050 mpu;

int16_t ax, ay, az;
float arx, ary, arz;

void setup(void) {
  Serial.begin(115200);
  while (!Serial)
    delay(10); // will pause Zero, Leonardo, etc until serial console opens

  Serial.println("Adafruit MPU6050 test!");

  // Try to initialize!
  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    while (1) {
      delay(10);
    }
  }
  Serial.println("MPU6050 Found!");

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  Serial.print("Accelerometer range set to: ");
  switch (mpu.getAccelerometerRange()) {
  case MPU6050_RANGE_2_G:
    Serial.println("+-2G");
    break;
  case MPU6050_RANGE_4_G:
    Serial.println("+-4G");
    break;
  case MPU6050_RANGE_8_G:
    Serial.println("+-8G");
    break;
  case MPU6050_RANGE_16_G:
    Serial.println("+-16G");
    break;
  }

  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  Serial.print("Gyro range set to: ");
  switch (mpu.getGyroRange()) {
  case MPU6050_RANGE_250_DEG:
    Serial.println("+- 250 deg/s");
    break;
  case MPU6050_RANGE_500_DEG:
    Serial.println("+- 500 deg/s");
    break;
  case MPU6050_RANGE_1000_DEG:
    Serial.println("+- 1000 deg/s");
    break;
  case MPU6050_RANGE_2000_DEG:
    Serial.println("+- 2000 deg/s");
    break;
  }

  mpu.setFilterBandwidth(MPU6050_BAND_5_HZ);
  Serial.print("Filter bandwidth set to: ");
  switch (mpu.getFilterBandwidth()) {
  case MPU6050_BAND_260_HZ:
    Serial.println("260 Hz");
    break;
  case MPU6050_BAND_184_HZ:
    Serial.println("184 Hz");
    break;
  case MPU6050_BAND_94_HZ:
    Serial.println("94 Hz");
    break;
  case MPU6050_BAND_44_HZ:
    Serial.println("44 Hz");
    break;
  case MPU6050_BAND_21_HZ:
    Serial.println("21 Hz");
    break;
  case MPU6050_BAND_10_HZ:
    Serial.println("10 Hz");
    break;
  case MPU6050_BAND_5_HZ:
    Serial.println("5 Hz");
    break;
  }

  Serial.println("");
  delay(100);
}

void loop() {
  /* Get new sensor events with the readings */
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  ax = a.acceleration.x;
  ay = a.acceleration.y;
  az = a.acceleration.z;

  // Per-sample roll/pitch/tilt angle from the accelerometer (no gyro
  // fusion) — these three values, in degrees, are what actually gets
  // logged below, NOT raw acceleration. See the file header note.
  arx = (180 / PI) * atan(ax / sqrt((ay * ay) + (az * az)));
  ary = (180 / PI) * atan(ay / sqrt((ax * ax) + (az * az)));
  arz = (180 / PI) * atan(sqrt((ay * ay) + (ax * ax)) / az);

  Serial.print(arx);
  Serial.print(',');
  Serial.print(ary);
  Serial.print(',');
  Serial.println(arz);

  delay(10);
}
