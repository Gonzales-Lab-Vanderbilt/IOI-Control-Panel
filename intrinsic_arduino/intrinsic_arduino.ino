// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Gonzales Lab, Vanderbilt University

#include <Adafruit_NeoPixel.h>
#include <SPI.h>

// -----------------------------
// Pins
// -----------------------------
const int outerPixelPin = 6;   // 24-LED outer ring DIN
const int innerPixelPin = 7;   // 16-LED inner ring DIN
const int camTrigPin   = 8;
const int stimPin      = 9;    // Electrical stim TTL marker AND LRA amp SD (shared, see lraAmpEnable/Disable)

// LRA drive chain (collision-checked against everything above + SPI bus)
const int lraFsyncPin  = 10;   // AD9833 FSYNC
const int mcpCsPin     = 5;    // MCP41010 CS
// SPI MOSI (D11) / SCK (D13) are hardware SPI pins on the Uno, shared by both
// chips. Each chip gets its own SPISettings so mode differences (AD9833 wants
// SPI_MODE2, MCP41010 wants SPI_MODE0) don't corrupt each other on a shared bus.

// -----------------------------
// NeoPixel rings
// -----------------------------
// 24-LED ring outside, 16-LED ring inside.
// They are controlled as synchronized rings: same brightness, same color,
// same on/off timing.
const int outerNumPixels = 24;
const int innerNumPixels = 16;

Adafruit_NeoPixel outerRing(outerNumPixels, outerPixelPin, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel innerRing(innerNumPixels, innerPixelPin, NEO_GRB + NEO_KHZ800);

// -----------------------------
// LRA drive chain (AD9833 DDS -> MCP41010 digipot attenuator -> PAM8302A amp -> LRA)
// -----------------------------
// AD9833_MCLK must match your breakout's crystal. 25 MHz is the common default
// on Adafruit/generic AD9833 boards -- verify with a scope on the SYNC/OUT pin
// before trusting the frequency math, since a wrong MCLK silently gives you a
// wrong frequency with no error.
const double AD9833_MCLK      = 25000000.0;
const double LRA_RESONANCE_HZ = 100.0;   // Vybronics VL32158H-L25 resonance target

// Wiper value for MCP41010 (0-255). Direction (higher = more or less
// attenuation) depends on your wiring -- verify against the MPU-6050
// accelerometer rig the same way you validated the ERM sweep before trusting
// this as a calibrated amplitude.
uint8_t lraDigipotValue = 128;

SPISettings ad9833SpiSettings(4000000, MSBFIRST, SPI_MODE2);
SPISettings mcp41010SpiSettings(4000000, MSBFIRST, SPI_MODE0);

// -----------------------------
// Session-green ISI timing / frame counts
// -----------------------------
// For 10 Hz: 1000 ms / 100 ms = 10 frames/s.
// 4 seconds at 10 Hz is 40 frames.
//
// IMPORTANT: These Arduino frame counts should match the Python command-line
// values. For the current session-green Python script, typical values are:
//   --green-frames 30 --baseline-frames 40 --post-frames 40
// Arduino sends a few extra post triggers as a cushion, while Python only saves
// the requested number of post frames.
const int sessionGreenFrames = 50;
const int redBaselineFrames  = 45;
const int postFrames         = 60;   // camera trigger count, includes buffer/insurance frames

// Number of post-block trigger periods the stimulus stays active for.
// Python only saves the first 40 post frames -- the remaining postFrames-40
// are trigger-only cushion for missed frames, not part of the analysis
// window. Stimulus duration = gapMs + stimActiveFrames*triggerPeriodMs.
// With current constants: 1000ms + 40*100ms = 5000ms = 5 s.
//
// This is exact, not approximate: acquireFramesWithStimCutoff() cuts the
// stimulus at the START of post period index stimActiveFrames, so periods
// 0..stimActiveFrames-1 (40 periods = 4000 ms) are stimulated in full.
// An earlier version cut it immediately after the 40th pulse instead, which
// truncated the stimulus 94 ms early (4906 ms actual vs the 5000 ms claimed
// here and recorded in trial metadata).
const int stimActiveFrames   = 40;

const unsigned long triggerPeriodMs = 100;
const unsigned long triggerPulseMs  = 5;      // 5 ms is enough for camera trigger detection
const unsigned long gapMs           = 1000;   // 1 s gap / visual-stim lead-in
const unsigned long guardMs         = 400;

// Keep this high only if your camera is not saturating.
const uint8_t ringBrightness = 125;

bool trialRunning = false;
bool redSessionHold = false;

// Per-trial camera trigger counter. Reset to 0 at TRIAL_START (and at the start
// of the session-green block); incremented by sendTriggerPulse() after every
// pulse. Phase-boundary markers carry this so the host can label each frame by
// the trigger that produced it instead of by when the host got around to
// dequeuing it -- the host's queue depth no longer shifts phase labels.
unsigned long trialTriggerCount = 0;

// -----------------------------
// Lighting helpers
// -----------------------------
void setRingColor(uint8_t r, uint8_t g, uint8_t b) {
  outerRing.setBrightness(ringBrightness);
  innerRing.setBrightness(ringBrightness);

  for (int i = 0; i < outerNumPixels; i++) {
    outerRing.setPixelColor(i, outerRing.Color(r, g, b));
  }

  for (int i = 0; i < innerNumPixels; i++) {
    innerRing.setPixelColor(i, innerRing.Color(r, g, b));
  }

  outerRing.show();
  innerRing.show();
}

void setGreenLight() { setRingColor(0, 255, 0); }
void setRedLight()   { setRingColor(255, 0, 0); }

void ringOff() {
  outerRing.clear();
  innerRing.clear();
  outerRing.show();
  innerRing.show();
}

// -----------------------------
// Serial marker helpers
// Format expected by Python: MARKER,arduino_millis
// -----------------------------
void logMarker(const char* marker) {
  Serial.print(marker);
  Serial.print(",");
  Serial.println(millis());
}

// Three-field marker: MARKER,arduino_millis,triggerIndex
//
// CONVENTION: triggerIndex is always "the index the NEXT trigger pulse will
// carry", with the first pulse of a trial being index 1. Frames are numbered by
// the pulse that produced them, so this value reads directly as:
//   - an INCLUSIVE start boundary for markers logged before a block's first
//     pulse (RED_BASELINE_START, GAP_START, STIM_START, POST_START,
//     POST_TRAILING_START, BASELINE_FIRST_FRAME_TRIGGER, ...)
//   - an EXCLUSIVE end boundary for markers logged after a pulse (STIM_END is
//     the first frame with the stimulus already off).
// One consequence of keeping the rule uniform: a "..._LAST_FRAME_TRIGGER"
// marker reports lastIndex+1, not lastIndex. Python ignores those for phase
// boundaries; they stay informational.
//
// Python's parser treats the third field as optional, so firmware without it
// still works against the current host script.
void logMarkerWithTrigger(const char* marker) {
  Serial.print(marker);
  Serial.print(",");
  Serial.print(millis());
  Serial.print(",");
  Serial.println(trialTriggerCount + 1UL);
}

// -----------------------------
// Camera trigger helpers
// -----------------------------
void sendTriggerPulse() {
  digitalWrite(camTrigPin, HIGH);
  delay(triggerPulseMs);
  digitalWrite(camTrigPin, LOW);
  trialTriggerCount++;
}

void acquireFramesWithDebug(int frameCount, const char* firstMarker, const char* lastMarker) {
  for (int i = 0; i < frameCount; i++) {
    unsigned long t0 = millis();

    if (i == 0) {
      logMarkerWithTrigger(firstMarker);
    }

    sendTriggerPulse();

    if (i == frameCount - 1) {
      logMarkerWithTrigger(lastMarker);
    }

    while (millis() - t0 < triggerPeriodMs) {
      // Intentionally blocking during tightly timed trigger train.
    }
  }
}

// Same trigger train as acquireFramesWithDebug, but calls stimOffFn() (and
// logs STIM_END) partway through, instead of waiting for the full (padded)
// frameCount to finish. This lets postFrames stay generous for camera-trigger
// insurance without dragging the actual stimulus duration along with it.
//
// The cut happens at the START of period index cutoffFrameIndex, so periods
// 0..cutoffFrameIndex-1 are stimulated in full and the stimulus lasts exactly
// cutoffFrameIndex*triggerPeriodMs from the first post pulse (plus the
// preceding gapMs lead-in). Cutting right after the cutoffFrameIndex-th pulse
// instead -- as this used to -- ends the stimulus one period minus one pulse
// width early.
void acquireFramesWithStimCutoff(int frameCount, int cutoffFrameIndex,
                                  void (*stimOffFn)(),
                                  const char* firstMarker, const char* lastMarker) {
  for (int i = 0; i < frameCount; i++) {
    unsigned long t0 = millis();

    if (i == 0) {
      logMarkerWithTrigger(firstMarker);
    }

    bool stimCutoffHere = (cutoffFrameIndex > 0 && i == cutoffFrameIndex);
    // Safety net: a cutoff at or past the end of this block would otherwise
    // never fire and would leave the stimulus running past the post block.
    if (cutoffFrameIndex >= frameCount && i == frameCount - 1) {
      stimCutoffHere = true;
    }

    // digitalWrite/SPI only -- microseconds -- so the pulse below still fires
    // on schedule. STIM_END is logged *after* the pulse so its serial write
    // has the rest of the period to drain and can never push a trigger late.
    if (stimCutoffHere) {
      stimOffFn();
    }

    sendTriggerPulse();

    if (stimCutoffHere) {
      logMarkerWithTrigger("STIM_END");
    }

    if (i == frameCount - 1) {
      logMarkerWithTrigger(lastMarker);
    }

    while (millis() - t0 < triggerPeriodMs) {
      // Intentionally blocking during tightly timed trigger train.
    }
  }
}

// Same trigger cadence as acquireFramesWithDebug, but for a fixed duration
// instead of a fixed frame count -- used to keep the camera triggered through
// what used to be blocking delay() calls (the guardMs debounce buffers on
// either side of the gap, and the gapMs stim lead-in itself) so no part of
// the baseline-to-post span is a dead spot in acquisition. No first/last
// frame markers: GAP_START/STIM_START/POST_START/TRIAL_END already bracket
// the call sites. Assumes durationMs is a whole multiple of triggerPeriodMs
// (true for the current guardMs/gapMs/triggerPeriodMs constants); a
// non-multiple truncates to the last full trigger period.
void acquireGapFrames(unsigned long durationMs) {
  int frameCount = (int)(durationMs / triggerPeriodMs);
  for (int i = 0; i < frameCount; i++) {
    unsigned long t0 = millis();
    sendTriggerPulse();
    while (millis() - t0 < triggerPeriodMs) {
      // Intentionally blocking during tightly timed trigger train.
    }
  }
}

// -----------------------------
// AD9833 DDS helpers
// -----------------------------
void ad9833WriteRegister(uint16_t data) {
  SPI.beginTransaction(ad9833SpiSettings);
  digitalWrite(lraFsyncPin, LOW);
  SPI.transfer(highByte(data));
  SPI.transfer(lowByte(data));
  digitalWrite(lraFsyncPin, HIGH);
  SPI.endTransaction();
}

// Sets the DDS output to a sine wave at freqHz and lets it free-run.
void ad9833SetFrequency(double freqHz) {
  uint32_t freqWord = (uint32_t)((freqHz * 268435456.0) / AD9833_MCLK); // 2^28

  uint16_t lsb = (uint16_t)(freqWord & 0x3FFF) | 0x4000;         // FREQ0 select
  uint16_t msb = (uint16_t)((freqWord >> 14) & 0x3FFF) | 0x4000; // FREQ0 select

  ad9833WriteRegister(0x2100); // B28=1, RESET=1 while loading the frequency word
  ad9833WriteRegister(lsb);
  ad9833WriteRegister(msb);
  ad9833WriteRegister(0x2000); // B28=1, RESET=0 -> sine output running
}

// Holds the DDS in reset. Output settles to a fixed level, no signal reaches
// the amp input even if the amp SD line is accidentally left enabled.
void ad9833Sleep() {
  ad9833WriteRegister(0x2100);
}

// -----------------------------
// MCP41010 digipot helper
// -----------------------------
void mcp41010SetWiper(uint8_t value) {
  SPI.beginTransaction(mcp41010SpiSettings);
  digitalWrite(mcpCsPin, LOW);
  SPI.transfer(0x11); // write command, select pot 0
  SPI.transfer(value);
  digitalWrite(mcpCsPin, HIGH);
  SPI.endTransaction();
}

// -----------------------------
// LRA amp enable/disable
// PAM8302A SD pin: HIGH = enabled, LOW = shutdown.
// Reuses stimPin so LRA trials get the same TTL marker channel the electrical
// stim trials use -- downstream analysis doesn't need to care which modality
// drove a given trial.
// -----------------------------
void lraAmpEnable()  { digitalWrite(stimPin, HIGH); }
void lraAmpDisable() { digitalWrite(stimPin, LOW); }

// Stim-off callbacks for acquireFramesWithStimCutoff.
void stopElectricalStim() { digitalWrite(stimPin, LOW); }
void stopLraStim()        { lraAmpDisable(); ad9833Sleep(); }

void lraInit() {
  pinMode(lraFsyncPin, OUTPUT);
  pinMode(mcpCsPin, OUTPUT);
  digitalWrite(lraFsyncPin, HIGH);
  digitalWrite(mcpCsPin, HIGH);

  SPI.begin();

  mcp41010SetWiper(lraDigipotValue);
  ad9833SetFrequency(LRA_RESONANCE_HZ);
  ad9833Sleep();     // start silent; woken explicitly when a trial needs it
  lraAmpDisable();
}

// -----------------------------
// Session setup phases
// -----------------------------
void runSessionGreenReference() {
  if (trialRunning) {
    logMarker("BUSY_TRIAL_RUNNING");
    return;
  }

  redSessionHold = false;
  digitalWrite(stimPin, LOW);

  trialTriggerCount = 0;

  setGreenLight();
  logMarker("SESSION_GREEN_REFERENCE_START");
  // Backward-compatible marker; harmless if Python is listening for the newer one.
  logMarker("GREEN_BASELINE_START");

  acquireFramesWithDebug(
    sessionGreenFrames,
    "SESSION_GREEN_FIRST_FRAME_TRIGGER",
    "SESSION_GREEN_LAST_FRAME_TRIGGER"
  );

  logMarker("SESSION_GREEN_REFERENCE_END");

  // Turn green off after reference acquisition. Red will be turned on by SESSION_RED_ON.
  ringOff();
}

void turnSessionRedOn() {
  if (trialRunning) {
    logMarker("BUSY_TRIAL_RUNNING");
    return;
  }

  digitalWrite(stimPin, LOW);
  setRedLight();
  redSessionHold = true;

  logMarker("RED_STABILIZATION_START");
  logMarker("RED_ON");
  // Python performs the 600-second stabilization wait; Arduino just holds red on.
}

void turnAllLightsOff() {
  digitalWrite(stimPin, LOW);
  ad9833Sleep();
  ringOff();
  redSessionHold = false;
  logMarker("LIGHTS_OFF");
}

// -----------------------------
// Red-only trial (electrical/peripheral stim)
// stimEnabled=false runs an identical-timing "catch" trial: every marker,
// camera trigger, and frame count is unchanged, only the physical stimPin
// actuation is skipped -- so catch and stim trials are directly comparable.
// -----------------------------
void runRedOnlyTrial(bool stimEnabled) {
  if (trialRunning) {
    logMarker("BUSY_TRIAL_RUNNING");
    return;
  }

  trialRunning = true;

  // Safety: if Python forgot SESSION_RED_ON, turn red on here but do not turn it off
  // at trial end. This preserves the session-hold behavior once trials start.
  if (!redSessionHold) {
    setRedLight();
    redSessionHold = true;
    logMarker("RED_ON_AUTO");
  }

  digitalWrite(stimPin, LOW);

  trialTriggerCount = 0;
  logMarkerWithTrigger("TRIAL_START");

  // Baseline: red light stays on continuously.
  logMarkerWithTrigger("RED_BASELINE_START");
  acquireFramesWithDebug(
    redBaselineFrames,
    "BASELINE_FIRST_FRAME_TRIGGER",
    "BASELINE_LAST_FRAME_TRIGGER"
  );

  logMarkerWithTrigger("GAP_START");
  acquireGapFrames(guardMs);

  // Visual stimulus starts at STIM_START. The host uses this marker to command
// the visual stimulus server.
  // The marker itself always fires (timing/window alignment must not depend
  // on condition); only the physical TTL actuation is gated by stimEnabled.
  logMarkerWithTrigger("STIM_START");
  if (stimEnabled) {
    digitalWrite(stimPin, HIGH);  // optional hardware TTL marker for stimulus period
  }
  acquireGapFrames(gapMs);

  // Post/stimulation imaging. Red light remains on; do not change LED state here.
  // Stim TTL turns off after stimActiveFrames (matches Python's saved frame
  // count); the remaining postFrames-stimActiveFrames are trigger-only buffer.
  logMarkerWithTrigger("POST_START");
  acquireFramesWithStimCutoff(
    postFrames,
    stimActiveFrames,
    stopElectricalStim,
    "POST_FIRST_FRAME_TRIGGER",
    "POST_LAST_FRAME_TRIGGER"
  );

  logMarkerWithTrigger("POST_TRAILING_START");
  acquireGapFrames(guardMs);
  logMarkerWithTrigger("TRIAL_END");
  logMarker("CYCLE_COMPLETE");

  // CRITICAL: Do not call ringOff() here. Red remains on across trials until
  // Python sends LIGHTS_OFF during normal completion, error cleanup, or Ctrl+C.
  trialRunning = false;
  logMarker("ARDUINO_READY");
}

// -----------------------------
// LRA-only trial (proprioceptive stim)
// Mirrors runRedOnlyTrial() structure exactly -- same baseline/gap/stim/post
// timing, same marker names -- so the Python/analysis side treats an LRA
// trial identically to an electrical trial. Only the stimulus delivery
// mechanism differs (AD9833+amp instead of a direct TTL to the isolated
// stimulator).
//
// Amp is enabled from STIM_START and disabled after stimActiveFrames of the
// post block (via acquireFramesWithStimCutoff), giving gapMs +
// stimActiveFrames*triggerPeriodMs of active buzz -- 1000ms + 4000ms = 5s
// with current constants. The remaining postFrames-stimActiveFrames frames
// are still triggered for camera-buffer insurance, but silent.
//
// stimEnabled=false runs an identical-timing "catch" trial: only the DDS/
// amp actuation is skipped, matching runRedOnlyTrial()'s stimEnabled param.
// -----------------------------
void runLraTrial(bool stimEnabled) {
  if (trialRunning) {
    logMarker("BUSY_TRIAL_RUNNING");
    return;
  }

  trialRunning = true;

  if (!redSessionHold) {
    setRedLight();
    redSessionHold = true;
    logMarker("RED_ON_AUTO");
  }

  lraAmpDisable();
  ad9833Sleep();

  trialTriggerCount = 0;
  logMarkerWithTrigger("TRIAL_START");

  logMarkerWithTrigger("RED_BASELINE_START");
  acquireFramesWithDebug(
    redBaselineFrames,
    "BASELINE_FIRST_FRAME_TRIGGER",
    "BASELINE_LAST_FRAME_TRIGGER"
  );

  logMarkerWithTrigger("GAP_START");
  acquireGapFrames(guardMs);

  logMarkerWithTrigger("STIM_START");
  if (stimEnabled) {
    ad9833SetFrequency(LRA_RESONANCE_HZ); // re-arm in case anything put the DDS to sleep
    lraAmpEnable();
  }
  acquireGapFrames(gapMs);

  logMarkerWithTrigger("POST_START");
  acquireFramesWithStimCutoff(
    postFrames,
    stimActiveFrames,
    stopLraStim,
    "POST_FIRST_FRAME_TRIGGER",
    "POST_LAST_FRAME_TRIGGER"
  );

  logMarkerWithTrigger("POST_TRAILING_START");
  acquireGapFrames(guardMs);
  logMarkerWithTrigger("TRIAL_END");
  logMarker("CYCLE_COMPLETE");

  trialRunning = false;
  logMarker("ARDUINO_READY");
}

// -----------------------------
// Serial command interface
// -----------------------------
void handleCommand(String cmd) {
  cmd.trim();

  if (cmd == "SESSION_GREEN_REFERENCE") {
    runSessionGreenReference();
    logMarker("ARDUINO_READY");
  }

  else if (cmd == "SESSION_RED_ON") {
    turnSessionRedOn();
    logMarker("ARDUINO_READY");
  }

  else if (cmd == "START_TRIAL") {
    runRedOnlyTrial(true);
  }

  // Catch-trial variant: identical timing/markers, stim actuation skipped.
  // See runRedOnlyTrial()'s stimEnabled param.
  else if (cmd.startsWith("START_TRIAL,")) {
    int v = cmd.substring(cmd.indexOf(',') + 1).toInt();
    runRedOnlyTrial(v != 0);
  }

  // Backward compatibility with older Python versions.
  // START now behaves like START_TRIAL in this session-green firmware.
  else if (cmd == "START") {
    runRedOnlyTrial(true);
  }

  else if (cmd == "STIM_LRA") {
    runLraTrial(true);
  }

  // Catch-trial variant: identical timing/markers, LRA actuation skipped.
  else if (cmd.startsWith("STIM_LRA,")) {
    int v = cmd.substring(cmd.indexOf(',') + 1).toInt();
    runLraTrial(v != 0);
  }

  else if (cmd == "CAL_TRIGGER") {
    if (!trialRunning) {
      sendTriggerPulse();
      logMarker("CAL_TRIGGER");
    } else {
      logMarker("BUSY_TRIAL_RUNNING");
    }
  }

  else if (cmd == "CAL_GREEN_ON") {
    if (!trialRunning) {
      redSessionHold = false;
      setGreenLight();
      logMarker("CAL_GREEN_ON");
    } else {
      logMarker("BUSY_TRIAL_RUNNING");
    }
  }

  else if (cmd == "CAL_RED_ON" || cmd == "RED_ON") {
    if (!trialRunning) {
      setRedLight();
      redSessionHold = true;
      logMarker("CAL_RED_ON");
      logMarker("RED_ON");
    } else {
      logMarker("BUSY_TRIAL_RUNNING");
    }
  }

  else if (cmd == "LIGHTS_OFF" || cmd == "RED_OFF") {
    if (!trialRunning) {
      turnAllLightsOff();
    } else {
      // Do not interrupt a timed trial mid-trigger-train. Python sends cleanup
      // after interruption/error; if a command arrives during a trial, report busy.
      logMarker("BUSY_TRIAL_RUNNING");
    }
  }

  // --- LRA bench/calibration commands ---
  // These exist so you can drive the LRA standalone (e.g. against the
  // MPU-6050 characterization rig) without running a full imaging trial.

  else if (cmd == "CAL_LRA_ON") {
    if (!trialRunning) {
      ad9833SetFrequency(LRA_RESONANCE_HZ);
      lraAmpEnable();
      logMarker("CAL_LRA_ON");
    } else {
      logMarker("BUSY_TRIAL_RUNNING");
    }
  }

  else if (cmd == "CAL_LRA_OFF") {
    // Always allowed, including mid-trial -- this is the emergency kill
    // switch for the LRA if something looks wrong on the bench.
    lraAmpDisable();
    ad9833Sleep();
    logMarker("CAL_LRA_OFF");
  }

  else if (cmd.startsWith("LRA_SET_FREQ,")) {
    double freq = cmd.substring(cmd.indexOf(',') + 1).toDouble();
    ad9833SetFrequency(freq);
    logMarker("LRA_FREQ_SET");
  }

  else if (cmd.startsWith("LRA_SET_AMP,")) {
    int amp = cmd.substring(cmd.indexOf(',') + 1).toInt();
    amp = constrain(amp, 0, 255);
    lraDigipotValue = (uint8_t)amp;
    mcp41010SetWiper(lraDigipotValue);
    logMarker("LRA_AMP_SET");
  }

  else if (cmd == "PING") {
    logMarker("ARDUINO_READY");
  }

  else {
    logMarker("UNKNOWN_COMMAND");
  }
}

void setup() {
  pinMode(camTrigPin, OUTPUT);
  pinMode(stimPin, OUTPUT);

  digitalWrite(camTrigPin, LOW);
  digitalWrite(stimPin, LOW);

  outerRing.begin();
  innerRing.begin();
  ringOff();

  lraInit();

  Serial.begin(115200);
  delay(500);
  logMarker("ARDUINO_READY");
}

void loop() {
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    handleCommand(cmd);
  }
}
