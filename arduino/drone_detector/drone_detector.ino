#include <PDM.h>
#include <arduinoFFT.h>
#include <ArduTFLite.h>
#include <ArduinoBLE.h>
#include "model.h"
#include "mfcc_tables.h"

// ---------------------------------------------------------------------
// CONFIGURACAO GERAL
// ---------------------------------------------------------------------
constexpr int kSampleRate = 16000;
constexpr int kWindowSamples = kSampleRate; // 1 segundo = 16000 amostras
constexpr float kDetectionThreshold = 0.5f;
constexpr float kHighConfidenceThreshold = 0.8f;

// Area de memoria de trabalho do TensorFlow Lite Micro
constexpr int kTensorArenaSize = 80 * 1024;
byte tensorArena[kTensorArenaSize];

const int kNumInputValues = MFCC_N_MFCC * MFCC_N_FRAMES; // 40 * 63 = 2520

// ---------------------------------------------------------------------
// LED RGB EMBUTIDO -- indica o status sem precisar do Serial Monitor
// ---------------------------------------------------------------------
// No Nano 33 BLE (Sense), os LEDs sao "ativos em LOW" (LOW = aceso).
enum LedMode {
  LED_MODE_OFF,
  LED_MODE_INIT_BLUE,      // inicializando
  LED_MODE_BACKGROUND,     // verde solido -- sem drone
  LED_MODE_DETECTED,       // vermelho solido -- drone (confianca moderada)
  LED_MODE_DETECTED_HIGH,  // vermelho piscando rapido -- drone (alta confianca)
  LED_MODE_ERROR           // vermelho piscando bem rapido, para sempre
};

volatile LedMode currentLedMode = LED_MODE_OFF;

void setRawColor(bool red, bool green, bool blue) {
  digitalWrite(LEDR, red ? LOW : HIGH);
  digitalWrite(LEDG, green ? LOW : HIGH);
  digitalWrite(LEDB, blue ? LOW : HIGH);
}

// Deve ser chamada com frequencia (a cada volta do loop) para o
// piscar funcionar direito, sem travar o resto do programa.
void updateLED() {
  static unsigned long lastToggle = 0;
  static bool blinkOn = false;
  unsigned long now = millis();

  switch (currentLedMode) {
    case LED_MODE_OFF:
      setRawColor(false, false, false);
      break;
    case LED_MODE_INIT_BLUE:
      setRawColor(false, false, true);
      break;
    case LED_MODE_BACKGROUND:
      setRawColor(false, true, false);
      break;
    case LED_MODE_DETECTED:
      setRawColor(true, false, false);
      break;
    case LED_MODE_DETECTED_HIGH:
      if (now - lastToggle >= 150) {
        blinkOn = !blinkOn;
        lastToggle = now;
      }
      setRawColor(blinkOn, false, false);
      break;
    case LED_MODE_ERROR:
      if (now - lastToggle >= 100) {
        blinkOn = !blinkOn;
        lastToggle = now;
      }
      setRawColor(blinkOn, false, false);
      break;
  }
}

// Chamada quando algo da errado de forma irrecuperavel na inicializacao.
// Fica piscando vermelho rapido para sempre (visivel mesmo sem Serial).
void fatalError(const char* message) {
  Serial.println(message);
  currentLedMode = LED_MODE_ERROR;
  while (true) {
    updateLED();
  }
}

// ---------------------------------------------------------------------
// BLUETOOTH (BLE) -- permite acompanhar a deteccao pelo celular, sem
// precisar de cabo/Serial Monitor. Use um app generico de BLE (ex:
// "LightBlue", gratuito, iOS/Mac) para se conectar e ver os valores.
// ---------------------------------------------------------------------
BLEService droneService("19B10000-E8F2-537E-4F6C-D104768A1214");

// Estado da deteccao, como um numero simples: 0 = background, 1 = drone (confianca moderada), 2 = drone (alta confianca)
BLEByteCharacteristic stateCharacteristic(
    "19B10001-E8F2-537E-4F6C-D104768A1214", BLERead | BLENotify);

// Probabilidade "crua" (0.0 a 1.0)
BLEFloatCharacteristic probabilityCharacteristic(
    "19B10002-E8F2-537E-4F6C-D104768A1214", BLERead | BLENotify);

// Caracteristica com string legivel: "No Drone", "Possible", "Drone"
BLEStringCharacteristic statusStringCharacteristic(
    "19B10003-E8F2-537E-4F6C-D104768A1214", BLERead | BLENotify, 20);

// Variaveis para controlar envio apenas em mudanca de estado
byte lastSentState = 0xFF;      // valor inicial invalido
float lastSentProb = -1.0f;

void setupBLE() {
  if (!BLE.begin()) {
    fatalError("ERRO: falha ao iniciar o Bluetooth (BLE).");
  }

  BLE.setLocalName("DroneDetector");
  BLE.setAdvertisedService(droneService);

  droneService.addCharacteristic(stateCharacteristic);
  droneService.addCharacteristic(probabilityCharacteristic);
  droneService.addCharacteristic(statusStringCharacteristic);
  BLE.addService(droneService);

  // Valores iniciais
  stateCharacteristic.writeValue(0);
  probabilityCharacteristic.writeValue(0.0f);
  statusStringCharacteristic.writeValue("No Drone");

  BLE.advertise();
  Serial.println("BLE anunciando como 'DroneDetector'.");
}

// ---------------------------------------------------------------------
// CAPTURA DE AUDIO (PDM)
// ---------------------------------------------------------------------
short audioBuffer[kWindowSamples];
volatile int audioBufferIndex = 0;
volatile bool bufferReady = false;
short pdmTempBuffer[512];

void onPDMdata() {
  int bytesAvailable = PDM.available();
  PDM.read(pdmTempBuffer, bytesAvailable);
  int samplesIn = bytesAvailable / 2;

  for (int i = 0; i < samplesIn; i++) {
    if (audioBufferIndex < kWindowSamples) {
      audioBuffer[audioBufferIndex] = pdmTempBuffer[i];
      audioBufferIndex++;
    }
    if (audioBufferIndex >= kWindowSamples) {
      bufferReady = true;
      break;
    }
  }
}

// ---------------------------------------------------------------------
// CALCULO DO MFCC
// ---------------------------------------------------------------------
float hannWindow[MFCC_N_FFT];              // janela de Hann (periodica), calculada uma vez
float fftReal[MFCC_N_FFT];                 // buffers de trabalho da FFT
float fftImag[MFCC_N_FFT];
ArduinoFFT<float> FFT = ArduinoFFT<float>(fftReal, fftImag, MFCC_N_FFT, (float)kSampleRate);

// Guardado como inteiro de ponto fixo (valor real = armazenado / 100.0)
// em vez de float -- economiza ~16KB de RAM, com precisao de 0.01 dB
int16_t melEnergyDbFixed[MFCC_N_MELS][MFCC_N_FRAMES];

void computeHannWindow() {
  for (int n = 0; n < MFCC_N_FFT; n++) {
    hannWindow[n] = 0.5f - 0.5f * cosf(2.0f * PI * n / MFCC_N_FFT);
  }
}

void computeMFCC() {
  const int padding = MFCC_N_FFT / 2; // 256
  float globalMaxDb = -1e9f;

  for (int frame = 0; frame < MFCC_N_FRAMES; frame++) {
    int frameStart = frame * MFCC_HOP_LENGTH - padding;

    for (int n = 0; n < MFCC_N_FFT; n++) {
      int sampleIndex = frameStart + n;
      float sample = 0.0f;
      if (sampleIndex >= 0 && sampleIndex < kWindowSamples) {
        sample = audioBuffer[sampleIndex] / 32768.0f;
      }
      fftReal[n] = sample * hannWindow[n];
      fftImag[n] = 0.0f;
    }

    FFT.compute(FFTDirection::Forward);

    for (int mel = 0; mel < MFCC_N_MELS; mel++) {
      float energy = 0.0f;
      for (int k = 0; k < MFCC_N_FREQ_BINS; k++) {
        float power = fftReal[k] * fftReal[k] + fftImag[k] * fftImag[k];
        energy += mel_filterbank(mel, k) * power;
      }
      if (energy < 1e-10f) energy = 1e-10f;
      float db = 10.0f * log10f(energy);
      melEnergyDbFixed[mel][frame] = (int16_t)roundf(db * 100.0f);
      if (db > globalMaxDb) globalMaxDb = db;
    }
  }

  float dbFloor = globalMaxDb - 80.0f;
  int16_t dbFloorFixed = (int16_t)roundf(dbFloor * 100.0f);
  for (int mel = 0; mel < MFCC_N_MELS; mel++) {
    for (int frame = 0; frame < MFCC_N_FRAMES; frame++) {
      if (melEnergyDbFixed[mel][frame] < dbFloorFixed) {
        melEnergyDbFixed[mel][frame] = dbFloorFixed;
      }
    }
  }
}

float computeMfccValue(int coef, int frame) {
  float sum = 0.0f;
  for (int mel = 0; mel < MFCC_N_MELS; mel++) {
    sum += dct_matrix(coef, mel) * (melEnergyDbFixed[mel][frame] / 100.0f);
  }
  return sum;
}

bool loadMFCCIntoModel() {
  double sum = 0.0;
  for (int coef = 0; coef < MFCC_N_MFCC; coef++) {
    for (int frame = 0; frame < MFCC_N_FRAMES; frame++) {
      sum += computeMfccValue(coef, frame);
    }
  }
  double mean = sum / kNumInputValues;

  double sumSqDiff = 0.0;
  for (int coef = 0; coef < MFCC_N_MFCC; coef++) {
    for (int frame = 0; frame < MFCC_N_FRAMES; frame++) {
      double diff = computeMfccValue(coef, frame) - mean;
      sumSqDiff += diff * diff;
    }
  }
  double stddev = sqrt(sumSqDiff / kNumInputValues);

  for (int coef = 0; coef < MFCC_N_MFCC; coef++) {
    for (int frame = 0; frame < MFCC_N_FRAMES; frame++) {
      float value = computeMfccValue(coef, frame);
      float normalized = (float)((value - mean) / (stddev + 1e-8));
      int index = coef * MFCC_N_FRAMES + frame;
      if (!modelSetInput(normalized, index)) {
        Serial.print("ERRO: falha ao definir entrada no indice ");
        Serial.println(index);
        return false;
      }
    }
  }
  return true;
}

// ---------------------------------------------------------------------
// SETUP / LOOP
// ---------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  unsigned long serialWaitStart = millis();
  while (!Serial && (millis() - serialWaitStart) < 3000) {
    ;
  }

  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);
  currentLedMode = LED_MODE_INIT_BLUE;
  updateLED();

  Serial.println("=== Detector de drone -- inicializando ===");

  computeHannWindow();

  Serial.println("Carregando modelo...");
  bool ok = modelInit(g_drone_model_data, tensorArena, kTensorArenaSize);
  if (!ok) {
    fatalError("ERRO: falha ao inicializar o modelo.");
  }
  Serial.println("Modelo carregado com sucesso!");

  PDM.onReceive(onPDMdata);
  if (!PDM.begin(1, kSampleRate)) {
    fatalError("ERRO: falha ao iniciar o PDM.");
  }

  PDM.setGain(80);

  setupBLE();

  currentLedMode = LED_MODE_BACKGROUND;
  Serial.println("Pronto! Escutando continuamente...");
  Serial.println("Conecte um app BLE (ex: LightBlue) e procure por 'DroneDetector'.");
  Serial.println();
}

void loop() {
  BLE.poll();
  updateLED();

  if (bufferReady) {
    unsigned long t0 = millis();

    short minVal = 32767, maxVal = -32768;
    for (int i = 0; i < kWindowSamples; i++) {
      if (audioBuffer[i] < minVal) minVal = audioBuffer[i];
      if (audioBuffer[i] > maxVal) maxVal = audioBuffer[i];
    }

    computeMFCC();

    if (!loadMFCCIntoModel()) {
      audioBufferIndex = 0;
      bufferReady = false;
      return;
    }

    bool inferenceOk = modelRunInference();
    if (!inferenceOk) {
      Serial.println("ERRO: falha ao rodar a inferencia.");
      audioBufferIndex = 0;
      bufferReady = false;
      return;
    }

    float probability = modelGetOutput(0);
    unsigned long elapsedMs = millis() - t0;

    // --- Determina estado ---
    byte state;
    if (probability >= kHighConfidenceThreshold) {
      currentLedMode = LED_MODE_DETECTED_HIGH;
      state = 2;
    } else if (probability >= kDetectionThreshold) {
      currentLedMode = LED_MODE_DETECTED;
      state = 1;
    } else {
      currentLedMode = LED_MODE_BACKGROUND;
      state = 0;
    }

    // --- Atualiza BLE apenas se houve mudança de estado OU se a probabilidade variou mais de 0.05 (opcional) ---
    bool stateChanged = (state != lastSentState);
    bool probChanged = (fabs(probability - lastSentProb) > 0.05);

    if (stateChanged) {
      stateCharacteristic.writeValue(state);
      lastSentState = state;

      // Atualiza a string de status junto com o estado
      if (state == 2) {
        statusStringCharacteristic.writeValue("Drone");
      } else if (state == 1) {
        statusStringCharacteristic.writeValue("Possible");
      } else {
        statusStringCharacteristic.writeValue("No Drone");
      }
    }

    // Atualiza probabilidade somente se mudar significativamente (evita spam)
    if (probChanged) {
      probabilityCharacteristic.writeValue(probability);
      lastSentProb = probability;
    }

    // --- Serial (opcional) ---
    Serial.print("Amplitude [min/max]: ");
    Serial.print(minVal);
    Serial.print(" / ");
    Serial.print(maxVal);
    Serial.print("  |  Probabilidade de drone: ");
    Serial.print(probability, 4);
    Serial.print("  |  ");
    if (probability >= kDetectionThreshold) {
      Serial.print("*** DRONE DETECTADO ***");
    } else {
      Serial.print("background");
    }
    Serial.print("  |  tempo de processamento: ");
    Serial.print(elapsedMs);
    Serial.println(" ms");

    audioBufferIndex = 0;
    bufferReady = false;
  }
}