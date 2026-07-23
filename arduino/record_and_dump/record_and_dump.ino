#include <PDM.h>

// Duracao de cada gravacao, em segundos. 5s cabe com folga na RAM
// disponivel (nao estamos rodando o modelo neste sketch, entao ha
// bastante memoria livre).
constexpr int kRecordSeconds = 5;
constexpr int kSampleRate = 16000;
constexpr int kRecordSamples = kSampleRate * kRecordSeconds;

short audioBuffer[kRecordSamples]; // 5s * 16000 * 2 bytes = 160.000 bytes (~156 KB)
volatile int audioBufferIndex = 0;
volatile bool recording = false;
volatile bool bufferReady = false;

short pdmTempBuffer[512];

void onPDMdata() {
  int bytesAvailable = PDM.available();
  PDM.read(pdmTempBuffer, bytesAvailable);
  int samplesIn = bytesAvailable / 2;

  if (!recording) return;

  for (int i = 0; i < samplesIn; i++) {
    if (audioBufferIndex < kRecordSamples) {
      audioBuffer[audioBufferIndex] = pdmTempBuffer[i];
      audioBufferIndex++;
    }
    if (audioBufferIndex >= kRecordSamples) {
      recording = false;
      bufferReady = true;
      break;
    }
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    ;
  }
  delay(500);

  PDM.onReceive(onPDMdata);
  if (!PDM.begin(1, kSampleRate)) {
    Serial.println("ERRO: falha ao iniciar o PDM.");
    while (true) {
      ;
    }
  }
  PDM.setGain(40);

  Serial.println("=== Gravador de audio ===");
  Serial.print("Envie 'r' para gravar ");
  Serial.print(kRecordSeconds);
  Serial.println(" segundos de audio.");
}

void loop() {
  if (Serial.available() > 0) {
    char c = Serial.read();
    if (c == 'r' && !recording && !bufferReady) {
      Serial.println("Gravando...");
      audioBufferIndex = 0;
      recording = true;
    }
  }

  if (bufferReady) {
    Serial.println("Gravacao concluida. Enviando dados...");
    Serial.println("BEGIN_DUMP");
    delay(50); // pequena pausa para o Python se preparar para ler os bytes

    // Envia os bytes brutos do audio (formato: int16 little-endian,
    // exatamente o mesmo formato que PDM.read() entrega)
    Serial.write((uint8_t*)audioBuffer, kRecordSamples * sizeof(short));
    Serial.flush();

    delay(50);
    Serial.println();
    Serial.println("END_DUMP");
    Serial.print("Envie 'r' para gravar ");
    Serial.print(kRecordSeconds);
    Serial.println(" segundos de audio novamente.");

    bufferReady = false;
  }
}
