"""
15_record_via_arduino.py

Recebe gravacoes de audio do Arduino (feitas pelo sketch
record_and_dump.ino) via porta serial e salva como arquivos .wav de
16kHz mono -- prontos para usar no fine-tuning, desta vez capturados
pelo MESMO microfone que vai fazer a deteccao de verdade.

Uso:
    python 15_record_via_arduino.py COM5
    (troque COM5 pela porta que aparece no Gerenciador de Dispositivos)

Fluxo de uso recomendado:
    1. Rode o script.
    2. Digite um nome de arquivo (ex: drone_01) e pressione Enter.
    3. O script manda o comando de gravar para o Arduino automaticamente.
    4. Aguarde -- vai aparecer "Gravando..." e depois o arquivo .wav
       sera salvo em gravacoes_arduino/<nome>.wav
    5. Repita quantas vezes quiser (varias gravacoes do drone em
       posicoes/distancias diferentes, e varias gravacoes so do
       ambiente sem o drone).
    6. Digite 'sair' para encerrar.

Requisitos: pip install pyserial
"""

import sys
import wave
import os

try:
    import serial
except ImportError:
    print("Faltando a biblioteca pyserial. Instale com: pip install pyserial")
    sys.exit(1)

BAUD_RATE = 115200
SAMPLE_RATE = 16000
RECORD_SECONDS = 5  # deve bater com kRecordSeconds no sketch .ino
RECORD_SAMPLES = SAMPLE_RATE * RECORD_SECONDS
RECORD_BYTES = RECORD_SAMPLES * 2  # int16 = 2 bytes por amostra

OUTPUT_DIR = "dataset/gravacoes_arduino"


def record_one_clip(ser, output_path):
    print("Enviando comando de gravacao...")
    ser.reset_input_buffer()
    ser.write(b"r")

    # Le linhas de texto ate encontrar o marcador BEGIN_DUMP
    print("Aguardando o Arduino gravar e comecar o envio...")
    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if line:
            print(f"  [Arduino] {line}")
        if line == "BEGIN_DUMP":
            break
        if not line and ser.in_waiting == 0:
            # timeout de leitura sem receber nada -- evita loop infinito
            continue

    # A partir daqui, le exatamente RECORD_BYTES bytes brutos
    print(f"Recebendo {RECORD_BYTES} bytes de audio...")
    raw_data = ser.read(RECORD_BYTES)

    if len(raw_data) != RECORD_BYTES:
        print(
            f"AVISO: esperava {RECORD_BYTES} bytes, recebi {len(raw_data)}. "
            "A gravacao pode estar incompleta -- tente novamente."
        )

    # Salva como .wav (16-bit PCM, mono, 16000 Hz)
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16 bits
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(raw_data)

    print(f"OK -> salvo em {output_path} ({len(raw_data)} bytes)\n")


def main():
    if len(sys.argv) < 2:
        print("Uso: python 15_record_via_arduino.py <porta>  (ex: COM5 ou /dev/cu.usbmodemXXXX)")
        sys.exit(1)

    port = sys.argv[1]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Conectando na porta {port}...")
    ser = serial.Serial(port, BAUD_RATE, timeout=2)
    print("Conectado! Aguarde a mensagem inicial do Arduino...\n")

    import time
    time.sleep(2)  # da tempo do Arduino reiniciar apos abrir a porta serial
    ser.reset_input_buffer()

    while True:
        name = input("Nome do arquivo (sem .wav), ou 'sair': ").strip()
        if name.lower() == "sair":
            break
        if not name:
            continue

        output_path = os.path.join(OUTPUT_DIR, f"{name}.wav")
        record_one_clip(ser, output_path)

    ser.close()
    print("Encerrado.")


if __name__ == "__main__":
    main()