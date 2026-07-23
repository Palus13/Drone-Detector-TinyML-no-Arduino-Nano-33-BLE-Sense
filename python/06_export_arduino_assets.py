"""
13_export_arduino_assets.py

Gera os arquivos que vao para dentro da pasta do sketch Arduino:

    arduino_sketch/model.h        <- modelo .tflite convertido para C++
    arduino_sketch/mfcc_tables.h  <- matrizes (mel filterbank + DCT) para
                                      calcular o MFCC dentro do Arduino,
                                      extraidas DIRETAMENTE do librosa
                                      (garante fidelidade com o treino).

IMPORTANTE: os dados binarios (bytes do modelo e das matrizes) sao
codificados como STRING LITERAL com escapes octais (ex: "\\101\\102...")
em vez de uma lista de numeros separados por virgula (ex: {0x41, 0x42,...}).
Isso e proposital: compiladores C++ gastam MUITO mais memoria RAM (no seu
PC, durante a compilacao) processando listas gigantes de numeros do que
processando uma string de texto do mesmo tamanho. Em PCs com pouca RAM
livre, a versao "lista de numeros" pode fazer o compilador (cc1plus.exe)
travar com erro de "out of memory". A versao "string" produz exatamente
os mesmos bytes no Arduino, so muda como o compilador enxerga o texto.

Tambem valida matematicamente que a reconstrucao manual do MFCC (usando
essas matrizes) e identica ao librosa.feature.mfcc() original, antes de
gerar qualquer arquivo -- se a validacao falhar, o script para e avisa.

Requisitos: tensorflow, librosa, numpy, scipy (mesmo ambiente ja usado
no projeto).
"""

import os

import numpy as np
import librosa
from scipy.fft import dct

# --------------------------------------------------------------------------
# CONFIGURACAO -- deve bater EXATAMENTE com o que foi usado no treino
# --------------------------------------------------------------------------
TFLITE_MODEL_PATH = "models/finetuned_lightweight_drone_arduino_mic.tflite"
OUTPUT_DIR = "arduino/drone_detector"

SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 256
N_MFCC = 40
N_MELS = 128          # default do librosa quando n_mels nao e especificado
WINDOW_SAMPLES = SAMPLE_RATE  # 1 segundo = 16000 amostras
N_FRAMES_EXPECTED = 63  # conferido: 1 + floor(16000/256) = 63

BYTES_PER_LINE = 100  # quantos bytes por linha de texto no arquivo gerado


def bytes_to_cpp_string_literal(data: bytes, bytes_per_line: int = BYTES_PER_LINE) -> str:
    """
    Converte uma sequencia de bytes em um literal de string C++ usando
    escapes octais de 3 digitos (\\ooo), que sao sempre inequivocos
    (ao contrario de escapes hexadecimais, que tem tamanho variavel).
    Quebra em varias linhas -- o C++ concatena strings adjacentes
    automaticamente, entao o resultado final e uma unica string continua.
    """
    lines = []
    for i in range(0, len(data), bytes_per_line):
        chunk = data[i:i + bytes_per_line]
        escaped = "".join(f"\\{b:03o}" for b in chunk)
        lines.append(f'  "{escaped}"')
    return "\n".join(lines)


def validate_mfcc_pipeline():
    """
    Garante, com um sinal de teste, que reconstruir o MFCC "na mao"
    (STFT -> filtro mel -> log -> DCT) usando as matrizes que vamos
    exportar da o MESMO resultado do librosa.feature.mfcc().
    Se a diferenca for maior que uma tolerancia pequena, o script para.
    """
    print("Validando pipeline de MFCC manual vs. librosa.feature.mfcc()...")

    rng = np.random.default_rng(0)
    y = rng.uniform(-1, 1, WINDOW_SAMPLES).astype(np.float32)

    mfcc_ref = librosa.feature.mfcc(
        y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )

    stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    power_spec = np.abs(stft) ** 2

    mel_fb = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS)
    mel_spec = mel_fb @ power_spec

    log_mel = 10.0 * np.log10(np.maximum(1e-10, mel_spec))
    log_mel = np.maximum(log_mel, log_mel.max() - 80.0)

    dct_mat = dct(np.eye(N_MELS), axis=0, type=2, norm="ortho")[:N_MFCC, :]
    mfcc_manual = dct_mat @ log_mel

    diff = np.abs(mfcc_ref - mfcc_manual)
    print(f"  Shape esperado: (40, 63) | Shape obtido: {mfcc_ref.shape}")
    print(f"  Diferenca maxima: {diff.max():.2e}")

    assert mfcc_ref.shape == (N_MFCC, N_FRAMES_EXPECTED), (
        f"Shape inesperado: {mfcc_ref.shape}. Confira SAMPLE_RATE/N_FFT/HOP_LENGTH."
    )
    assert diff.max() < 1e-3, (
        "Diferenca grande demais entre MFCC manual e o do librosa! "
        "NAO prossiga -- avise antes de gerar os arquivos."
    )

    print("  OK -- pipeline validado, seguro para exportar.\n")
    return mel_fb, dct_mat


def export_mfcc_tables(mel_fb, dct_mat):
    """Exporta mel_fb (128x257) e dct_mat (40x128) como strings C++ (bytes float32)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "mfcc_tables.h")

    n_freq_bins = N_FFT // 2 + 1  # 257

    mel_bytes = mel_fb.astype("<f4").tobytes()   # float32 little-endian, row-major
    dct_bytes = dct_mat.astype("<f4").tobytes()

    with open(path, "w") as f:
        f.write("// Arquivo gerado automaticamente por 13_export_arduino_assets.py\n")
        f.write("// NAO editar manualmente.\n")
        f.write("#ifndef MFCC_TABLES_H\n#define MFCC_TABLES_H\n\n")
        f.write("#include <string.h>  // memcpy\n\n")

        f.write(f"#define MFCC_N_MELS {N_MELS}\n")
        f.write(f"#define MFCC_N_MFCC {N_MFCC}\n")
        f.write(f"#define MFCC_N_FREQ_BINS {n_freq_bins}\n")
        f.write(f"#define MFCC_N_FFT {N_FFT}\n")
        f.write(f"#define MFCC_HOP_LENGTH {HOP_LENGTH}\n")
        f.write(f"#define MFCC_N_FRAMES {N_FRAMES_EXPECTED}\n")
        f.write(f"#define MFCC_SAMPLE_RATE {SAMPLE_RATE}\n\n")

        f.write(f"// Banco de filtros mel ({N_MELS} x {n_freq_bins}), como bytes float32 brutos\n")
        f.write("alignas(4) const char mel_filterbank_raw[] =\n")
        f.write(bytes_to_cpp_string_literal(mel_bytes) + ";\n\n")

        f.write(f"// Matriz DCT-II ortonormal ({N_MFCC} x {N_MELS}), como bytes float32 brutos\n")
        f.write("alignas(4) const char dct_matrix_raw[] =\n")
        f.write(bytes_to_cpp_string_literal(dct_bytes) + ";\n\n")

        f.write(
            "// Funcoes de acesso (usam memcpy para ler com seguranca, sem\n"
            "// depender de alinhamento de memoria -- NUNCA acesse\n"
            "// mel_filterbank_raw/dct_matrix_raw diretamente, use estas funcoes)\n"
        )
        f.write(
            "inline float mel_filterbank(int mel_idx, int freq_idx) {\n"
            "  float v;\n"
            "  memcpy(&v, mel_filterbank_raw + (mel_idx * MFCC_N_FREQ_BINS + freq_idx) * 4, sizeof(float));\n"
            "  return v;\n"
            "}\n\n"
        )
        f.write(
            "inline float dct_matrix(int mfcc_idx, int mel_idx) {\n"
            "  float v;\n"
            "  memcpy(&v, dct_matrix_raw + (mfcc_idx * MFCC_N_MELS + mel_idx) * 4, sizeof(float));\n"
            "  return v;\n"
            "}\n\n"
        )

        f.write("#endif // MFCC_TABLES_H\n")

    size_kb = os.path.getsize(path) / 1024
    print(f"OK -> {path} ({size_kb:.1f} KB)")


def export_model_header():
    """Converte o .tflite em uma string C++ de bytes (model.h)."""
    if not os.path.exists(TFLITE_MODEL_PATH):
        raise FileNotFoundError(
            f"Nao encontrei {TFLITE_MODEL_PATH}. Ajuste TFLITE_MODEL_PATH no script."
        )

    with open(TFLITE_MODEL_PATH, "rb") as f:
        model_bytes = f.read()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "model.h")

    with open(path, "w") as f:
        f.write("// Arquivo gerado automaticamente por 13_export_arduino_assets.py\n")
        f.write("// NAO editar manualmente.\n")
        f.write("#ifndef MODEL_H\n#define MODEL_H\n\n")
        f.write(f"// Modelo original: {TFLITE_MODEL_PATH} ({len(model_bytes)} bytes)\n")
        f.write("alignas(8) const char g_drone_model_data_raw[] =\n")
        f.write(bytes_to_cpp_string_literal(model_bytes) + ";\n\n")
        f.write(
            "const unsigned char* g_drone_model_data = "
            "reinterpret_cast<const unsigned char*>(g_drone_model_data_raw);\n"
        )
        f.write(f"const unsigned int g_drone_model_data_len = {len(model_bytes)};\n\n")
        f.write("#endif // MODEL_H\n")

    size_kb = os.path.getsize(path) / 1024
    print(f"OK -> {path} ({size_kb:.1f} KB) | modelo original: {len(model_bytes)/1024:.1f} KB")


if __name__ == "__main__":
    mel_fb, dct_mat = validate_mfcc_pipeline()
    export_mfcc_tables(mel_fb, dct_mat)
    export_model_header()

    print("\n" + "=" * 70)
    print(f"DONE! model.h and mfcc_tables.h written directly to '{OUTPUT_DIR}/'.")
    print("Open that folder in the Arduino IDE and upload.")
    print("=" * 70)