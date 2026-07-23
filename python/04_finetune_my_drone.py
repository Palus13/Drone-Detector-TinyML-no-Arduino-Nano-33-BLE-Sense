"""
Fine-tuning da DS-CNN com áudios do drone DJI Mini 4 Pro e do ambiente.
Autor: Ighor Ribeiro
"""

from pathlib import Path
import random
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

DRONE_AUDIO_DIR = Path("dataset/gravacoes_arduino")
DRONE_AUDIO_PATTERN = "drone_*.wav"
BACKGROUND_AUDIO_PATTERN = "bg_*.wav"

MODEL_PATH = "models/lightweight_drone_base_v2.keras"
OUTPUT_MODEL = "models/finetuned_lightweight_drone_arduino_mic.keras"

SR = 16000
WINDOW_SECONDS = 1.0
HOP_SECONDS = 0.5
WINDOW_SAMPLES = int(SR * WINDOW_SECONDS)
HOP_SAMPLES = int(SR * HOP_SECONDS)

N_MFCC = 40
N_FFT = 512
HOP_LENGTH = 256

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ==========================================================
# FUNÇÕES DE PRÉ-PROCESSAMENTO E AUMENTAÇÃO
# ==========================================================

def segment_audio_single_file(audio_path):
    """
    Carrega UM arquivo de audio e recorta em janelas de 1s (hop 0.5s).
    Retorna lista de segmentos (sem rotulo ainda).
    """
    audio, sr = librosa.load(audio_path, sr=SR, mono=True)
    n_windows = max(0, int((len(audio) - WINDOW_SAMPLES) / HOP_SAMPLES) + 1)
    print(f"  {audio_path.name}: {len(audio) / SR:.2f}s -> {n_windows} janelas")

    segments = []
    start = 0
    while start + WINDOW_SAMPLES <= len(audio):
        segments.append(audio[start:start + WINDOW_SAMPLES])
        start += HOP_SAMPLES
    return segments


def segment_audio_from_folder(folder, pattern, label):
    """
    Carrega TODOS os arquivos que casam com 'pattern' dentro de 'folder',
    recorta cada um em janelas de 1s, e junta tudo em uma unica lista.
    Retorna (lista_de_segmentos, label).
    """
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"Nenhum arquivo encontrado em '{folder}' com o padrao '{pattern}'."
        )
    print(f"Carregando {len(files)} arquivo(s) com padrao '{pattern}'...")

    all_segments = []
    for f in files:
        all_segments.extend(segment_audio_single_file(f))

    print(f"Total de janelas geradas para '{pattern}': {len(all_segments)}")
    return all_segments, label


def augment_audio(y):
    """
    Aplica aumentação aleatória: ruído gaussiano, pitch shift, time stretch.
    Retorna o áudio aumentado (com tamanho padronizado).
    """
    # 1. Ruído gaussiano (prob 50%)
    if np.random.rand() > 0.5:
        noise_amp = np.random.uniform(0.001, 0.01)
        noise = np.random.normal(0, noise_amp, len(y))
        y = y + noise

    # 2. Pitch shift (prob 50%)
    if np.random.rand() > 0.5:
        steps = np.random.randint(-3, 4)
        y = librosa.effects.pitch_shift(y, sr=SR, n_steps=steps)

    # 3. Time stretch (prob 50%)
    if np.random.rand() > 0.5:
        rate = np.random.uniform(0.85, 1.15)
        y = librosa.effects.time_stretch(y, rate=rate)
        # Ajusta tamanho para exatamente WINDOW_SAMPLES
        if len(y) < WINDOW_SAMPLES:
            y = np.pad(y, (0, WINDOW_SAMPLES - len(y)))
        else:
            y = y[:WINDOW_SAMPLES]

    # Garantir tamanho exato (caso nenhuma aumentação tenha mudado)
    if len(y) < WINDOW_SAMPLES:
        y = np.pad(y, (0, WINDOW_SAMPLES - len(y)))
    elif len(y) > WINDOW_SAMPLES:
        y = y[:WINDOW_SAMPLES]

    return y


def extract_mfcc(segment):
    """
    Extrai MFCC e normaliza (mesmo procedimento do treino original).
    Retorna array (40, 63, 1)
    """
    mfcc = librosa.feature.mfcc(
        y=segment,
        sr=SR,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    # Normalização por amostra
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
    return mfcc[..., np.newaxis]  # (40, 63, 1)


# ==========================================================
# PREPARA O DATASET
# ==========================================================

print("\n========== PREPARANDO DATASET ==========\n")

# Segmenta todos os arquivos de audio (varias gravacoes cada)
drone_segments, label_drone = segment_audio_from_folder(
    DRONE_AUDIO_DIR, DRONE_AUDIO_PATTERN, 1
)
bg_segments, label_bg = segment_audio_from_folder(
    DRONE_AUDIO_DIR, BACKGROUND_AUDIO_PATTERN, 0
)

# Junta todos os segmentos
all_segments = drone_segments + bg_segments
all_labels = [1] * len(drone_segments) + [0] * len(bg_segments)

print(f"\nTotal de segmentos originais: {len(all_segments)} (Drone: {len(drone_segments)}, BG: {len(bg_segments)})")

# ==========================================================
# AUMENTAÇÃO DE DADOS (cria versões aumentadas)
# ==========================================================

print("\nAplicando aumentação...")

augmented_segments = []
augmented_labels = []

# Para cada segmento, geramos versões aumentadas (exceto se for muito pequeno)
for seg, lab in zip(all_segments, all_labels):
    # Salva o original
    augmented_segments.append(seg)
    augmented_labels.append(lab)

    # Cria 2 versões aumentadas por segmento
    for _ in range(2):
        aug_seg = augment_audio(seg.copy())
        augmented_segments.append(aug_seg)
        augmented_labels.append(lab)

print(
    f"Total após aumentação: {len(augmented_segments)} (Drone: {augmented_labels.count(1)}, BG: {augmented_labels.count(0)})")

# ==========================================================
# EXTRAI MFCC DE TODOS (pode demorar um pouco)
# ==========================================================

print("\nExtraindo MFCCs...")
X = []
y = []
for seg, lab in zip(augmented_segments, augmented_labels):
    X.append(extract_mfcc(seg))
    y.append(lab)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# ==========================================================
# DIVIDE TREINO / VALIDAÇÃO
# ==========================================================

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

print(f"\nTreino: {X_train.shape[0]} amostras")
print(f"Validação: {X_val.shape[0]} amostras")

# ==========================================================
# CARREGA E AJUSTA O MODELO
# ==========================================================

print("\n========== CARREGANDO MODELO ==========")
model = tf.keras.models.load_model(MODEL_PATH)
print("Modelo carregado com sucesso.")

# Congela as camadas de BatchNormalization: elas guardam estatisticas
# (media/variancia) aprendidas no dataset original, diverso. Se deixarmos
# essas camadas treinaveis, o dataset pequeno e pouco variado do
# fine-tuning distorce essas estatisticas rapido, "desaprendendo" a
# normalizacao boa -- e quanto mais camadas de BatchNorm a arquitetura
# tiver, pior esse efeito. As demais camadas (Conv2D, DepthwiseConv2D,
# Dense) continuam treinaveis normalmente.
n_frozen = 0
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
        n_frozen += 1
print(f"Camadas de BatchNormalization congeladas: {n_frozen}")

# Recompila com learning rate um pouco maior (a arquitetura e profunda --
# 1e-5 faz o treino praticamente nao sair do lugar)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\n========== INICIANDO FINE-TUNING ==========")

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=6,
        verbose=1
    )
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=80,
    batch_size=16,
    callbacks=callbacks,
    verbose=1
)

# ==========================================================
# SALVA O MODELO FINAL
# ==========================================================

model.save(OUTPUT_MODEL)
print(f"\nModelo fine-tunado salvo em: {OUTPUT_MODEL}")

# ==========================================================
# AVALIAÇÃO RÁPIDA NA VALIDAÇÃO
# ==========================================================

loss, acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\nAcurácia na validação: {acc:.4f}")

# ==========================================================
# PLOTA GRÁFICOS
# ==========================================================

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Treino")
plt.plot(history.history["val_accuracy"], label="Validação")
plt.title("Acurácia")
plt.xlabel("Época")
plt.ylabel("Acurácia")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Treino")
plt.plot(history.history["val_loss"], label="Validação")
plt.title("Loss")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("logs/finetune_history.png", dpi=150)
plt.close()

print("Gráficos salvos em logs/finetune_history.png")

# Salva histórico em CSV
pd.DataFrame(history.history).to_csv("logs/finetune_history.csv", index=False)

print("\nFine-tuning concluído com sucesso!")