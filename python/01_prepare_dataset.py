"""
Prepara o DroneAudioDataset para treinamento.

Saída:

processed/
    dataset.npz
    metadata.csv

Autor: Ighor Ribeiro
"""

from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from tqdm import tqdm

# ============================================================
# CONFIGURAÇÕES
# ============================================================

DATASET = Path("dataset")

OUTPUT = Path("processed")
OUTPUT.mkdir(exist_ok=True)

TARGET_SR = 16000
TARGET_SAMPLES = 16000

N_MFCC = 40

# ============================================================

X = []
y = []
metadata = []

sample_id = 0

classes = {
    "unknown": 0,
    "yes_drone": 1
}

print("\nLendo dataset...\n")

for classe in classes:

    pasta = DATASET / classe

    arquivos = sorted(pasta.glob("*.wav"))

    for arquivo in tqdm(arquivos, desc=classe):

        try:

            audio, sr = librosa.load(
                arquivo,
                sr=TARGET_SR,
                mono=True
            )

            # -----------------------------
            # Padroniza duração
            # -----------------------------

            if len(audio) < TARGET_SAMPLES:

                audio = np.pad(
                    audio,
                    (0, TARGET_SAMPLES - len(audio))
                )

            elif len(audio) > TARGET_SAMPLES:

                audio = audio[:TARGET_SAMPLES]

            # -----------------------------
            # MFCC
            # -----------------------------

            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=TARGET_SR,
                n_mfcc=N_MFCC,
                n_fft=512,
                hop_length=256
            )

            # Normalização

            mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)

            X.append(mfcc.astype(np.float32))

            y.append(classes[classe])

            metadata.append({
                "arquivo": arquivo.name,
                "classe": classe,
                "caminho": str(arquivo.resolve()),
                "sample_rate": TARGET_SR,
                "samples": len(audio)
            })
            
            sample_id += 1

        except Exception as e:

            print(f"Erro: {arquivo.name}")
            print(e)

# ============================================================
# Converte para numpy
# ============================================================

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

metadata = pd.DataFrame(metadata)

print("\n================================")
print("Dataset preparado")
print("================================")

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

print("\nArquivos por classe:")
print(metadata["classe"].value_counts())

# ============================================================
# Salva dataset completo
# ============================================================

np.savez_compressed(
    OUTPUT / "dataset.npz",
    X=X,
    y=y,
    file_names=metadata["arquivo"].to_numpy(),
    class_names=metadata["classe"].to_numpy(),
    file_paths=metadata["caminho"].to_numpy()
)

metadata.to_csv(
    OUTPUT / "metadata.csv",
    index=False
)

print("\nArquivos salvos em:")
print(OUTPUT.resolve())