"""
14_train_lightweight_dscnn.py

Retreina o modelo do ZERO (no dataset original, o mesmo usado no
04_train_dscnn.py) com uma arquitetura redesenhada: as camadas agora usam
strides=2, o que reduz a resolucao espacial do mapa de ativacoes
progressivamente (40x63 -> 20x32 -> 10x16 -> 5x8), em vez de manter
40x63 constante ate o final como na arquitetura original.

Isso resolve o erro "Failed to resize buffer. Requested: 1290240..." que
aconteceu no Arduino: a arquitetura original nunca reduzia o tamanho
espacial, entao os mapas intermediarios ficavam grandes demais (~1.29MB)
para caber nos 256KB de RAM do nRF52840.

Depois de rodar este script, o proximo passo e refazer o fine-tuning
(09_finetune_my_drone.py) usando como base o arquivo gerado aqui
(models/lightweight_drone_base.keras) no lugar do best_model.keras antigo.

Requisitos: mesmo ambiente ja usado no projeto (tensorflow, numpy).
"""

import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# --------------------------------------------------------------------------
# CONFIGURACAO
# --------------------------------------------------------------------------
DATASET_NPZ_PATH = "processed/dataset.npz"
OUTPUT_MODEL_PATH = "models/lightweight_drone_base_v2.keras"

INPUT_SHAPE = (40, 63, 1)
EPOCHS = 40
BATCH_SIZE = 32
LEARNING_RATE = 1e-3


from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


def load_dataset():
    """
    Carrega o dataset.npz no formato real gerado pelo 03_prepare_dataset.py:
    chaves 'X' (N, 40, 63) e 'y' (N,) -- SEM splits prontos.
    Faz um split estratificado 70/15/15 (treino/val/teste) aqui mesmo,
    preservando a proporcao de classes em cada parte.
    """
    if not os.path.exists(DATASET_NPZ_PATH):
        raise FileNotFoundError(
            f"Nao encontrei {DATASET_NPZ_PATH}. Ajuste DATASET_NPZ_PATH no script."
        )
    data = np.load(DATASET_NPZ_PATH, allow_pickle=True)
    print(f"Chaves encontradas em {DATASET_NPZ_PATH}: {list(data.keys())}")

    X, y = data["X"], data["y"]
    print(f"  X: {X.shape} | y: {y.shape}")
    classes, counts = np.unique(y, return_counts=True)
    print(f"  Distribuicao de classes: {dict(zip(classes.tolist(), counts.tolist()))}")

    if X.ndim == 3:
        X = X[..., np.newaxis]
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    # 70% treino / 15% validacao / 15% teste, mantendo a proporcao de classes
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    print(f"  Treino: {X_train.shape[0]} | Validacao: {X_val.shape[0]} | Teste: {X_test.shape[0]}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def compute_weights(y_train):
    """Calcula pesos de classe para compensar o desbalanceamento (poucos exemplos de drone)."""
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    print(f"  Pesos de classe calculados: {class_weight_dict}")
    return class_weight_dict


def build_lightweight_dscnn(input_shape=INPUT_SHAPE):
    """
    Arquitetura DS-CNN com reducao progressiva de resolucao espacial via
    strides=2 em 4 blocos (em vez de 2, como na primeira tentativa) --
    isso permite usar MAIS canais (mais capacidade de aprendizado) sem
    aumentar o tamanho dos tensores intermediarios, porque a resolucao
    espacial cai mais rapido. Tamanho maximo de tensor intermediario:
    ~60KB (cabe folgadamente nos 256KB de RAM do Arduino Nano 33 BLE).

    Motivo da mudanca: a primeira versao (2 blocos, canais 16/32/32,
    2657 parametros) rodou no Arduino mas so acertou 55.87% no audio
    real do drone -- capacidade insuficiente para aprender a tarefa
    (praticamente chute). Esta versao tem 11169 parametros, mais proxima
    da arquitetura original (~8000 parametros, que chegou a 83.5% de
    validacao no fine-tuning).
    """
    inputs = layers.Input(shape=input_shape)

    # Bloco 1: (40,63,1) -> (20,32,24)
    x = layers.Conv2D(24, 3, strides=2, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Bloco 2: (20,32,24) -> (10,16,24) -> (10,16,48)
    x = layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(48, 1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Bloco 3: (10,16,48) -> (5,8,48) -> (5,8,64)
    x = layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, 1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Bloco 4: (5,8,64) -> (3,4,64) -> (3,4,64)
    x = layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, 1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return models.Model(inputs, outputs, name="lightweight_dscnn_v2")


def estimate_max_activation_bytes(model):
    """Estimativa grosseira do maior tensor intermediario, so para conferencia."""
    max_elements = 0
    max_layer_name = ""
    for layer in model.layers:
        try:
            shape = layer.output.shape
        except AttributeError:
            continue
        if shape and None not in shape[1:]:
            n = int(np.prod(shape[1:]))  # ignora dimensao de batch
            if n > max_elements:
                max_elements = n
                max_layer_name = layer.name
    print(
        f"\nMaior tensor intermediario: camada '{max_layer_name}' "
        f"com {max_elements} elementos (~{max_elements * 4 / 1024:.1f} KB em float32)."
    )
    print(f"Pior caso (2 tensores simultaneos na memoria): ~{2 * max_elements * 4 / 1024:.1f} KB")


if __name__ == "__main__":
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_dataset()

    model = build_lightweight_dscnn()
    model.summary()
    estimate_max_activation_bytes(model)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss" if X_val is not None else "loss",
            patience=8,
            restore_best_weights=True,
        )
    ]

    class_weight_dict = compute_weights(y_train)

    validation_data = (X_val, y_val) if X_val is not None else None

    model.fit(
        X_train,
        y_train,
        validation_data=validation_data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weight_dict,
    )

    if X_test is not None:
        from sklearn.metrics import classification_report

        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        print("\nRelatorio de classificacao no conjunto de teste:")
        print(
            classification_report(
                y_test, y_pred, target_names=["unknown/background", "yes_drone"]
            )
        )

    os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH), exist_ok=True)
    model.save(OUTPUT_MODEL_PATH)
    print(f"\nOK -> modelo salvo em {OUTPUT_MODEL_PATH}")
    print("\nPROXIMO PASSO: refaca o fine-tuning (09_finetune_my_drone.py) usando")
    print(f"'{OUTPUT_MODEL_PATH}' como modelo base, no lugar do best_model.keras antigo.")