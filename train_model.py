"""
train_model.py
----------------
entrena una CNN para reconocer letras del alfabeto en lengua de señas (ASL)
usando el dataset "Sign Language MNIST".

Dataset (descárgalo antes de correr este script):
  https://www.kaggle.com/datasets/datamunge/sign-language-mnist
  Archivos esperados: sign_mnist_train.csv, sign_mnist_test.csv
  Colócalos en la carpeta ./data/

Formato del dataset:
  - Imágenes de 28x28 píxeles en escala de grises.
  - La primera columna es la etiqueta (0-25, representando A-Z),
    SIN incluir J (9) ni Z (25) porque requieren movimiento.
  - Las siguientes 784 columnas son los valores de los píxeles (0-255).

Salida:
  - modelo entrenado guardado en ./model/asl_cnn.h5
  - mapa de índices a letras guardado en ./model/labels.json
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
MODEL_DIR = "model"
IMG_SIZE = 28
NUM_CLASSES = 25  # 26 letras - J y Z no están (requieren movimiento)

os.makedirs(MODEL_DIR, exist_ok=True)


def cargar_datos():
    print("Cargando datasets...")
    train_path = os.path.join(DATA_DIR, "sign_mnist_train.csv")
    test_path = os.path.join(DATA_DIR, "sign_mnist_test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"No se encontraron los CSV en '{DATA_DIR}/'.\n"
            "Descarga el dataset de Kaggle:\n"
            "https://www.kaggle.com/datasets/datamunge/sign-language-mnist\n"
            "y coloca sign_mnist_train.csv y sign_mnist_test.csv en esa carpeta."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    y_train = train_df["label"].values
    y_test = test_df["label"].values

    X_train = train_df.drop("label", axis=1).values
    X_test = test_df.drop("label", axis=1).values

    # Reshape a (N, 28, 28, 1) y normalización a [0, 1]
    X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype("float32") / 255.0
    X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype("float32") / 255.0

    return X_train, y_train, X_test, y_test


def construir_mapa_etiquetas():
    """
    Las etiquetas del dataset van de 0 a 25 (letras A-Z),
    pero J (9) y Z (25) no existen porque requieren movimiento.
    Aun así, el dataset deja esos índices "vacíos". Construimos
    el mapa completo y luego lo re-indexamos a 0..24 sin huecos.
    """
    letras = [chr(ord("A") + i) for i in range(26)]  # A-Z
    letras_validas = [l for l in letras if l not in ("J", "Z")]
    return letras_validas  # 24 letras... pero el dataset real usa 24 clases (0-23 tras remapeo)


def remapear_etiquetas(y, letras_validas):
    """
    El dataset original tiene etiquetas 0-25 saltándose 9 (J) y 25 (Z).
    Remapeamos a un rango denso 0..N-1 para que coincida con la última
    capa softmax del modelo.
    """
    letras_todas = [chr(ord("A") + i) for i in range(26)]
    idx_original_a_letra = {i: letras_todas[i] for i in range(26)}
    letra_a_idx_denso = {letra: idx for idx, letra in enumerate(letras_validas)}

    y_remap = np.array([letra_a_idx_denso[idx_original_a_letra[val]] for val in y])
    return y_remap


def construir_modelo(num_classes):
    modelo = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])

    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


def main():
    letras_validas = construir_mapa_etiquetas()
    num_classes = len(letras_validas)
    print(f"Clases ({num_classes}): {letras_validas}")

    X_train, y_train_raw, X_test, y_test_raw = cargar_datos()

    y_train = remapear_etiquetas(y_train_raw, letras_validas)
    y_test = remapear_etiquetas(y_test_raw, letras_validas)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    # Data augmentation ligero: ayuda a que la CNN generalice mejor
    # a las condiciones de tu propia webcam (que difieren del dataset).
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
    )
    datagen.fit(X_train)

    modelo = construir_modelo(num_classes)
    modelo.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
    ]

    print("Entrenando...")
    modelo.fit(
        datagen.flow(X_train, y_train, batch_size=128),
        validation_data=(X_val, y_val),
        epochs=25,
        callbacks=callbacks,
    )

    print("Evaluando en test...")
    test_loss, test_acc = modelo.evaluate(X_test, y_test)
    print(f"Precisión en test: {test_acc:.4f}")

    modelo_path = os.path.join(MODEL_DIR, "asl_cnn.h5")
    modelo.save(modelo_path)
    print(f"Modelo guardado en: {modelo_path}")

    labels_path = os.path.join(MODEL_DIR, "labels.json")
    with open(labels_path, "w") as f:
        json.dump(letras_validas, f)
    print(f"Mapa de etiquetas guardado en: {labels_path}")


if __name__ == "__main__":
    main()