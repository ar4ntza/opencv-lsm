"""
webcam_demo.py
----------------
Demo en vivo: captura video de la webcam, recorta una región de interés (ROI)
donde el usuario coloca la mano, y usa la CNN entrenada (train_model.py) para
predecir qué letra del alfabeto en lengua de señas se está mostrando.

Controles:
  - Coloca tu mano dentro del recuadro verde.
  - Presiona 'q' para salir.
  - Presiona 'b' para alternar el fondo binarizado (ayuda si el fondo distrae al modelo).

Requiere:
  - model/asl_cnn.h5   (generado por train_model.py)
  - model/labels.json  (generado por train_model.py)
"""

import json
import os
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = os.path.join("model", "asl_cnn.h5")
LABELS_PATH = os.path.join("model", "labels.json")
IMG_SIZE = 28

# Coordenadas del recuadro (ROI) donde el usuario coloca la mano.
# Ajusta estos valores según la resolución de tu cámara.
ROI_TOP, ROI_BOTTOM = 100, 400
ROI_LEFT, ROI_RIGHT = 350, 650

CONFIDENCE_THRESHOLD = 0.60  # por debajo de esto, mostramos "..."


def cargar_modelo_y_etiquetas():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No se encontró '{MODEL_PATH}'. Corre primero train_model.py."
        )
    modelo = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, "r") as f:
        letras = json.load(f)
    return modelo, letras


def preprocesar_roi(roi, usar_binarizacion=False):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    if usar_binarizacion:
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    normalized = resized.astype("float32") / 255.0
    return normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1), resized


def main():
    modelo, letras = cargar_modelo_y_etiquetas()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo acceder a la webcam. Revisa permisos/índice de cámara.")

    usar_binarizacion = False
    print("Presiona 'q' para salir, 'b' para alternar binarización.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer el frame de la cámara.")
            break

        frame = cv2.flip(frame, 1)  # efecto espejo, más natural para el usuario

        # Dibuja el recuadro guía
        cv2.rectangle(frame, (ROI_LEFT, ROI_TOP), (ROI_RIGHT, ROI_BOTTOM), (0, 255, 0), 2)

        roi = frame[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT]
        entrada, roi_visual = preprocesar_roi(roi, usar_binarizacion)

        pred = modelo.predict(entrada, verbose=0)[0]
        idx = np.argmax(pred)
        confianza = pred[idx]

        if confianza >= CONFIDENCE_THRESHOLD:
            letra = letras[idx]
            texto = f"{letra} ({confianza*100:.1f}%)"
            color = (0, 255, 0)
        else:
            texto = f"... ({confianza*100:.1f}%)"
            color = (0, 165, 255)

        cv2.putText(frame, texto, (ROI_LEFT, ROI_TOP - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        cv2.imshow("Demo CNN - Lengua de Senas (webcam)", frame)

        # Muestra también lo que "ve" el modelo, útil para explicar al público
        roi_grande = cv2.resize(roi_visual, (200, 200), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Entrada al modelo (28x28)", roi_grande)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("b"):
            usar_binarizacion = not usar_binarizacion
            print(f"Binarización: {'activada' if usar_binarizacion else 'desactivada'}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()