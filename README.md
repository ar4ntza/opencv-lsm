# Demo: CNN para reconocimiento de lengua de señas (ASL) con webcam

Proyecto de ejemplo para presentación sobre **Redes Neuronales Convolucionales (CNN)**
y **Visión por Computadora**. Entrena un modelo con el dataset *Sign Language MNIST*
y lo usa para clasificar letras del alfabeto en lengua de señas en tiempo real desde
la webcam.

## Levantar el proyecto

Windows:

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
macOS y Linux:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

## 1. Instalación

```bash
pip install -r requirements.txt
```

## 2. Descargar el dataset

1. Ve a: https://www.kaggle.com/datasets/datamunge/sign-language-mnist
2. Descarga `sign_mnist_train.csv` y `sign_mnist_test.csv`
3. Colócalos en una carpeta `data/` dentro de este proyecto:

```
asl-cnn-demo/
├── data/
│   ├── sign_mnist_train.csv
│   └── sign_mnist_test.csv
├── train_model.py
├── webcam_demo.py
└── ...
```

## 3. Entrenar el modelo

```bash
python train_model.py
```

Esto genera:
- `model/asl_cnn.h5` — pesos del modelo entrenado
- `model/labels.json` — mapa de índices a letras

**Importante:** entrena esto ANTES de tu presentación, no en vivo. El entrenamiento
puede tardar varios minutos según tu hardware.

## 4. Correr la demo en vivo

```bash
python webcam_demo.py
```

- Coloca tu mano dentro del recuadro verde.
- Presiona `b` para alternar el modo binarizado (blanco y negro), útil si el fondo
  de tu salón distrae mucho al modelo.
- Presiona `q` para salir.

## Notas para la presentación

- El dataset no incluye las letras **J** y **Z** porque en lengua de señas real
  requieren movimiento, no solo una postura estática — buen dato para mencionar
  al público.
- Si el modelo falla mucho en vivo, casi siempre es un problema de **iluminación**
  o de que el **fondo real es distinto al fondo del dataset** (que son imágenes
  ya recortadas y centradas en la mano). Es un buen punto para hablar de la
  importancia de la calidad y variedad de los datos de entrenamiento.
- Ten un video de respaldo grabado por si hay problemas técnicos de último minuto
  (cámara, drivers, permisos del sistema operativo).
- Considera limitar la demo en vivo a 5-8 letras claramente distintas entre sí
  (evita letras muy parecidas como M, N, S) para una demo más robusta y confiable.

## Estructura del modelo

CNN simple con 3 bloques convolucionales:

```
Input (28x28x1)
 → Conv2D(32) + BatchNorm + MaxPool
 → Conv2D(64) + BatchNorm + MaxPool
 → Conv2D(128) + BatchNorm + MaxPool
 → Flatten
 → Dense(256) + Dropout(0.5)
 → Dense(24, softmax)
```

Suficientemente simple para explicar en una presentación, pero con Batch
Normalization y Dropout para ilustrar buenas prácticas de regularización.