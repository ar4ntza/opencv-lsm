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

## 4. Correr la demo en vivo

```bash
python webcam_demo.py
```

- Coloca tu mano dentro del recuadro verde.
- Presiona `b` para alternar el modo binarizado (blanco y negro), útil si el fondo
  de tu salón distrae mucho al modelo.
- Presiona `q` para salir.

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