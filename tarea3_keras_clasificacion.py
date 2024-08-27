# Keras: Simplicidad y Facilidad de Uso
# PyTorch: Flexibilidad y Control(Investigación)

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# Cargar los datos (conjunto de datos MNIST, que es un problema clásico de clasificación de dígitos escritos a mano)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalizar y convertir etiquetas a one-hot encoding
X_train, X_test = X_train / 255.0, X_test / 255.0
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# División adicional para validación
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Definir el modelo
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)

# Evaluar el modelo
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Accuracy en Test: {test_acc}")

# Predicciones y reporte de clasificación
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)
print(classification_report(y_test_classes, y_pred_classes))
