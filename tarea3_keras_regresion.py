# Keras: Simplicidad y Facilidad de Uso
# PyTorch: Flexibilidad y Control(Investigación)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Cargar datos (Mismos usados en "Fundamentos de ciencia de datos")
df = pd.read_csv('housing.csv')

# Identificar columnas categóricas
columnas_categoricas = df.select_dtypes(include=['object']).columns
print("Columnas categóricas:", columnas_categoricas)

# Convertir columnas categóricas en variables numéricas usando one-hot encoding
df = pd.get_dummies(df, columns=columnas_categoricas, drop_first=True)

# Definir las características (X) y la etiqueta (y)
X = df.drop('median_house_value', axis=1).values
y = df['median_house_value'].values

# Dividir los datos en entrenamiento, validación y prueba
X_entrenamiento, X_temp, y_entrenamiento, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_validacion, X_prueba, y_validacion, y_prueba = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Escalar las características
escalador = StandardScaler()
X_entrenamiento = escalador.fit_transform(X_entrenamiento)
X_validacion = escalador.transform(X_validacion)
X_prueba = escalador.transform(X_prueba)

# Construir la red neuronal en Keras
modelo = Sequential([
    Dense(64, activation='relu', input_shape=(X_entrenamiento.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)  # Una sola salida para regresión(no se cuantas hacen falta)
])

# Compilar el modelo
modelo.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Entrenar el modelo
historial = modelo.fit(X_entrenamiento, y_entrenamiento, epochs=100, batch_size=32, validation_data=(X_validacion, y_validacion))

# Evaluar el modelo en el conjunto de prueba
y_predicho = modelo.predict(X_prueba)
mse = mean_squared_error(y_prueba, y_predicho)
mae = mean_absolute_error(y_prueba, y_predicho)

print(f"Error Cuadrático Medio (MSE) en prueba: {mse}")
print(f"Error Absoluto Medio (MAE) en prueba: {mae}")

# Visualizar el rendimiento
plt.plot(historial.history['loss'], label='Pérdida')
plt.plot(historial.history['val_loss'], label='Pérdida de Validación')
plt.xlabel('Época')
plt.ylabel('Error Cuadrático Medio')
plt.legend()
plt.show()

# Guardar el modelo (Dudas)
modelo.save('modelo_viviendas_california.h5')
