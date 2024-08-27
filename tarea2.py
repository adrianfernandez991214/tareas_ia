import numpy as np

""" Paso Forward """

# Función de activación Sigmoide
def sigmoide(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la función sigmoide
def derivada_sigmoide(x):
    return x * (1 - x)

# Inicialización de pesos aleatorios
np.random.seed(1)
pesos_entrada_oculta = np.random.rand(2, 3)  # 2 entradas, 3 neuronas ocultas
pesos_oculta_salida = np.random.rand(3, 1)  # 3 neuronas ocultas, 1 salida

# Datos de entrada (XOR, por ejemplo)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Valores de salida esperados
y = np.array([[0], [1], [1], [0]])

# Paso forward
entrada_capa_oculta = np.dot(X, pesos_entrada_oculta)
salida_capa_oculta = sigmoide(entrada_capa_oculta)

entrada_capa_salida = np.dot(salida_capa_oculta, pesos_oculta_salida)
salida_predicha = sigmoide(entrada_capa_salida)

print("Salida predicha:\n", salida_predicha)

""" Retropropagación """

# Parámetros
tasa_aprendizaje = 0.5
epocas = 10000

for _ in range(epocas):
    # Paso forward
    entrada_capa_oculta = np.dot(X, pesos_entrada_oculta)
    salida_capa_oculta = sigmoide(entrada_capa_oculta)

    entrada_capa_salida = np.dot(salida_capa_oculta, pesos_oculta_salida)
    salida_predicha = sigmoide(entrada_capa_salida)

    # Cálculo del error
    error = y - salida_predicha

    # Retropropagación
    d_salida_predicha = error * derivada_sigmoide(salida_predicha)

    error_capa_oculta = d_salida_predicha.dot(pesos_oculta_salida.T)
    d_capa_oculta = error_capa_oculta * derivada_sigmoide(salida_capa_oculta)

    # Actualización de pesos
    pesos_oculta_salida += salida_capa_oculta.T.dot(d_salida_predicha) * tasa_aprendizaje
    pesos_entrada_oculta += X.T.dot(d_capa_oculta) * tasa_aprendizaje

print("Salida predicha después del entrenamiento:\n", salida_predicha)
