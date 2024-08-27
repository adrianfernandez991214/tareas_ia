import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler

# Generar datos
np.random.seed(42)
X = np.random.randn(100, 3)
y = X @ np.array([1.5, -2.0, 1.0]) + np.random.randn(100) * 0.5

# Crear multicolinealidad
X[:, 2] = X[:, 0] + X[:, 1]

# Estandarizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Regresión lineal sin regularización
lin_reg = LinearRegression()
lin_reg.fit(X_scaled, y)
coef_lin = lin_reg.coef_

# Regresión Ridge con regularización
ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_scaled, y)
coef_ridge = ridge_reg.coef_

# Mostrar los coeficientes
print("Coeficientes de regresión lineal:", coef_lin)
print("Coeficientes de Ridge Regression:", coef_ridge)

# Visualizar los coeficientes
plt.figure(figsize=(10, 5))
plt.bar(np.arange(len(coef_lin)), coef_lin, alpha=0.6, label="Regresión Lineal")
plt.bar(np.arange(len(coef_ridge)), coef_ridge, alpha=0.6, label="Ridge Regression")
plt.legend()
plt.xlabel("Índice de Coeficientes")
plt.ylabel("Valor de Coeficiente")
plt.title("Comparación de Coeficientes: Lineal vs Ridge")
plt.show()
