import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cargo y preproceso los datos
all_data = pd.read_csv("dataset.csv", delimiter=',')
L = 0.001  # La tasa de aprendizaje

# Extraigo columnas de entrada
all_inputs = all_data.iloc[:,:-1].values
all_outputs = all_data.iloc[:,-1].values

# Normalizo los datos usando Z-score
mean_vals = np.mean(all_inputs, axis=0)
std_vals = np.std(all_inputs, axis=0)
all_inputs_scaled = (all_inputs - mean_vals) / std_vals

# Divido el dataset en conjuntos de entrenamiento y prueba
def train_test_split(x, y, test_size=0.33): #valor por defecto para test_size sera de 0.33
    # Mezclar los índices aleatoriamente
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices) #np.random.shuffle() reordena los elementos del array de manera aleatoria, pero sin crear un nuevo array. Modifica el array original.

    # Dividir los índices según el tamaño de prueba
    test_set_size = int(test_size * len(indices))
    test_indices = indices[:test_set_size] # indices[:test_set_size] toma los primeros test_set_size elementos de indices.
    train_indices = indices[test_set_size:] # toma los indices restantes para entrenamiento

    # Crear conjuntos de entrenamiento y prueba
    x_train, x_test = x[train_indices], x[test_indices] # Ej: x_train contendra el subconjunto de X que corresponde a los índices presentes en train_indices.
    y_train, y_test = y[train_indices], y[test_indices]

    return x_train, x_test, y_train, y_test

X_train, X_test, Y_train, Y_test=train_test_split(all_inputs_scaled,all_outputs)
n = X_train.shape[0]

# Construyo red neuronal con pesos y sesgos inicializados aleatoriamente
np.random.seed(0)
w_hidden = np.random.rand(9,6)*2-1
w_output = np.random.rand(1,9)*2-1

b_hidden = np.random.rand(9, 1)*2-1
b_output = np.random.rand(1, 1)*2-1

# Funciones de activación
relu = lambda x: np.maximum(x, 0)
logistic = lambda x: 1 / (1 + np.exp(-x))

# Derivadas de las funciones de activación
d_relu = lambda x: x > 0
d_logistic = lambda x: np.exp(-x) / (1 + np.exp(-x)) ** 2

# Ejecuto entradas a través de la red neuronal para obtener salidas predichas
def forward_prop(X):
    Z1 = w_hidden @ X + b_hidden
    A1 = relu(Z1)
    Z2 = w_output @ A1 + b_output
    A2 = logistic(Z2)
    return Z1, A1, Z2, A2

# Devuelvo pendientes para pesos y sesgos usando la regla de la cadena
def backward_prop(Z1, A1, Z2, A2, X, Y):
    dC_dA2 = 2 * A2 - 2 * Y
    dA2_dZ2 = d_logistic(Z2)
    dZ2_dA1 = w_output
    dZ2_dW2 = A1
    dZ2_dB2 = 1
    dA1_dZ1 = d_relu(Z1)
    dZ1_dW1 = X
    dZ1_dB1 = 1

    dC_dW2 = dC_dA2 @ dA2_dZ2 @ dZ2_dW2.T
    dC_dB2 = dC_dA2 @ dA2_dZ2 * dZ2_dB2
    dC_dA1 = dC_dA2 @ dA2_dZ2 @ dZ2_dA1
    dC_dW1 = dC_dA1 @ dA1_dZ1 @ dZ1_dW1.T
    dC_dB1 = dC_dA1 @ dA1_dZ1 * dZ1_dB1

    return dC_dW1, dC_dB1, dC_dW2, dC_dB2

# Inicializo listas para almacenar la precisión y pérdida por época
train_accuracy = []
test_accuracy = []
train_loss = []
test_loss = []

# Función de pérdida (error cuadrático medio)
def compute_loss(Y, Y_pred):
    return np.mean((Y - Y_pred) ** 2)

# Ejecuto descenso de gradiente
epochs = 50_000
for i in range(epochs):
    # seleccionar aleatoriamente uno de los datos de entrenamiento
    idx = np.random.choice(n, 1, replace=False)
    X_sample = X_train[idx].transpose()
    Y_sample = Y_train[idx]

    # pasar datos seleccionados aleatoriamente a través de la red neuronal
    Z1, A1, Z2, A2 = forward_prop(X_sample)

    # distribuir error a través de la retropropagación
    # y devolver pendientes para pesos y sesgos
    dW1, dB1, dW2, dB2 = backward_prop(Z1, A1, Z2, A2, X_sample, Y_sample)

    # actualizar pesos y sesgos
    w_hidden -= L * dW1
    b_hidden -= L * dB1
    w_output -= L * dW2
    b_output -= L * dB2

    # Cálculo de precisión y pérdida en cada época
    if i % 10 == 0:
        # Calcular predicciones y pérdidas para el set de entrenamiento
        train_predictions = forward_prop(X_train.transpose())[3]
        train_loss.append(compute_loss(Y_train, train_predictions.flatten()))
        train_accuracy.append(np.mean((train_predictions.flatten() >= 0.5) == Y_train))

        # Calcular predicciones y pérdidas para el set de prueba
        test_predictions = forward_prop(X_test.transpose())[3]
        test_loss.append(compute_loss(Y_test, test_predictions.flatten()))
        test_accuracy.append(np.mean((test_predictions.flatten() >= 0.5) == Y_test))

# Grafico precisión y pérdida para los conjuntos de entrenamiento y prueba
epochs_range = range(0, epochs, 10)
plt.figure(figsize=(12, 5))

# Gráfico de precisión
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_accuracy, label="Entrenamiento")
plt.plot(epochs_range, test_accuracy, label="Validación")
plt.xlabel("Épocas")
plt.ylabel("Precisión")
plt.title("Precisión a lo largo de las épocas")
plt.legend()

# Gráfico de pérdida
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_loss, label="Entrenamiento")
plt.plot(epochs_range, test_loss, label="Validación")
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.title("Pérdida a lo largo de las épocas")
plt.legend()
plt.tight_layout()
plt.show()
