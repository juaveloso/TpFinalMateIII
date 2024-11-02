import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Cargo los datos
df = pd.read_csv("dataset.csv", delimiter=",")

# Extraigo variables de entrada (todas las filas, todas las columnas menos la última)
X = df.values[:, :-1]

# Extraigo columna de salida (todas las filas, última columna)
Y = df.values[:, -1]

# Normalizo los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Separo los datos de entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=1/3)

# Creo el clasificador
nn = MLPClassifier(solver='sgd',
                   hidden_layer_sizes=(9, ),
                   activation='relu',
                   max_iter=50_000,
                   learning_rate_init=.05,
                   verbose=False)  # Desactivar los mensajes de entrenamiento

# Entreno el modelo
nn.fit(X_train, Y_train)

# Función para hacer propagación hacia adelante usando los pesos aprendidos
def forward_prop(custom_input):

    # Escalo la entrada
    custom_input = scaler.transform([custom_input])  # Escalar la entrada

    # Extraigo los pesos y sesgos de la red entrenada
    weights_input_hidden = nn.coefs_[0]  # Pesos entre capa de entrada y capa oculta
    weights_hidden_output = nn.coefs_[1]  # Pesos entre capa oculta y capa de salida
    bias_hidden = nn.intercepts_[0]  # Sesgo de la capa oculta
    bias_output = nn.intercepts_[1]  # Sesgo de la capa de salida
    
    # Activación ReLU
    def relu(x):
        return np.maximum(0, x)
    
    # Activación Sigmoide
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Convertir la entrada personalizada en un array numpy
    custom_input = np.array(custom_input).reshape(1, -1)  # Redimensionar para que tenga la forma correcta

    # Cálculo de la capa oculta
    hidden_layer_input = np.dot(custom_input, weights_input_hidden) + bias_hidden
    hidden_layer_output = relu(hidden_layer_input)

    # Cálculo de la capa de salida
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    output_layer_output = sigmoid(output_layer_input)

    # Determinamos si la predicción indica diabetes o no
    if output_layer_output > 0.5:
        return "Positivo"
    else:
        return "Negativo"

print("Puntaje del conjunto de entrenamiento: %f" % nn.score(X_train, Y_train))
print("Puntaje del conjunto de prueba: %f" % nn.score(X_test, Y_test))
