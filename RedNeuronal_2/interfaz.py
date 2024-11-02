import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import importlib  # Para cargar el archivo modelo2.py dinámicamente

# Cargar y ejecutar modelo2.py
modelo2 = importlib.import_module("modelo2")
from modelo2 import forward_prop  # Importa la función forward_prop después de que modelo2.py se haya ejecutado

# Función para validar si una entrada es un número flotante válido
def es_numero_valido(valor, minimo, maximo, tipo='float'):
    try:
        if tipo == 'int':
            valor = int(valor)
        else:
            valor = float(valor)
        return minimo <= valor <= maximo
    except ValueError:
        return False

# Función para manejar la predicción de diabetes (puedes modificarla para integrar el modelo real)
def predecir_diabetes():
    # Recuperar valores de los inputs
    edad = entry_edad.get()
    hipertension = entry_hipertension.get()
    condicion_cardiaca = entry_condicion_cardiaca.get()
    imc = entry_imc.get()
    hba1c = entry_hba1c.get()
    glucosa = entry_glucosa.get()

    # Validaciones
    if not es_numero_valido(edad, 1, 80, tipo='int'):
        messagebox.showerror("Error de validación", "Cargue valores dentro de los rangos indicados.")
        return
    if hipertension not in ['0', '1']:
        messagebox.showerror("Error de validación", "Cargue valores dentro de los rangos indicados.")
        return
    if condicion_cardiaca not in ['0', '1']:
        messagebox.showerror("Error de validación", "Cargue valores dentro de los rangos indicados.")
        return
    if not es_numero_valido(imc, 10, 96):
        messagebox.showerror("Error de validación", "Cargue valores dentro de los rangos indicados.")
        return
    if not es_numero_valido(hba1c, 3.5, 9):
        messagebox.showerror("Error de validación", "Cargue valores dentro de los rangos indicados.")
        return
    if not es_numero_valido(glucosa, 80, 300, tipo='int'):
        messagebox.showerror("Error de validación", "Cargue valores dentro de los rangos indicados.")
        return
    
    # Convertir inputs a tipos numéricos
    edad = int(edad)
    hipertension = int(hipertension)
    condicion_cardiaca = int(condicion_cardiaca)
    imc = float(imc)
    hba1c = float(hba1c)
    glucosa = int(glucosa)
    
    parametro=[edad,hipertension,condicion_cardiaca,imc,hba1c,glucosa]

    resultado = forward_prop(parametro)

    # Mostrar el resultado en el campo de salida
    entry_resultado.config(state="normal")
    entry_resultado.delete(0, tk.END)
    entry_resultado.insert(0, resultado)
    entry_resultado.config(state="readonly")

# Crear ventana principal
ventana = tk.Tk()
ventana.title("Predicción de Diabetes")

# Crear etiquetas y campos de entrada
label_edad = ttk.Label(ventana, text="Edad (1-80):")
label_edad.grid(row=0, column=0, padx=10, pady=5, sticky="e")
entry_edad = ttk.Entry(ventana)
entry_edad.grid(row=0, column=1, padx=10, pady=5)

label_hipertension = ttk.Label(ventana, text="Hipertensión (0 o 1):")
label_hipertension.grid(row=1, column=0, padx=10, pady=5, sticky="e")
entry_hipertension = ttk.Entry(ventana)
entry_hipertension.grid(row=1, column=1, padx=10, pady=5)

label_condicion_cardiaca = ttk.Label(ventana, text="Condición Cardíaca (0 o 1):")
label_condicion_cardiaca.grid(row=2, column=0, padx=10, pady=5, sticky="e")
entry_condicion_cardiaca = ttk.Entry(ventana)
entry_condicion_cardiaca.grid(row=2, column=1, padx=10, pady=5)

label_imc = ttk.Label(ventana, text="IMC (10-96):")
label_imc.grid(row=3, column=0, padx=10, pady=5, sticky="e")
entry_imc = ttk.Entry(ventana)
entry_imc.grid(row=3, column=1, padx=10, pady=5)

label_hba1c = ttk.Label(ventana, text="HbA1c mmol/mol (3.5-9):")
label_hba1c.grid(row=4, column=0, padx=10, pady=5, sticky="e")
entry_hba1c = ttk.Entry(ventana)
entry_hba1c.grid(row=4, column=1, padx=10, pady=5)

label_glucosa = ttk.Label(ventana, text="Glucosa mg/dl (80-300):")
label_glucosa.grid(row=5, column=0, padx=10, pady=5, sticky="e")
entry_glucosa = ttk.Entry(ventana)
entry_glucosa.grid(row=5, column=1, padx=10, pady=5)

# Botón para predecir diabetes
boton_predecir = ttk.Button(ventana, text="Predecir Diabetes", command=predecir_diabetes)
boton_predecir.grid(row=6, column=0, columnspan=2, padx=10, pady=10)

# Campo de resultado
label_resultado = ttk.Label(ventana, text="Resultado (Diabetes):")
label_resultado.grid(row=7, column=0, padx=10, pady=5, sticky="e")
entry_resultado = ttk.Entry(ventana, state="readonly")
entry_resultado.grid(row=7, column=1, padx=10, pady=5)

# Ejecutar el bucle principal de la interfaz
ventana.mainloop()
