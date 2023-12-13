import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tkinter as tk
from PIL import Image, ImageTk

# Carga tu modelo previamente entrenado
modelo = load_model('/home/luis/Desktop/content/carpeta_salida/cubiertos/1')

def categorizar_frame(frame):
    # Preprocesamiento de la imagen del fotograma
    frame = cv2.resize(frame, (224, 224))
    frame = frame.astype(float) / 255.0

    # Realiza la predicción
    prediccion = modelo.predict(frame.reshape(-1, 224, 224, 3))

    # Devuelve la categoría predicha
    return np.argmax(prediccion[0], axis=-1)

def mostrar_frame():
    ret, frame = captura.read()

    # Realiza la categorización del fotograma
    categoria = categorizar_frame(frame)

    # Muestra la categoría predicha en la interfaz
    label.config(text=f'Categoría: {categoria}')

    # Convierte el frame de OpenCV a formato RGB para Tkinter
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)

    # Actualiza la etiqueta de la imagen en la interfaz
    panel.img = img
    panel.config(image=img)

    # Llama a esta función de nuevo después de 10 milisegundos
    root.after(10, mostrar_frame)

# Inicializa la interfaz gráfica
root = tk.Tk()
root.title("Categorización en tiempo real")

# Inicializa la captura de video desde la cámara web
captura = cv2.VideoCapture(0)

# Crea una etiqueta para mostrar la imagen
panel = tk.Label(root)
panel.pack(side="top", fill="both", expand="yes")

# Crea una etiqueta para mostrar la categoría
label = tk.Label(root, text="Categoría: ")
label.pack(side="bottom", fill="both", expand="yes")

# Crea un botón para salir de la aplicación
btn_salir = tk.Button(root, text="Salir", command=root.destroy)
btn_salir.pack(side="bottom")

# Llama a la función mostrar_frame para comenzar la captura y categorización
mostrar_frame()

# Inicia el bucle principal de la interfaz
root.mainloop()

# Libera los recursos
captura.release()
cv2.destroyAllWindows()
