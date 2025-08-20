import numpy as np
import os
import pickle
from sklearn.neighbors import KNeighborsClassifier

# Función para calcular características de un audio .npy
def calcular_caracteristicas(audio):
    rms = np.sqrt(np.mean(audio**2))
    fft = np.fft.fft(audio)
    freq = np.fft.fftfreq(len(fft), 1/44100)  # fs=44100
    idx = np.argmax(np.abs(fft[:len(fft)//2]))
    pitch = abs(freq[idx])
    return [rms, pitch]

# Cargar las grabaciones
caracteristicas = []
etiquetas = []

folder = "calibracion"
for archivo in os.listdir(folder):
    if archivo.endswith(".npy"):
        audio = np.load(os.path.join(folder, archivo))
        car = calcular_caracteristicas(audio)
        caracteristicas.append(car)

        # Extraer etiqueta del nombre del archivo, e.g., "VERDE_0.npy"
        etiqueta = archivo.split("_")[0]
        etiquetas.append(etiqueta)

# Entrenar modelo KNN
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(caracteristicas, etiquetas)

# Guardar el modelo entrenado
if not os.path.exists("modelos"):
    os.mkdir("modelos")
with open("modelos/modelo_knn.pkl", "wb") as f:
    pickle.dump(clf, f)

print("Entrenamiento completado con exito.")
