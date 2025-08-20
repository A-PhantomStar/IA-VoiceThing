import sounddevice as sd
import numpy as np
import os
import time
import pickle

# Configuración
fs = 44100       # Frecuencia de muestreo
duracion = 2     # Segundos por grabación
n_muestras = 5   # Repeticiones por nivel
niveles = ["VERDE", "AMARILLO", "ROJO"]
caracteristicas = []
etiquetas = []

# Crear carpeta de grabaciones si no existe
if not os.path.exists("calibracion"):
    os.mkdir("calibracion")

def grabar_muestra(nombre_archivo):
    print("Grabando...")
    audio = sd.rec(int(duracion * fs), samplerate=fs, channels=1)
    sd.wait()
    audio = audio.flatten()
    np.save(nombre_archivo, audio)  # Guardar como .npy
    return audio

def calcular_caracteristicas(audio):
    rms = np.sqrt(np.mean(audio**2))
    fft = np.fft.fft(audio)
    freq = np.fft.fftfreq(len(fft), 1/fs)
    idx = np.argmax(np.abs(fft[:len(fft)//2]))
    pitch = abs(freq[idx])
    return [rms, pitch]

for nivel in niveles:
    print(f"\n=== Graba el nivel {nivel} ===")
    for i in range(n_muestras):
        input(f"Presiona ENTER y habla ({i+1}/{n_muestras})...")
        nombre_archivo = f"calibracion/{nivel}_{i}.npy"
        audio = grabar_muestra(nombre_archivo)
        car = calcular_caracteristicas(audio)
        caracteristicas.append(car)
        etiquetas.append(nivel)

# Guardar características y etiquetas
with open("modelos/calibracion.pkl", "wb") as f:
    pickle.dump((caracteristicas, etiquetas), f)

print("\nCalibración completada con exito.")
