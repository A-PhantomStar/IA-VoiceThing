import sounddevice as sd
import numpy as np
import pickle
import time

# Cargar el modelo entrenado
with open("modelos/modelo_knn.pkl", "rb") as f:
    clf = pickle.load(f)

fs = 44100      # Frecuencia de muestreo
duracion = 2    # Duración de la captura en segundos

def grabar_muestra():
    audio = sd.rec(int(duracion * fs), samplerate=fs, channels=1)
    sd.wait()
    return audio.flatten()

def calcular_caracteristicas(audio):
    rms = np.sqrt(np.mean(audio**2))
    fft = np.fft.fft(audio)
    freq = np.fft.fftfreq(len(fft), 1/fs)
    idx = np.argmax(np.abs(fft[:len(fft)//2]))
    pitch = abs(freq[idx])
    return [rms, pitch]

print("Detector activo, Habla ahora... (Ctrl+C para salir)")

try:
    while True:
        audio = grabar_muestra()
        car = np.array(calcular_caracteristicas(audio)).reshape(1, -1)
        pred = clf.predict(car)[0]
        print(f"Nivel detectado: {pred}")
        time.sleep(0.2)  # pequeña pausa para no saturar la consola
except KeyboardInterrupt:
    print("\nDetector detenido.")
