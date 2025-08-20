import sounddevice as sd
import numpy as np
import pickle
import serial
import time


# Configuración Arduino

arduino = serial.Serial('COM3', 9600, timeout=1)  # Ajusta COM según tu sistema
time.sleep(2)  # Espera a que Arduino se inicialice


# Cargar modelo entrenado

with open("modelos/modelo_knn.pkl", "rb") as f:
    clf = pickle.load(f)

# Configuración de audio
fs = 44100      # Frecuencia de muestreo
duracion = 2    # Segundos por muestra


# Funciones
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


# Detección en tiempo real

print("Detector activo, Habla ahora... (Ctrl+C para salir)")

try:
    while True:
        # Captura audio
        audio = grabar_muestra()
        # Calcula características
        car = np.array(calcular_caracteristicas(audio)).reshape(1, -1)
        # Predicción de nivel
        pred = clf.predict(car)[0]
        print(f"Nivel detectado: {pred}")
        # Enviar a Arduino
        arduino.write((pred + "\n").encode())
        time.sleep(0.2)  # Pequeña pausa para no saturar la consola

except KeyboardInterrupt:
    print("\nDetector detenido")
