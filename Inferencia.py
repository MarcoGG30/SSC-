import cv2  # Importamos OpenCV para el procesamiento de imágenes
import os  # Importamos el módulo os para operaciones del sistema operativo
from ultralytics import YOLO  # Importamos YOLO de Ultralytics para la detección de objetos
import SeguimientoManos as sm  # Importamos el módulo SeguimientoManos para el seguimiento de manos

cap = cv2.VideoCapture(0)  # Inicializamos la captura de vídeo desde la cámara
cap.set(3, 1280)  # Establecemos el ancho del frame
cap.set(4, 720)  # Establecemos la altura del frame

model = YOLO('vocales2.pt')  # Cargamos el modelo YOLO pre-entrenado para detectar objetos

detector = sm.detectormanos(Confdeteccion=0.9)  # Inicializamos el detector de manos con un umbral de confianza

while True:
    ret, frame = cap.read()  # Capturamos un frame desde la cámara

    frame = detector.encontrarmanos(frame, dibujar=False)  # Detectamos las manos en el frame sin dibujar puntos o cajas

    lista1, bbox, mano = detector.encontrarposicion(frame, ManoNum=0, dibujarPuntos=False, dibujarBox=False, color=[0, 255, 0])
    # Detectamos la posición de la mano sin dibujar puntos o cajas

    if mano == 1:  # Si se detecta una mano
        xmin, ymin, xmax, ymax = bbox  # Obtenemos las coordenadas del cuadro delimitador

        # Ajustamos las coordenadas para obtener un recorte más grande del área de la mano
        xmin = xmin - 40
        ymin = ymin - 40
        xmax = xmax + 40
        ymax = ymax + 40

        recorte = frame[ymin:ymax, xmin:xmax]  # Recortamos el área de la mano del frame original

        recorte = cv2.resize(recorte, (640, 640), interpolation=cv2.INTER_CUBIC)  # Redimensionamos el recorte a 640x640

        resultados = model.predict(recorte, conf=0.55)  # Realizamos la predicción de objetos en el recorte

        if len(resultados) != 0:  # Si se detectan objetos en el recorte
            for results in resultados:  # Recorremos los resultados de la detección
                masks = results.masks  # Obtenemos las máscaras de los objetos detectados
                coordenadas = masks  # Almacenamos las coordenadas de los objetos detectados

                anotaciones = resultados[0].plot()  # Creamos anotaciones visuales de los objetos detectados

        cv2.imshow('RECORTE', anotaciones)  # Mostramos el recorte con las anotaciones visuales de los objetos detectados

    cv2.imshow('Lenguaje Vocales', frame)  # Mostramos el frame original con el nombre 'Lenguaje Vocales'
    t = cv2.waitKey(1)  # Capturamos la tecla presionada
    if t == 27:  # Si se presiona la tecla 'Esc' (código 27 en ASCII), salimos del bucle
        break

cap.release()  # Liberamos la cámara
cv2.destroyAllWindows()  # Cerramos todas las ventanas
