# Importamos las librerías necesarias
import cv2  # Importamos OpenCV para el procesamiento de imágenes
import os  # Importamos el módulo os para operaciones del sistema operativo

# Importamos la clase de seguimiento de manos desde el módulo SeguimientoManos
import SeguimientoManos as sm

# Creamos la carpeta para almacenar las imágenes
nombre = 'Gracias'
direccion = 'C:/Users/marco/OneDrive/Escritorio/peaton/data'
carpeta = direccion + '/' + nombre

# Si la carpeta no existe, la creamos
if not os.path.exists(carpeta):
    print('Carpeta creada:', carpeta)
    os.makedirs(carpeta)

# Lectura de la cámara
cap = cv2.VideoCapture(0)

# Establecemos la resolución de la imagen
cap.set(3, 1280)
cap.set(4, 720)

cont = 0  # Contador para el nombre de las imágenes
# Declaración del detector de manos
detector = sm.detectormanos(Confdeteccion=0.9)

while True:
    # Realizamos la lectura de la cámara
    ret, frame = cap.read()
    
    # Extraemos información de las manos en el frame
    frame = detector.encontrarmanos(frame, dibujar=False)

    # Posición de una sola mano
    lista1, bbox, mano = detector.encontrarposicion(frame, ManoNum=0, dibujarPuntos=False, dibujarBox=False, color=[0, 255, 0])

    if mano == 1:  # Si se detecta una mano

        # Ajustamos las coordenadas para obtener un recorte más grande del área de la mano
        xmin, ymin, xmax, ymax = bbox
        xmin = xmin - 40
        ymin = ymin - 40
        xmax = xmax + 40
        ymax = ymax + 40

        recorte = frame[ymin:ymax, xmin:xmax]  # Recortamos el área de la mano del frame original

        cv2.imwrite(carpeta + "/gracias_{}.jpg".format(cont), recorte)  # Guardamos el recorte como una imagen

        cont = cont + 1  # Incrementamos el contador de imágenes

        # Mostramos el recorte
        cv2.imshow('RECORTE', recorte)

        # Mostramos el frame con el nombre 'Señas Peatonales'
        cv2.imshow('Señas Peatonales', frame)

    t = cv2.waitKey(1)  # Capturamos la tecla presionada
    if t == 27 or cont == 30:  # Si se presiona la tecla 'Esc' (código 27 en ASCII) o se capturan 30 imágenes, salimos del bucle
        break

cap.release()  # Liberamos la cámara
cv2.destroyAllWindows()  # Cerramos todas las ventanas
