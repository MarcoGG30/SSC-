# Importamos las librerías necesarias
import math  # Importamos la librería math para operaciones matemáticas
import cv2  # Importamos OpenCV para el procesamiento de imágenes
import mediapipe as mp  # Importamos MediaPipe para el seguimiento de manos
import time  # Importamos la librería time para el manejo del tiempo

# Creamos la clase para el seguimiento de manos
class detectormanos():
    # Inicializamos los parámetros de la clase
    def __init__(self, node=False, maxManos=2, model_complexity=1, Confdeteccion=3.5, Confsegui=0.5):
        self.node = node  # Indica si se ejecuta en modo GPU o CPU
        self.maxManos = maxManos  # Número máximo de manos a detectar
        self.compl = model_complexity  # Complejidad del modelo
        self.Confdeteccion = Confdeteccion  # Umbral de confianza para la detección de manos
        self.Confsegui = Confsegui  # Umbral de confianza para el seguimiento de manos

        # Creamos los objetos que detectarán las manos y dibujarán las líneas
        self.mpmanos = mp.solutions.hands
        self.manos = self.mpmanos.Hands(self.node, self.maxManos, self.compl, self.Confdeteccion, self.Confsegui)
        self.dibujo = mp.solutions.drawing_utils
        self.tip = [4, 8, 12, 16, 20]  # Puntos finales de los dedos

    # Función para detectar manos en un frame
    def encontrarmanos(self, frame, dibujar=True):
        imgcolor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertimos el frame a formato RGB
        self.resultados = self.manos.process(imgcolor)  # Procesamos el frame en busca de manos

        if self.resultados.multi_hand_landmarks:  # Si se detectan manos en el frame
            for mano in self.resultados.multi_hand_landmarks:
                if dibujar:
                    self.dibujo.draw_landmarks(frame, mano, self.mpmanos.HAND_CONNECTIONS) # Dibujamos los puntos y conexiones de la mano
        return frame

    # Función para detectar la posición de las manos
    def encontrarposicion(self, frame, ManoNum=0, dibujarPuntos=True, dibujarBox=True, color=[]):
        xlista = []  # Lista para las coordenadas x de los puntos de la mano
        ylista = []  # Lista para las coordenadas y de los puntos de la mano
        bbox = []  # Lista para las coordenadas del cuadro delimitador
        player = 0  # Contador de manos detectadas
        self.lista = []  # Lista para almacenar información de los puntos de la mano
        if self.resultados.multi_hand_landmarks:
            miMano = self.resultados.multi_hand_landmarks[ManoNum]  # Seleccionamos la mano específica
            prueba = self.resultados.multi_hand_landmarks
            player = len(prueba)  # Número de manos detectadas

            for id, lm in enumerate(miMano.landmark):  # Recorremos los puntos de la mano
                alto, ancho, c = frame.shape  # Dimensiones del frame
                cx, cy = int(lm.x * ancho), int(lm.y * alto)  # Convertimos las coordenadas normalizadas a píxeles
                xlista.append(cx)  # Guardamos la coordenada x
                ylista.append(cy)  # Guardamos la coordenada y
                self.lista.append([id, cx, cy])  # Guardamos la información del punto
                if dibujarPuntos:
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 0), cv2.FILLED)  # Dibujamos un círculo en cada punto

            xmin, xmax = min(xlista), max(xlista)  # Coordenadas x mínima y máxima
            ymin, ymax = min(ylista), max(ylista)  # Coordenadas y mínima y máxima
            bbox = xmin, ymin, xmax, ymax  # Definimos el cuadro delimitador
            if dibujarBox:
                cv2.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), color, 2)  # Dibujamos el cuadro delimitador

        return self.lista, bbox, player  # Devolvemos la lista de puntos, el cuadro delimitador y el número de manos detectadas

    # Función para detectar los dedos levantados
    def dedosarriba(self):
        dedos = []  # Lista para indicar si cada dedo está levantado o no
        if self.lista[self.tip[0]][1] > self.lista[self.tip[0] - 1][1]:
            dedos.append(1)  # El pulgar está levantado si su coordenada x es mayor que la del punto anterior
        else:
            dedos.append(0)

        for id in range(1, 5):
            if self.lista[self.tip[id]][2] < self.lista[self.tip[id] - 2][2]:
                dedos.append(1)  # El resto de los dedos están levantados si su coordenada y es menor que la del punto anterior
            else:
                dedos.append(0)
        return dedos  # Devolvemos la lista de dedos levantados

    # Función para calcular la distancia entre dos puntos
    def distancia(self, p1, p2, frame, dibujar=True, r=15, t=3):
        x1, y1 = self.lista[p1][1:]  # Coordenadas del primer punto
        x2, y2 = self.lista[p2][1:]  # Coordenadas del segundo punto
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Coordenadas del punto medio
        if dibujar:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), t)  # Dibujamos una línea entre los dos puntos
            cv2.circle(frame, (x1, y1), r, (0, 0, 255), cv2.FILLED)  # Dibujamos un círculo en el primer punto
            cv2.circle(frame, (x2, y2), r, (0, 0, 255), cv2.FILLED)  # Dibujamos un círculo en el segundo punto
            cv2.circle(frame, (cx, cy), r, (0, 0, 255), cv2.FILLED)  # Dibujamos un círculo en el punto medio
        length = math.hypot(x2 - x1, y2 - y1)  # Calculamos la longitud de la línea entre los dos puntos

        return length, frame, [x1, y1, x2, y2, cx, cy]  # Devolvemos la longitud, el frame modificado y las coordenadas del punto medio

# Función principal
def main():
    ptiempo = 0  # Tiempo anterior
    ctiempo = 0  # Tiempo actual

    # Lectura de la cámara web
    cap = cv2.VideoCapture(0)

    # Creamos el objeto para detectar manos
    detector = detectormanos()

    while True:
        ret, frame = cap.read()  # Capturamos un frame de la cámara
        frame = detector.encontrarmanos(frame)  # Detectamos las manos en el frame
        lista, bbox = detector.encontrarposicion(frame)  # Detectamos la posición de las manos
        # Mostramos los FPS
        ctiempo = time.time()  # Obtenemos el tiempo actual
        fps = 1 / (ctiempo - ptiempo)  # Calculamos los FPS
        ptiempo = ctiempo  # Actualizamos el tiempo anterior

        cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)  # Mostramos los FPS en el frame

        cv2.imshow('Manos', frame)  # Mostramos el frame con el nombre 'Manos'
        k = cv2.waitKey(1)  # Capturamos la tecla presionada

        if k == 27:  # Si se presiona la tecla 'Esc' (código 27 en ASCII), salimos del bucle
            break
    cap.release()  # Liberamos la cámara
    cv2.destroyAllWindows()  # Cerramos todas las ventanas
