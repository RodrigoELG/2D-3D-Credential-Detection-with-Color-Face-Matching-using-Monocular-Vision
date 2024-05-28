import cv2
import json
import os
import numpy as np

# Variables globales para valores HSV seleccionados
lower_color = np.array([110, 50, 50])
upper_color = np.array([130, 255, 255])

color_selection = {
    "blue": {
        "lower": np.array([110, 50, 50]),
        "upper": np.array([130, 255, 255]),
        "message": "Estudiante Tec"
    },
    "yellow": {
        "lower": np.array([20, 100, 100]),
        "upper": np.array([50, 255, 255]),
        "message": "Estudiante UDEM"
    },
}

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Añadir punto a la lista global
        puntos_ref[img_name].append((x, y))
        # Dibujar el punto seleccionado en la imagen
        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
        cv2.imshow(img_name, img)

# Función para detectar objetos del color seleccionado
def detect_color(frame):
    # Convertir de BGR a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Threshold la imagen HSV para obtener solo los colores en el rango seleccionado
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # Convertir la máscara de un solo canal a tres canales
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Asegurarse de que la máscara sea un entero sin signo de 8 bits
    mask = np.uint8(mask)
    
    # Bitwise-AND entre la máscara y la imagen original
    res = cv2.bitwise_and(frame, frame, mask=mask[:,:,0])
    
    return res, mask

# Función para seleccionar el color HSV haciendo doble clic
def on_double_click(event, x, y, flags, param):
    global lower_color, upper_color
    if event == cv2.EVENT_LBUTTONDBLCLK:
        frame = param  # Obtener el frame actual
        # Convertir el píxel seleccionado a HSV
        hsv_pixel = cv2.cvtColor(np.uint8([[frame[y, x]]]), cv2.COLOR_BGR2HSV)[0][0]
        print("Color seleccionado/Selected HSV:", hsv_pixel)
        # Actualizar los valores de color
        hue_adjustment = 20  # Ajuste de valor H, puedes ajustar este valor según tu necesidad
        lower_color = np.array([max(hsv_pixel[0] - hue_adjustment, 0), 50, 50])
        upper_color = np.array([min(hsv_pixel[0] + hue_adjustment, 179), 255, 255])

# Función para calcular la distancia basada en el tamaño aparente
def calculate_distance(apparent_size, real_size_width, real_size_height, focal_length_width, focal_length_height):
    distance_width = (real_size_width * focal_length_width) / apparent_size[0]
    distance_height = (real_size_height * focal_length_height) / apparent_size[1]
    distance = (distance_width + distance_height) / 2  # Distancia promedio entre ancho y alto
    return distance

def main():

    # Calibrate the camera and get focal lengths
    #focal_length_width, focal_length_height = click_event(event, x, y, flags, params)

    cap = cv2.VideoCapture(0)

    # Real-world size of the object (in inches)
    real_object_width = 3.3  # Example: 3.3 inches
    real_object_height = 2  # Example: 2 inches

    # Focal length of the camera (in pixels) - You need to calibrate your camera to obtain this value
    #focal_length_width = 539 #Numeros obtenidos de calibracion con ajedrez # Example: 1000 pixels
    #focal_length_height = 549 #Numeros obtenidos de calibracion con ajedrez  # Example: 1000 pixels
    
    # Load face detection classifier
    face_cascade = cv2.CascadeClassifier('C:/Users/rodri/anaconda3/pkgs/libopencv-4.9.0-qt6_py312hd35d245_612/Library/etc/haarcascades/haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        
        # Mostrar la imagen en la ventana para seleccionar el color
        cv2.imshow('Select Color', frame)
        
        # Esperar a que el usuario seleccione el color y ajustar los valores
        cv2.setMouseCallback('Select Color', on_double_click, frame)
        
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    
    # Segunda parte: Deteción de credenciales y caras
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # Detectar objetos del color seleccionado
        color_detected, color_mask = detect_color(frame)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(cv2.cvtColor(color_mask, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrar contornos basados en el área
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Ajustar este umbral según el tamaño esperado de la credencial
                filtered_contours.append(contour)
        
        # Dibujar rectángulos alrededor de los objetos y calcular la distancia
        for contour in filtered_contours:
            # Aproximar el contorno para verificar si es un rectángulo
            epsilon = .1 * cv2.arcLength(contour, True) #epsilon de 0.04
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Verificar si el contorno aproximado tiene cuatro esquinas (rectángulo)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Calcular el tamaño aparente
                apparent_size = (w, h)
                
                # Calcular la distancia
                distance = calculate_distance(apparent_size, real_object_width, real_object_height, focal_length_width, focal_length_height)
                
                # Mostrar distancia
                cv2.putText(frame, f"Distancia: {distance:.2f} pulgadas", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Detectar caras dentro del rectángulo
                roi_frame = frame[y:y + h, x:x + w]
                gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray_roi, 1.3, 5)
                for (fx, fy, fw, fh) in faces:
                    cv2.rectangle(roi_frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
                    
                    # Verificar si se detecta una cara dentro del rectángulo
                if len(faces) > 0:
                    best_match_color = None
                    min_difference = float('inf')
                    
                    # Iterar sobre cada color en color_selection y encontrar el que se asimila mejor a los valores actuales
                    for color, values in color_selection.items():
                        lower_diff = np.abs(lower_color - values["lower"])
                        upper_diff = np.abs(upper_color - values["upper"])
                        total_diff = np.sum(lower_diff) + np.sum(upper_diff)
                    
                        # Actualizar el mejor candidato si encontramos una mejor coincidencia
                        if total_diff < min_difference:
                            min_difference = total_diff
                            best_match_color = color
                
                    # Mostrar el mensaje del mejor color coincidente
                    if best_match_color is not None:
                        cv2.putText(frame, color_selection[best_match_color]["message"], (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Mostrar el marco resultante
        cv2.imshow('Object Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la captura cuando se termine
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

        # Diccionario para almacenar puntos de todas las imágenes
    puntos_ref = {}

    # Dirección de la imagen
    img_path = 'C:/Users/rodri/anaconda3/ImagenesID/ID7.jpg'
    img_name = os.path.basename(img_path)

    # Leer la imagen
    img = cv2.imread(img_path, 1)
    img = cv2.resize(img, (500, 400)) #500,400
    puntos_ref[img_name] = []  # Inicializar lista de puntos para esta imagen

    cv2.imshow(img_name, img)
    cv2.setMouseCallback(img_name, click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Guardar los puntos en un archivo
    with open('puntos_referencia.json', 'w') as fp:
        json.dump(puntos_ref, fp)

    print("Puntos de referencia guardados en 'puntos_referencia.json'")

    img_path = 'C:/Users/rodri/anaconda3/ImagenesID/ID7.jpg'
    img = cv2.imread(img_path, 1)
    #img = cv2.resize(img, (500, 400))

    # Leer los puntos de referencia desde el archivo JSON
    with open('puntos_referencia.json', 'r') as fp:
        puntos_ref = json.load(fp)

    # Definir puntos de imagen y puntos de objeto (esquinas conocidas de la credencial)
    puntos_imagen = []
    puntos_objeto = []

    credencial_ancho = 3.3  # Ancho de la credencial en pulgadas
    credencial_alto = 2.0  # Alto de la credencial en pulgadas

    for img_name, puntos in puntos_ref.items():
        for punto in puntos:
            puntos_imagen.append([punto])
            # Calcular puntos de objeto basados en el tamaño de las credenciales
            x, y = punto
            puntos_objeto.append([(float(x / img.shape[1]) * credencial_ancho, float(y / img.shape[0]) * credencial_alto, 0.0)])

    # Convertir puntos de imagen a NumPy array
    puntos_imagen = np.array(puntos_imagen, dtype=np.float32)

    # Convertir puntos de objeto a estructura requerida por cv2.calibrateCamera (using cv2.Point3d)
    puntos_objeto_nested = []
    for punto in puntos_objeto:
        punto_3d = tuple(punto)  # Convert each point to a tuple
        puntos_objeto_nested.append(punto_3d)
    puntos_objeto = np.array(puntos_objeto_nested, dtype=np.float32)

    # Calcular parámetros intrínsecos y extrínsecos de la cámara
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([puntos_objeto], [puntos_imagen], (img.shape[1], img.shape[0]), None, None)

    # Imprimir resultados
    print("Matriz de la cámara (parámetros intrínsecos):")
    print(mtx)
    print("\nCoeficientes de distorsión:")
    print(dist)
    print("\nVectores de rotación:")
    print(rvecs)
    print("\nVectores de traslación:")
    print(tvecs)

    img = cv2.imread('C:/Users/rodri/anaconda3/ImagenesID/ID7.jpg')
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('calibresult.png', dst)    

    tamano_pixel_mm = 0.00038

    focal_length_width = (mtx[0][0])*(tamano_pixel_mm)
    focal_length_height = (mtx[1][1])*(tamano_pixel_mm)

    print ("\nfocal width:")
    print (focal_length_width)
    print ("\nfocal height:")
    print (focal_length_height)

    main()

# los puntos se seleccionan de origen a credencial de la izquierda, luego derecha, ultima credencial y de arriba a abajo 