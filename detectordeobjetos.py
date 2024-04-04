import cv2
import numpy as np
import imutils
import time
import os

# Cargamos el modelo YOLOv4 pre-entrenado
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Inicializamos el video
cap = cv2.VideoCapture(0)

# Crear directorio para guardar las imágenes
if not os.path.exists('detected_images'):
    os.makedirs('detected_images')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensionar el frame y detectar objetos utilizando YOLOv4
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Inicializar lista de confianzas, clases y cajas del objeto
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Detección de movimiento: guardar imagen si se detecta un objeto
                (h, w) = frame.shape[:2]
                box = detection[0:4] * np.array([w, h, w, h])
                (center_x, center_y, width, height) = box.astype("int")
                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Dibujar rectángulos y guardar imágenes si se detectan objetos
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"detected_images/{timestamp}.jpg", frame)

    # Mostrar frame
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

