import cv2
from object_detector import *
import numpy as np

#definindo o marcador
parameters = cv2.aruco.DetectorParameters_create()
padrao = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)


# construir DO
DO = HomogeneousBgDetector()

# começar video
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    _, img = cap.read()

    # Get Aruco marker
    corners, _, _ = cv2.aruco.detectMarkers(img, padrao, parameters=parameters)
    if corners:

        #destacar o padrao nessa imagem
        int_corners = np.int0(corners)
        cv2.polylines(img, int_corners, True, (0, 0, 255), 5)
        perimetropadrao = cv2.arcLength(corners[0], True)

        # vai determinar a razão entre pixels e milimetros por meio do perimetro do padrao aruco
        RazaoPxmm = perimetropadrao / 200

        contours = DO.detect_objects(img)

        # Draw objects boundaries
        for cnt in contours:
            # fazer uma box em torno do objeto detectado
            rect = cv2.minAreaRect(cnt)
            (x, y), (l, a), _ = rect
            largura = l / RazaoPxmm
            altura = a / RazaoPxmm
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.polylines(img, [box], True, (255, 0, 0), 2)

            # colocar legendas no centro
            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.putText(img, "Largura {} mm".format(round(largura, 1)), (int(x - 100), int(y - 20)),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 5)
            cv2.putText(img, "Largura {} mm".format(round(largura, 1)), (int(x - 100), int(y - 20)),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)
            cv2.putText(img, "Altura {} mm".format(round(altura, 1)), (int(x - 100), int(y + 15)),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 5)
            cv2.putText(img, "Altura {} mm".format(round(altura, 1)), (int(x - 100), int(y + 15)),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)



    cv2.imshow("video", img)
    key = cv2.waitKey(1)

    #aperta esc para sair
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()