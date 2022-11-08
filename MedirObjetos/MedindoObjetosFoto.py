import cv2
from object_detector import *
import numpy as np

# definindo o padrão  de medição aruco (5cm x 5cm)
parameters = cv2.aruco.DetectorParameters_create()
padrao = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)


# construindo o detector de objetos (DO)
DO = HomogeneousBgDetector()

# selecionar uma imagem para ser lida
img = cv2.imread("carregadorbranco.jpg")


#destacar o padrao nessa imagem
corners, _, _ = cv2.aruco.detectMarkers(img, padrao, parameters=parameters)
int_corners = np.int0(corners)
cv2.polylines(img, int_corners, True, (0, 0, 255), 5)
perimetropadrao = cv2.arcLength(corners[0], True)

# vai determinar a razão entre pixels e milimetros por meio do perimetro do padrao aruco
RazaoPxmm = perimetropadrao / 200


contours = DO.detect_objects(img)

for cnt in contours:
    # fazer uma box em torno do objeto detectado
    rect = cv2.minAreaRect(cnt)
    (x, y), (l, a), _ = rect
    largura = l / RazaoPxmm
    altura = a / RazaoPxmm
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.polylines(img, [box], True, (255, 0, 0), 2)

    #colocar legendas no centro
    cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
    cv2.putText(img, "Largura {} mm".format(round(largura, 1)), (int(x - 100), int(y - 20)),
                cv2.FONT_HERSHEY_PLAIN, 1.6, (0, 0, 0), 5)
    cv2.putText(img, "Largura {} mm".format(round(largura, 1)), (int(x - 100), int(y - 20)),
                cv2.FONT_HERSHEY_PLAIN, 1.6, (0, 255, 255), 2)
    cv2.putText(img, "Altura {} mm".format(round(altura, 1)), (int(x - 100), int(y + 15)),
                cv2.FONT_HERSHEY_PLAIN, 1.6, (0, 0, 0), 5)
    cv2.putText(img, "Altura {} mm".format(round(altura, 1)), (int(x - 100), int(y + 15)),
                cv2.FONT_HERSHEY_PLAIN, 1.6, (0, 255, 255), 2)

imagem = cv2.resize(img, (960, 540))
cv2.imshow("Resultado", imagem)
cv2.waitKey(0)