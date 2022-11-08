import cv2


class HomogeneousBgDetector():
    def __init__(self):
        pass

    def detect_objects(self, frame):
        # Destaca limites nas regiões com grande diferença de valor entre os pixels (a imagem deve ser convertida para tons de preto e branco)
        PretoeBranco = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        filtro = cv2.adaptiveThreshold(PretoeBranco, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)

        # encontrar contornos
        contours, _ = cv2.findContours(filtro, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        objects_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 2000:
                objects_contours.append(cnt)

        return objects_contours
