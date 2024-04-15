import cv2
import numpy as np
from IPython.display import Image


def shi_tomasi_detect_corner(img_path, maxCornerNB, qualityLevel, minDistance=0.6):
    img = cv2.imread(img_path)

    # convert to gray image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # shi tomasi detect corners
    corners = cv2.goodFeaturesToTrack(gray, maxCornerNB, qualityLevel, minDistance)
    corners = np.int0(corners)

    for i in corners:
        # take (x, y) of corners
        x, y = i.ravel()

        # draw circle
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

    cv2.imwrite('C:/Users/Admin/PythonLession/pic/corner.png', img)
    return 'C:/Users/Admin/PythonLession/pic/corner.png'


img_path = shi_tomasi_detect_corner('house.jpg', 600, 0.05)
Image(img_path)
