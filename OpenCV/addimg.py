# Python program for adding
# images using OpenCV

# import OpenCV file
import cv2 as cv

# Read Image1

img1 = cv.imread('cmnd1.jpg')
img2 = cv.imread('bus1.jpg')
assert img1 is not None, "file could not be read, check with os.path.exists()"
assert img2 is not None, "file could not be read, check with os.path.exists()"
dst = cv.addWeighted(img1,0.7,img2,0.3,0)
cv.imshow('dst',dst)
cv.waitKey(0)
cv.destroyAllWindows()
