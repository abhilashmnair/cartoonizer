import cv2
import numpy as np

#Read image in color mode
def readImage(path):
    img = cv2.imread(path,1)
    return img

#Convert image into Grayscale and get borders
def edgeMask(img, lineSize, blurValue):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayBlur = cv2.medianBlur(gray, blurValue)
    edges = cv2.adaptiveThreshold(grayBlur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, lineSize, blurValue)
    return edges

#Cartoonize image using color quantization
def colorQuantization(img, k):
  data = np.float32(img).reshape((-1, 3))
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
  ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
  center = np.uint8(center)
  result = center[label.flatten()]
  result = result.reshape(img.shape)
  return result

#Mess with the values according to your choice
image = readImage('images/sample.jpg')
edges = edgeMask(image, 7,7)
image = colorQuantization(image, 9)
blurred = cv2.bilateralFilter(image, 7, 200, 200)
cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)
#cv2.imshow('image',cartoon)
#cv2.waitKey(0)
cv2.imwrite('images/processed.jpg',cartoon)