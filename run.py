import cv2
import numpy as np
import sys

#Read image
def readImage(path):
    img = cv2.imread(path,1)
    return img

def dodgeV2(x, y):
    return cv2.divide(x, 255 - y, scale=256)

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

def cartoonize():
  #Mess with the values according to your choice
  image = readImage('images/sample.jpg')
  edges = edgeMask(image, 7,7)
  image = colorQuantization(image, 9)
  blurred = cv2.bilateralFilter(image, 7, 200, 200)
  cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)
  #cv2.imshow('image',cartoon)
  #cv2.waitKey(0)
  cv2.imwrite('images/cartoon.jpg',cartoon)

def sketch():
  img = cv2.imread('images/sample.jpg', 1)
  imageGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img_invert = cv2.bitwise_not(imageGray)
  img_smoothing = cv2.GaussianBlur(img_invert, (21, 21),sigmaX=0, sigmaY=0)
  sketchedImage = dodgeV2(imageGray, img_smoothing)

  #cv2.imshow('image',sketchedImage)
  #cv2.waitKey(0)
  cv2.imwrite('images/sketched.jpg',sketchedImage)

n = len(sys.argv)
for i in range (1,n):
  if(sys.argv[i]=='sketch'):
    sketch()
  elif(sys.argv[i]=='cartoonize'):
    cartoonize()