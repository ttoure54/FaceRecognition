from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os



name_test_img="Alicia_test_img1"
name_test_path="/Test_image/"
name_test_var_img = name_test_path+name_test_img


cv2.resizeWindow('window',600,800)

print(name_test_var_img)

img_test=cv2.imread("Test_image/Alicia_test_img1.png", cv2.IMREAD_COLOR)
#img_test=image.load_img("/Test_image/Alicia_test_img1")
cv2.imshow('image',img_test)
#plt.imshow(img_test)
#plt.show()



"""
cv2.IMREAD_COLOR()
cv2.cvtColor( import_image.cv2.COLOR_BGR2RGB)


cv2.rectangle(frame,(rect[0],rect[1]),(rect[2],rect[3]),(255,0,0))

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')






cv2.imshow(‘IMport Image, import_image)
cv2.waitKey(0)




"""




