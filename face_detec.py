from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

test_img="Alicia_test_img1"
test_path="/Test_image/"
test_var_img = test_path+test_img


cv2.resizeWindow('window',600,800)
print(test_var_img)

img_test=cv2.imread("Test_image/Alicia_test_img1.jpg")
#img_test=image.load_img("/Test_image/Alicia_test_img1")
#cv2.imshow('image',img_test)
#cv2.waitKey(0)
#cv2.destroyAllWindows()



### -----Haar Cascade Face detection ---------####

gray= cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5) #Face detection 

for (x,y,w,h) in faces:
    cv2.rectangle(img_test,(x,y),(x+w,y+h),(255,0,0),2)
    cropped_face= img_test[y:y+h,x:x+w]

cv2.imshow('img_test',cropped_face)
cv2.waitKey(0)
cv2.destroyAllWindows()


#train_img=ImageDataGenerator(rescale=1/255)
#valid_img=ImageDataGenerator(rescale=1/255)
#train_data= train.flow_from_directory('train',size=(200,200)
#plt.imshow(img_test)
#plt.show()


"""
cv2.IMREAD_COLOR()
cv2.cvtColor( import_image.cv2.COLOR_BGR2RGB)


cv2.rectangle(frame,(rect[0],rect[1]),(rect[2],rect[3]),(255,0,0))

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')



"""




