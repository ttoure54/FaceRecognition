from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os


"""Image classification: process of taking an input (picture) and outputting a class
CNN Process:
- Convolutionnal Layers
- rELu Layers
- Pooling Layers
- A fully connected: connect every neuron in one layer to every neuron in the next layer.

"""

#from tqdm import tqdm


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

#test_img="Alicia_test"
#test_path="Test_image/"
#test_var_img = test_path+test_img
#img_test=cv2.imread("Test_image/Alicia_test_img1.jpg")


cv2.resizeWindow('window',600,800)
#print(test_var_img)


#img_test=image.load_img("/Test_image/Alicia_test_img1")
#cv2.imshow('image',img_test)
#cv2.waitKey(0)
#cv2.destroyAllWindows()



### -----Haar Cascade Face detection ---------####


def face_resizing(person_name):

    img_path="Test_image/"
    img_cam_path= img_path+person_name+"/cam/"

    img_cropped_path=img_path+person_name+"/cropped_face/"

    img_id=1

    while True:

        
    
        img_name = person_name+"_"+str(img_id)+".jpg"
        img_cam = img_cam_path+img_name

        img_cropped=img_cropped_path+img_name
    
        img_cam=cv2.imread(img_cam)

        gray= cv2.cvtColor(img_cam, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5) #Face detection 

        for (x,y,w,h) in faces:
            cv2.rectangle(img_cam,(x,y),(x+w,y+h),(255,0,0),2)
            cropped_face= img_cam[y:y+h,x:x+w]


        cv2.imshow(img_name,cropped_face)
        cv2.imwrite(img_cropped,cropped_face)

        if cv2.waitKey(1)==13 or img_id==9:
            break

        img_id=img_id+1

    cv2.destroyAllWindows()
    return 0
    
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


#train_img=ImageDataGenerator(rescale=1/255)
#valid_img=ImageDataGenerator(rescale=1/255)
#train_data= train.flow_from_directory('train',size=(200,200)
#plt.imshow(img_test)
#plt.show()


#3 layers networks

"""

(X_train, Y_train),(X_test, Y-test)= keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

def generate_cnn_model():


    
        32 filters
        relu actiavation g(z)=max(0,z);
    
    model = tf.keras.Sequential([
    
    tf.keras.layers.Conv2D(32,filter_size=(3,3),activation='relu')
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2)

    tf.keras.layers.Conv2D(64,filter_size=(3,3),activation='relu')
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2)

    tf.keras.layers.Flatten()
    tf.keras.layers.Dense(1024,activation="relu")
    tf.keras.layers.Dense(2,activation='softmax')

    ])

    return model

model= generate_cnn_model()

model.summary()
model.fit(X_train, y_train, n_epoch=12, validation_set=(X_test, y_test), show_metric = True, run_id="FRS" )
"""

"""
cv2.IMREAD_COLOR()
cv2.cvtColor( import_image.cv2.COLOR_BGR2RGB)




cv2.rectangle(frame,(rect[0],rect[1]),(rect[2],rect[3]),(255,0,0))

"""
face_resizing("Alicia")




