from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop


import PIL
import PIL.Image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os


"""
2021 - ISTY - SEE5 - BORG Arthur & TOURE Talla 
FaceRecognition library 
Using examples available in the script main.py 

Description : This library containts the Class "FaceRecognition" whiich includes methods to perform face detection and face recongnition using 
Convolutionnal Neural Network. 

"""



class FaceRecognition : 

    def __init__(self):
        #Contructor 
        #path initialisation 
        self.dirname = os.path.dirname(__file__)
        self.root = os.path.dirname(self.dirname)
        self.path_user = self.root+'/../Data/user/'
        self.path_facedetected = self.path_user+"face_detected/"
        self.user_dic ={}
        self.test_path = self.path_user+"test_images/"


        #tensor and model initailisation 
        
        self.class_names=[]
        self.model = tf.keras.Sequential()
        self.train_data = self.model
        self.validation_data = self.model


        self.face_haarcascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_haarcascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


        #for user in os.listdir(self.user_path):
        #    self.user_dic[user]=user
        
        #print(self.user_dic)

        try: 
            os.mkdir(self.path_user)
        except:
            print ("LOG: user folder already initialized")

        try: 
            os.mkdir(self.path_facedetected)
        except:
            print ("LOG: user folder already initialized")

        try: 
            os.mkdir(self.test_path)
        except:
            print ("LOG: user folder already initialized")


    
    def user_init(self, raw_imgfolder, username):

        img_id =0
        
        path_facedetected = self.path_facedetected+username+'/'
        path_username = self.path_user+username


        try: 
            os.mkdir(path_username)
            print ("LOG:",path_username,"created")
        except:
            print ("LOG: user path already initialized")

        try:
            os.mkdir(path_facedetected)
            print ("LOG:",path_facedetected,"created")
        except:
            print ("LOG: username",username,"already initialized")
        

        try:    
            for img in os.listdir(raw_imgfolder):
                name, ext =os.path.splitext(img)
                if ext == '.jpeg' or ext =='.jpg':
                    img_dir = raw_imgfolder+img
                    print("LOG: loading image", ext, " : ",img_dir)
                    self.face_detection(img_dir,path_facedetected,username+'_'+str(img_id))
                    img_id = img_id +1
        except:
            print ("ERROR USER INIT: can't find the image check the path written without backslah at the end")
            exit

        
        print("LOG: resized done")


    def face_detection(self,raw_image_path,path_imgsaved,img_name):


        ### -----Haar Cascade Face detection ---------####
        face_haarcascade = self.face_haarcascade
        eye_haarcascade = self.eye_haarcascade

        cv2.resizeWindow('window',600,800)
        cropped_face=np.float32([0])


        try :
            raw_image_path=cv2.imread(raw_image_path)
        except:
            print("ERROR FACE DETECTION: ", raw_image_path)
            print("ERROR FACE DETECTION: can't read the raw image")
            exit

        gray= cv2.cvtColor(raw_image_path, cv2.COLOR_BGR2GRAY)
        faces = face_haarcascade.detectMultiScale(gray, 1.3, 5) #Face detection 

        for (x,y,w,h) in faces:
            cv2.rectangle(raw_image_path,(x,y),(x+w,y+h),(255,0,0),2)
            cropped_face= raw_image_path[y:y+h,x:x+w]

        cv2.resize(cropped_face,(200,200))
        cv2.imshow(img_name,cropped_face)
        cv2.imwrite(path_imgsaved+img_name+'.jpg',cropped_face)
        print("LOG: img saved in", path_imgsaved+img_name+'.jpg')

        if cv2.waitKey(1)==13:
            exit

        cv2.destroyAllWindows()


        return 0
        

    def CNNmodel_train(self):
        #####Model compile
        self.model.compile(optimizer='adam',#Adam/Xavier algorithms help in Optimization
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

        history = self.model.fit(self.train_data,validation_data=self.validation_data,epochs=20)
    

        #test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

        self.model.summary()


    def processing_dataset(self):

        #Prepocess data 

        BATCH_SIZE=32
        IMG_SIZE=(200,200)

        train_path_img = self.path_facedetected

        self.train_data = tf.keras.preprocessing.image_dataset_from_directory(train_path_img,
                                             shuffle=True,
                                             validation_split=0.1,                                     
                                             subset="training",
                                             seed=123,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE)

        self.validation_data = tf.keras.preprocessing.image_dataset_from_directory(train_path_img,
                                             validation_split=0.1,
                                             shuffle=True,
                                             subset="validation",
                                             seed=123,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE)


        self.class_names=self.train_data.class_names #Storing the classes gather in the attributes of the classes to reuse
        #DEBUG
        print(self.train_data)
        #print(train_data[2])
        print(self.validation_data)
        print(self.class_names)




    def CNNmodel_create(self):

        self.model = tf.keras.Sequential([

        #Image suosampling 
        #Conv + subsmapling 


        #normalization of the matrix
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
    

        #tf.keras.layers.Conv2D(32,(3,3),activation='relu'),#32: number of nodes, (3,3), filter dimension
        tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

        #tf.keras.layers.Conv2D(62,(3,3),activation='relu'),#64: number of nodes, (3,3), filter dimension
        #tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

        tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

        #tf.keras.layers.Conv2D(64,(3,3),activation='relu'),#64: number of nodes, (3,3), filter dimension
        tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

        tf.keras.layers.Conv2D(128,(3,3),activation='relu'),#128: number of nodes, (3,3), filter dimension
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    
        #tf.keras.layers.Conv2D(512,(3,3),activation='relu'),
        #tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

        #tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
        #tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

        #Connected layer
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512,activation="relu"), #creating a 
        tf.keras.layers.Dense(len(self.class_names),activation='softmax'), #probaility to get one of the classes 

        ])

        
    
    
    def CNNmodel_userpredict(self, imgname_test):

        img=self.test_path+imgname_test
        cropped_face=np.float32([0])

        print("Test Image:",img)

        img_res = cv2.imread(img)

        #### Perform face detection on the input image
        gray= cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
        faces = self.face_haarcascade.detectMultiScale(gray, 1.3, 5) #Face detection 

        for (x,y,w,h) in faces:
            cv2.rectangle(img_res,(x,y),(x+w,y+h),(255,0,0),2)
            cropped_face= img_res[y:y+h,x:x+w]
        
        img_test = cv2.resize(cropped_face,(200,200))
        
        #img_test = cv2.imread(img,cv2.COLOR_BGR2GRAY)
        #img_test = cv2.resize(img_test,(200,200))

        #### Testing and reshaping image to tensor 
        img_test=tf.convert_to_tensor(img_test) #Converting img into tensor
        img_test=tf.reshape(img_test, (1,200,200,3)) #Reshaping img to 200*200
        #test_img= np.expand_dims(img_test,0)
        print (img_test.shape)
        #test_img=tf.io.decode_image(test_path)

        ####Image prediction 
        result=self.model.predict(img_test,verbose=1)
        res_classes = self.model.predict_classes(img_test, verbose=1)
        res_name = self.class_names[np.argmax(result[0])]
        print("Result " ,res_classes," : ",res_name)#get the name o
        #res_name = self.class_names[res_classes]
        print('Prediction is: ',result)


        ####Text name position 
        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(res_name, font, 0.5, 1)[0]
        
        Xtext = int((img_res.shape[1] - textsize[0]) / 2) #Center text on X 
        Ytext = int(img_res.shape[0] - 20)
        
        ####Showning the input immage with 
        img_res = cv2.putText(img_res, res_name, (Xtext,Ytext), font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow('img', img_res)
        k = cv2.waitKey(0)
        if k == 27:
           cv2.destroyAllWindows()
        

    def CNNmodel_save(self):
        try:
            tf.saved_model.save(self.model,"../Data/FaceReco_model/")
            print("LOG: model saved in saved_model/")
        except:
            print("ERROR: model saving in saved_model/ failed")
            

    def CNNmodel_load(self):
        self.model = tf.keras.models.load_model("../Data/FaceReco_model")
        print("LOG: model loaded in saved_model/")

        