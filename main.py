
"""
2021 - ISTY - SEE5 - BORG Arthur & TOURE Talla 
Exmaples of code using the FaceRecognition library

main.py
Description : This library containts the Class "FaceRecognition" whiich includes methods to perform face detection and face recongnition using 
Convolutionnal Neural Network. 

"""

from FaceRecognition import FaceRecognition

face_recognition= FaceRecognition.FaceRecognition()

raw_imgpath = '/Users/tallatoure/Documents/SEE4_Controleur_d_accès_prj/Data/Raw_img/'
DEBUGimg = '/Users/tallatoure/Documents/SEE4_Controleur_d_accès_prj/Data/user/face_detected/Will_Smith/Will_Smith_1.jpg'


rawimgWill= raw_imgpath+'Will_Smith/'
rawimgScar= raw_imgpath + "Scartlett_Johansson/"
rawimgTalla= raw_imgpath+'Talla_Toure/'
rawimgArthur= raw_imgpath + "Arthur_Borg/"


#face_recognition.DEBUG_printimg(DEBUGimg)

#face_recognition.user_init(rawimgTalla, "Talla_Toure")
#face_recognition.user_init(rawimgArthur, "Arthur_Borg")


face_recognition.processing_dataset()
face_recognition.CNNmodel_load()

#face_recognition.CNNmodel_create()

#face_recognition.CNNmodel_train()

#face_recognition.CNNmodel_save()

#face_recognition.CNNmodel_userpredict("Scar4.jpg")
#face_recognition.CNNmodel_userpredict("Will2.jpg")
#face_recognition.CNNmodel_userpredict("Arthur13.jpg")
face_recognition.CNNmodel_userpredict("Talla5.jpg")

