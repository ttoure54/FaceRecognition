
"""
TOURE Talla - tallatoure.pro@gmail.com
Exmaples of code using the FaceRecognition library

main.py
Description : This library containts the Class "FaceRecognition" whiich includes methods to perform face detection and face recongnition using 
Convolutionnal Neural Network. 

"""

from FaceRecognition import FaceRecognition



if __name__ == '__main__':

face_recognition= FaceRecognition.FaceRecognition()

raw_imgpath = '/Users/tallatoure/Documents/SEE5_FaceReco/Data/Raw_img/'



rawimgTalla= raw_imgpath+'Talla_Toure/'



#face_recognition.DEBUG_printimg(DEBUGimg)

face_recognition.user_init(rawimgTalla, "Talla_Toure")



face_recognition.processing_dataset()
#face_recognition.CNNmodel_load()

face_recognition.CNNmodel_create()

face_recognition.CNNmodel_train()

face_recognition.CNNmodel_save()

#face_recognition.CNNmodel_userpredict("Talla5.jpg")

