
from FaceRecognition import FaceRecognition

face_recognition= FaceRecognition.FaceRecognition()

raw_imgpath = '/Users/tallatoure/Documents/SEE4_Controleur_d_accès_prj/Data/Raw_img/'
DEBUGimg = '/Users/tallatoure/Documents/SEE4_Controleur_d_accès_prj/Data/user/face_detected/Will_Smith/Will_Smith_1.jpg'

#raw_imgpath= '/Users/tallatoure/Documents/SEE4_Controleur_d_accès_prj/Test_image/test_Talla/'
rawimgWill= raw_imgpath+'Will_Smith/'
rawimgScar= raw_imgpath + "Scartlett_Johansson/"
rawimgLeo= raw_imgpath+'Leonardo_DiCaprio/'
rawimgBrad= raw_imgpath+'Brad_Pitt/'


test_img1 = "Test_1.jpeg"
test_img2 = "Test_2.jpeg"


#face_recognition.DEBUG_printimg(DEBUGimg)

#face_recognition.user_init(raw_imgpath, "Talla")
face_recognition.user_init(rawimgWill, "Will_Smith")
#face_recognition.user_init(rawimgBrad, "Brad_Pitt")
#face_recognition.user_init(rawimgLeo, "Leonardo_DiCaprio")
#face_recognition.user_init(rawimgScar, "Scartlett_Johansson")

"""
face_recognition.processing_dataset()
#face_recognition.CNNmodel_load()

face_recognition.CNNmodel_create()

face_recognition.CNNmodel_train()

face_recognition.CNNmodel_save()

face_recognition.CNNmodel_userpredict("Scar10.jpg")
#face_recognition.CNNmodel_userpredict(test_img1)
"""
