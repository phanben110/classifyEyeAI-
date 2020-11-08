import cv2
import numpy as np
import BEN_classifyStatusEye as eye

JSON_FILE = "jsonEyeModelV1.json"
WEIGHTS_FILE = "eyeModelV1.h5"


face     = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
leftEye  = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
#leftEye  = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
rightEye = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
video = 'driver.mp4'
#video = 'ben.mp4'
#video = 'tramy.mp4'

cap = cv2.VideoCapture(video)

modelAI = eye.runModel( JSON_FILE , WEIGHTS_FILE ) 
model  = modelAI.wakeUpModel() 
count = 0 

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #faces = face_cascade.detectMultiScale(gray, 1.3,5)
    faces = face.detectMultiScale(gray, 1.3, 5)
    for(fx,fy,fw,fh) in faces:
        cv2.rectangle(img,(fx,fy), (fx+fw , fy+fh), (255,0,0) ,2)
        roi_gray1 = gray[fy:fy+fh, fx:fx+fw]
        roi_color1= img[fy:fy+fh, fx:fx+fw]
        leftEyes = leftEye.detectMultiScale(roi_gray1)
        rightEyes = rightEye.detectMultiScale(roi_gray1)
        i = True  
        j = True
        status = "open"
       
        for (lx,ly,lw,lh) in leftEyes:
            if ( i == True ) : 
                
                if ( lx >= fw/4 )  and ( lx <= 3*fw/4 ) and ( ly <= fh/3) and (lw*lh/(fw*fh)<=0.09):

                    cv2.rectangle(roi_color1, (lx, ly), (lx + lw, ly + lh), (0, 255, 0), 2)
                    s1 = lw*lh /( fw*fh ) 
                    i= False 
                    roi_gray2  = roi_gray1[ly:ly + lh, lx:lx + lw ]
                    roi_color2 = roi_color1[ly:ly + lh, lx:lx + lw ]
                    #imageEye = cv2.resize ( roi_gray2 , (50,50)) 
                    name = "/home/ben/ai/image/eyeAI/dataV3/eyeLeftClose/"+"leftEyes_" + str (status) + str (count) + ".jpg" 
                    cv2.imwrite(name , roi_gray2 ) 
                    count+=1 
                    result1 = modelAI.getResult ( roi_gray2 , model ) 
                    cv2.putText(img,str (result1), (fx + lx , fy + ly - 10 ), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 255, 0), 2)
                    #roi_gray2  = cv2.resize( roi_gray2 , (24,24)) 
                    print ( result1 )
        for (rx,ry,rw,rh) in rightEyes :
            if ( j == True ) :
                if ( rx >= 0 ) and ( rx <=fw/4 )  and ( ry <= fh/3) and (rw*rh/(fw*fh)<=0.09):
                    s2 = rw*rh /(  fw*fh )
                    cv2.rectangle(roi_color1, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 2)
                    j = False 
                    roi_gray2 = roi_gray1[ly:ly + lh, lx:lx + lw ]
                    roi_color2 = roi_color1[ly:ly + lh, lx:lx + lw ]
                    #roi_gray2 = cv2.resize(roi_gray2  ,( 24,24)) 
                    result2 = modelAI.getResult ( roi_gray2 , model ) 

                    cv2.putText(img,str(result2) , (fx + rx   , fy + ry  -10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 0, 255), 2)
                    print ( result2 )


    cv2.imshow('img', img )
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
