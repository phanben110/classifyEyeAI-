from tensorflow.keras.models import load_model , model_from_json
import numpy as np
import json
import cv2

img1 = '/home/ben/ai/image/face_rec-master/dataset/train/open/Zulfiqar_Ahmed_0001_L.jpg'
img2 = '/home/ben/ai/image/face_rec-master/dataset/train/closed/closed_eye_1886.jpg_face_1_L.jpg'

JSON_FILE = "jsonEyeModelV1.json"
WEIJSON_FILE = "jsonEyeModelV1.json"
WEIGHTS_FILE = "eyeModelV1.h5"
GHTS_FILE = "eyeModelV1.h5"

class runModel: 
    def __init__( self  , jsonFile , weightsFile, debug = True ) : 
        self.jsonFile = jsonFile 
        self.weightsFile = weightsFile 
        self.debug = debug 
        self.model = None 
    



    def labels(self , predict):
        labels =  [
                    "closed",
                    "open" , 
                    "sorry i can't predic this sound, which may be is noise" 
                ]
        print (f"this state eye is : {labels[predict]}")
        return labels[predict]
    
    def loadModel(self , JSON_FILE , WEIGHTS_FILE , model ):
        with open(JSON_FILE , 'r') as f:
            modelJson = json.load(f)
    
        self.model = model_from_json(modelJson)
        self.model.load_weights(WEIGHTS_FILE)
        print(self.model.summary())
        return self.model
    
    
    def predictImage( self , img , model ) :
     
        #img = Image.fromarray(img  ).convert('L')
        #IMG = image.load_img(img , target_size= (24,24) )
        
        #x = image.img_to_array(IMG)
        #x = np.expand_dims( x , axis = 0 )
        #images = np.vstack([x])
        #prediction = model.predict_classes ( images , batch_size = 10 )
        #IMG = cv2.imread ( img , 0 )
        IMG = cv2.resize ( img , (50, 50))
        
        IMG = np.reshape(IMG , [1,50,50, 1])
        prediction = model.predict(IMG)
      
        value = np.argmax( prediction )
       
        myyeu =  prediction[0][value] 
        if ( myyeu >= 0.7 ) :
            return value
        
        return 2 
    
    
    
    def wakeUpModel(self ):
    
        # B1 : load data
    
        Model = self.loadModel(self.jsonFile , self.weightsFile , self.model )
        return Model 
    def getResult( self , img , Model ) :
        
        result = self.predictImage( img , Model)
        ben = self.labels ( result ) 
        return ben 
    

    
if __name__ == "__main__":
    image1 = runModel(  JSON_FILE  ,WEIGHTS_FILE ) 
    Model = image1.wakeUpModel()  
    image1.getResult ( img1 , Model ) 
    image1.getResult ( img2 , Model )

    print("done")
    
