# Main.py

import cv2
import numpy as np
import urllib.request
import os
import requests
import json
from PIL import Image
from io import BytesIO

import DetectChars
import DetectPlates
import PossiblePlate

SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False

def main():

    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN() 

    if blnKNNTrainingSuccessful == False:                               
        print("\nerror: KNN traning was not successful\n")  
        return                                                          
    # end if

    select_image = int(input("Enter the image location:\n"))

    with open('number_plates.json','r') as data:
        number_plates_dict = json.load(data)
		
        print(number_plates_dict[select_image]['content'])
        
        imgOriginalScene = url_to_image(number_plates_dict[select_image]['content'])     

        if imgOriginalScene is None:                            
            print("\nerror: image not read from file \n\n")  
            os.system("pause")                                  
            return                                              
        # end if

        listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)           

        listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)

        cv2.imshow("imgOriginalScene", imgOriginalScene)    
        if len(listOfPossiblePlates) == 0:                  
            print("\nno license plates were detected\n")
        else:                                                     

            listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

           
            licPlate = listOfPossiblePlates[0]

            cv2.imshow("imgPlate", licPlate.imgPlate)          
            cv2.imshow("imgThresh", licPlate.imgThresh)

            if len(licPlate.strChars) == 0:                    
                print("\nno characters were detected\n\n")  
            return                                          
        

            drawRedRectangleAroundPlate(imgOriginalScene, licPlate)             

            print("\nlicense plate read from image = " + licPlate.strChars + "\n")  
            print("----------------------------------------")

            writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)       

            cv2.imshow("imgOriginalScene", imgOriginalScene)                

            cv2.imwrite("imgOriginalScene.png", imgOriginalScene)           

        

    cv2.waitKey(0)					

    return
# end main


def url_to_image(url):
    
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, -1)
    return image
#end function


def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):

    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)            

    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)         
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)
# end function


def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0                             
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0                          
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX                      
    fltFontScale = float(plateHeight) / 30.0                    
    intFontThickness = int(round(fltFontScale * 1.5))           

    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale, intFontThickness)        

            
    ( (intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg ) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)             
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)         

    if intPlateCenterY < (sceneHeight * 0.75):                                                 
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))      
    else:                                                                                      
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))      
    # end if

    textSizeWidth, textSizeHeight = textSize               

    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))           
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))          

            
    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace, fltFontScale, SCALAR_YELLOW, intFontThickness)
# end function

if __name__ == "__main__":
    main()


















