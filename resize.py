import cv2
import os

#uses openCV library to intake images and resize them for use in ML in this case.
def preprocessImages(inputFolder, outputFolder, size=(128, 128)):#sets size to 128x128 with input and output sources
    os.makedirs(outputFolder, exist_ok=True)#creates folder
    for filename in os.listdir(inputFolder):#loop through files from input
        imgPath = os.path.join(inputFolder, filename)#path for file
        img = cv2.imread(imgPath)#reads the image
        if img is not None:#if statement checking if valid    
            img = cv2.resize(img, size)#resizes to 128x128
            savePath = os.path.join(outputFolder, filename)#saves file to disk given name and output
            cv2.imwrite(savePath, img)

# Preprocess real and fake images
preprocessImages('data/real', 'data/processed/realpreprocessed')#using these set folder names and paths
preprocessImages('data/fake', 'data/processed/fakepreprocessed')