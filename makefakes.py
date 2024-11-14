import cv2
import numpy as np
import os
import random
import re

def modifications(image):
    #random noise
    if random.random() < 0.5:#greater than 50%
        noise = np.random.normal(0, 10, image.shape).astype(np.uint8)#generates noise by random using numpy, dimensions from image, and initalized values
        image = cv2.add(image, noise)#adds the noise to image

    #random brightness and contrast
    if random.random() < 0.5:
        brightness = random.randint(-50, 50)#generates random brightness value
        contrast = random.uniform(0.5, 1.5)#generates random contrast value
        image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)#adds new brightness and contrast values to image

    #randomly Gaussian blur
    if random.random() < 0.5:
        blurValue = random.choice([3, 5, 7, 9])#generates random blur value
        image = cv2.GaussianBlur(image, (blurValue, blurValue), 0)#adds blur to image

    return image#returns modified image

def createFakeImages(inputFolder, outputFolder, numCopies=1):
    os.makedirs(outputFolder, exist_ok=True)#creates folder and checks if it does not exist

    for filename in os.listdir(inputFolder):#loop to grab files in specified folder
        imgPath = os.path.join(inputFolder, filename)#makes new image path for each file
        img = cv2.imread(imgPath)#reads file from disk

        if img is not None:#checks if file is empty
            match = re.search(r'\d+', filename)#using regEx reads raw string any digit one or more in filename
            if match:
                baseNumber = match.group(0)#use the extracted number
                
                for i in range(numCopies):#loop to repeat process per image
                    alteredImg = modifications(img.copy())#copies image and applies random noise, brightness, contrast, and blur
                    alteredFilename = f"fake_{baseNumber}.jpg"#rename image as fake #
                    savePath = os.path.join(outputFolder, alteredFilename)#save to new image path
                    cv2.imwrite(savePath, alteredImg)#write to disk

#creates 1 fake for each real image
createFakeImages('data/real', 'data/fake', numCopies=1)
