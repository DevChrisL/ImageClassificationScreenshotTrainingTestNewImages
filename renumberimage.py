import os
import cv2

def renameImages(inputFolder, outputFolder, prefix="real"):
    os.makedirs(outputFolder, exist_ok=True)#creates folder and checks if it does not exist

    files = sorted(os.listdir(inputFolder))#grabs and lists all files and sorts them

    for i, filename in enumerate(files):#loop checking filename, enumerate used for 0,1,2,3,4,5,6,etc.
        inputPath = os.path.join(inputFolder, filename)#makes filepath
        
        img = cv2.imread(inputPath)#reads file path and image
        if img is not None:#checks if file is empty
            newFilename = f"{prefix}_{i}.jpg"#rename image as real #
            outputPath = os.path.join(outputFolder, newFilename)#save to new image path
            cv2.imwrite(outputPath, img)#write to disk

#usage
renameImages('data/ogimages', 'data/real')