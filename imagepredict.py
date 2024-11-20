import tensorflow as tf
import numpy as np
import os

#preprocess the image
def preprocessImage(imgPath):
    img = tf.keras.preprocessing.image.load_img(imgPath, target_size=(128, 128))#resize to 128x128
    imgArray = tf.keras.preprocessing.image.img_to_array(img)#convert image to numpy array
    imgArray = np.expand_dims(imgArray, axis=0)#adds axis=0 to an image with dimensions to 128x128 and 3 channels => (1,128,128,3) makes it into a batch
    imgArray = imgArray / 255.0#rescale to [0, 1], changes rgb from 0-255 to 0-1, so decimal values
    return imgArray#return modified image that it can be feed to model

model = tf.keras.models.load_model("trained_model.h5")#load trained model
predictFolder = "data/imagepredict"#path to test image(s)

if not os.path.isdir(predictFolder) or not os.listdir(predictFolder):#check if the folder exists and also if it contains images
    print(f"No images found in {predictFolder}. Please add images to continue.")#if does not it prints
else:
    for filename in os.listdir(predictFolder):#loop to grab files in specified folder
        imgPath = os.path.join(predictFolder, filename)#makes new image path for each file
        if not os.path.isfile(imgPath):#reads file from disk
            continue #skip non-file items

        testImage = preprocessImage(imgPath)#calls to modify image so that it can be feed to model so that it can predict is real or fake
        prediction = model.predict(testImage)#feed modified image

        result = "real" if prediction[0][0] > prediction[0][1] else "fake"#getting prediction value to determine if real or fake, [real][fake] so if first value is higher than it is real and vice versa
        print(f"The image '{filename}' is predicted to be {result}.\n")