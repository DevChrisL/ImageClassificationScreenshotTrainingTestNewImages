import tensorflow as tf
import numpy as np
import os

#preprocess the image
def preprocess_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(128, 128))#resize to 128x128
    img_array = tf.keras.preprocessing.image.img_to_array(img)#convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)#adds axis=0 to an image with dimensions to 128x128 and 3 channels => (1,128,128,3) makes it into a batch
    img_array = img_array / 255.0#rescale to [0, 1], changes rgb from 0-255 to 0-1, so decimal values
    return img_array#return modified image that it can be feed to model

model = tf.keras.models.load_model("trained_model.h5")#load trained model
predict_folder = "data/imagepredict"#path to test image(s)

if not os.path.isdir(predict_folder) or not os.listdir(predict_folder):#check if the folder exists and also if it contains images
    print(f"No images found in {predict_folder}. Please add images to continue.")#if does not it prints
else:
    for filename in os.listdir(predict_folder):#loop to grab files in specified folder
        img_path = os.path.join(predict_folder, filename)#makes new image path for each file
        if not os.path.isfile(img_path):#reads file from disk
            continue #skip non-file items

        test_image = preprocess_image(img_path)#calls to modify image so that it can be feed to model so that it can predict is real or fake
        prediction = model.predict(test_image)#feed modified image

        result = "real" if prediction[0][0] > prediction[0][1] else "fake"#getting prediction value to determine if real or fake, [real][fake] so if first value is higher than it is real and vice versa
        print(f"The image '{filename}' is predicted to be {result}.\n")