import tensorflow as tf
import os

#load VGG16 model without top classification layer, vgg trained on imagenet, 128x128
modelBase = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

#not replacing/retraining weights, layers in modelBase are frozen to keep VGG16 pre-trained feature maps
for layer in modelBase.layers:
    layer.trainable = False

#build model, top layer
model = tf.keras.models.Sequential([
    modelBase,#VGG16 added to grab features
    tf.keras.layers.Flatten(),#flatten feature maps for them to go into the dense layers
    tf.keras.layers.Dense(64, activation='relu'),#dense layer for more learning using ReLU for more complex layers
    tf.keras.layers.Dense(2, activation='softmax')#output layer for catagorical classification
])

#compile the model
model.compile(
    optimizer='adam',#used for gradient descent
    loss='categorical_crossentropy',#loss function using catagorical classification
    metrics=['accuracy']#save accuracy during training for later
)

#data generator for images
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,#rescales values for easier "learn"ing since RGB values
    rotation_range=40,#randomly rotate
    width_shift_range=0.2,#randomly shift left and right
    height_shift_range=0.2,#ranomly shift up and down
    zoom_range=0.3,#randomly zoom in
    horizontal_flip=True,#randomly flip horizontally
    validation_split=0.2#20% of data for validation
)

#generate training data set from images in "data/"
trainData = datagen.flow_from_directory(
    'data/processed',#folder/path for data
    target_size=(128, 128),#resize images to 128x128
    batch_size=32,#number of images in each batch
    class_mode='categorical',#using categorical for multi-class
    subset='training'#subset used for training
)

#generate validation data set from images in "data/"
valData = datagen.flow_from_directory(
    'data/processed',#folder/path for data
    target_size=(128, 128),#resize images to 128x128
    batch_size=32,#number of images in each batch
    class_mode='categorical',#using categorical for multi-class
    subset='validation'#subset used for validation,
)

#fit/train model, 10 * 32(batch size) steps for validating are 320 per round. 80% * 1026 images real and fake = 820.8. This is due validation split
#Steps per epoch = 820.8/32 = 25.65 steps per epoch, 10 complete passes of the dataset This due to the amount of images for training being 513
model.fit(trainData, validation_data=valData, validation_steps=10, steps_per_epoch=26, epochs=10)

#print out model results
results = model.evaluate(valData)
loss, accuracy = results
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

#save model
model.save("trained_model.h5")
print("Model saved successfully!")