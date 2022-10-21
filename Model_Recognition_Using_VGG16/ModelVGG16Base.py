""" Gather hand gesture dataset. Initial dataset is around 800 images, 200 sample pictures for each hand gesture.
Load the dataset, create labels, perform shuffling
Perform train_test_split on the dataset
Create a base model from the pre trained VGG-16.
Customized the model by creating a new model on top of the based model
Perform data augmentation
Train the model
Plot model accuracy and loss """

import os
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input

from PIL import Image
from plot_keras_history import show_history, plot_history

import tensorflow as tf

#Transfer 'jpg' images to an array IMG
def Dataset_loader(DIR, RESIZE, sigmaX=10):
    IMG = []
    read = lambda imname:np.asarray(Image.open(imname)
                                         .convert("RGB"))     
    for IMAGE_NAME in tqdm(os.listdir(DIR)):
        PATH = os.path.join(DIR,IMAGE_NAME)
        _, ftype = os.path.splitext(PATH)
        if ftype == ".jpg":
            img = read(PATH)
            img = cv2.resize(img, (RESIZE,RESIZE))
            IMG.append(np.array(img))
    return IMG

four = np.array(Dataset_loader('Simples/training/four',224))
one = np.array(Dataset_loader('Simples/training/one',224))
two_up = np.array(Dataset_loader('Simples/training/two_up',224))
#play = np.array(Dataset_loader('training/play',224))

# Create labels
four_label = np.zeros(len(four))
one_label = np.ones(len(one))
two_up_label = np.full((len(two_up),), 2)
#play_label = np.full((len(play),), 3)

# Performing shuffling
data = np.concatenate((four, one, two_up), axis = 0)
labels = np.concatenate((four_label, one_label, two_up_label), axis = 0)
data = np.array(data, dtype="float32")
labels = np.array(labels, dtype="int")
data /= 255.0

# shuffle data
arr = np.arange(len(data))
np.random.shuffle(arr)
data = data[arr]
labels = labels[arr]

# Train — Test — Split

le = LabelBinarizer()
labels = le.fit_transform(labels)
counts = labels.sum(axis=0)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# Data augmentation
# initialize an our data augmenter as an "empty" image data generator
aug = ImageDataGenerator()

# check to see if we are applying "on the fly" data augmentation, and
# if so, re-instantiate the object
#if args["augment"] > 0:
print("[INFO] performing 'on the fly' data augmentation")
aug = ImageDataGenerator(
		rotation_range=20,
		zoom_range=0.15,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.15,
		horizontal_flip=True,
        preprocessing_function=preprocess_input,
		fill_mode="nearest")
# Creating the base model
# To make it easier to repeat the process of creating a base model, I made a function which is design to select the base model either DenseNet or Resnet or VGG-16. 
# For the purpose of this project, the VGG-16 model is selected and instantiated with weights trained on ImageNet. 
# The top layer which is the classification layer is excluded and the input shape of the image is set to 224x224x3.

def createBackboneModel(denseNet=False, resnet=False, vgg=False):
    if denseNet: 
       backbone = tf.keras.layers.DenseNet201(weights='imagenet',include_top=False,
                  input_shape=(224,224,3))
    if resnet: 
       backbone = tf.keras.layers.ResNet50(weights='imagenet',include_top=False,
                  input_shape=(224,224,3))
    if vgg: 
       backbone = VGG16(weights='imagenet',include_top=False,
                  input_shape=(224,224,3))

    output = backbone.layers[-1].output
    output = tf.keras.layers.Flatten()(output)
    backboneModel = tf.keras.models.Model(backbone.input, outputs=output)
    for layer in backboneModel.layers:
        layer.trainable = False
    return backboneModel

  #  Creating the model
# This is part where a new model is created on top base model. To easily configure the new model, a function is created with parameter set to some default values. This way I could easily adjust the learning rate, the dense layer and the dropout layer.

def createModel(backbone, lr=1e-4, dense=128, drpout1=0.5, drpout2=0.25):
    EPOCHS = 50
    INIT_LR = 1e-1
    BS = 128
    model = tf.keras.Sequential()
    model.add(backbone)
    model.add(tf.keras.layers.Dropout(drpout1))
    model.add(tf.keras.layers.Dense(dense, activation='relu'))
#     model.add(LeakyReLU(alpha=0.1))
#     model.add(BatchNormalization())
    model.add(tf.keras.layers.Dropout(drpout2))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))
    
  
    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=Adam(learning_rate=lr),
        metrics=['accuracy']
    )
    
    return model

#Loading the pretrained VGG-16 model
backbone = createBackboneModel(vgg=True)
backbone.summary()


# Loading the new model
model = createModel(backbone,lr=1e-4, drpout1=0.3, drpout2=0.2)
model.summary()
Model: "sequential_1"


#Fit and Plot

""" Once the new model is created, the next step is to trained the new model and plot the model accuracy and loss. Using similar approach in create base model and new model, 
a function was made that accepts some parameters like EPOCHS, learning rate, data augmentation, etc. 
 used two callback available in Keras, in conjunction with the fit method. 
 ModelCheckpoint is used to save the model or weights (in a checkpoint file) at some interval, so the model or weights can be loaded later to continue the training from the state saved. 
 The other callback is the ReduceLROnPlateau which will reduce the learning rate once the metrics (i.e. accuracy) is no longer improving.
 """
 # Checkpoint
checkpoint_path="weights.best.hdf5"
def fitandPlot(model, dataAug, trainX, trainY, 
               testX, testY, EPOCHS = 10, INIT_LR = 1e-1, BS = 20, filepath= checkpoint_path):
    
    # Learning Rate Reducer
    learn_control = ReduceLROnPlateau(
                             monitor='val_accuracy',
                             patience=5,
                             verbose=1,factor=0.2, 
                             min_lr=1e-7)
   
    # CSVLoger logs epoch, acc, loss, val_acc, val_loss
    los_csv=CSVLogger('VGGBase_logs.csv', separator=',', append=False )
    checkpoint = ModelCheckpoint(filepath, 
                                 monitor='val_accuracy', 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='max')
    
    history = model.fit(
    dataAug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=trainX.shape[0] / BS,
    epochs=EPOCHS,
    validation_data=(testX, testY),
    callbacks=[learn_control, checkpoint, los_csv])
    show_history(history)
    plot_history(history, path="standard.png")
    plt.close()

    #Grad-Cam

loss, acc = model.evaluate(testX, testY, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

    # To do predictions on the trained model I need to load the best saved model and pre-process the image and pass the image to the model for output.
fitandPlot(model=model, dataAug=aug, trainX=trainX, trainY= trainY, 
               testX=testX, testY=testY)
# Evaluate the model


#Then load the weights from the checkpoint and re-evaluate:


# Loads the weights
model.load_weights(checkpoint_path)
checkpoint_dir = os.path.dirname(checkpoint_path)
# Re-evaluate the model
loss, acc = model.evaluate(testX, testY,  verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
os.listdir(checkpoint_dir)
latest = tf.train.latest_checkpoint(checkpoint_dir)
latest
# Create a new model instance
#model = create_model()

# Load the previously saved weights
#model.load_weights(latest)

# Re-evaluate the model
#loss, acc = model.evaluate(test_images, test_labels, verbose=2)
#print("Restored model, accuracy: {:5.2f}%".format(100 * acc))