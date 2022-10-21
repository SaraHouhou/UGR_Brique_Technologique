
import os
from tabnanny import verbose
import tensorflow as tf
import data_generator, designPlot,predict, modelNetVGG16
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.optimizers import Adam
from livelossplot.inputs.keras import PlotLossesCallback
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
import matplotlib.pyplot as plt
from plot_keras_history import show_history, plot_history
import itertools
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

#=============================#
#Use GPU
# physical_devices=tf.config.experimental.list_physical_devices('GPU')
# print("Num GPUs Available:", len(physical_devices))
# if len(physical_devices)==0 : 
#     raise SystemError('GPU devise not found')

# tf.config.experimental.set_memory_growth(physical_devices[0], True)

#========== test the implemented methods ===========
# GESTURES_Names = [
#      "four",
#     "one",
#     "two_up"
# ]

GESTURES_Names = [
    "five",
    "four",
    "like",
    "mute",
    "ok",
    "one",
    "rock",
    "two",
    "two_inverted"
   # "three", 
   # "call"
]
#Data_DIR_PATH='Data'

Data_DIR_PATH='D:\Data'

TRAIN_DATA_SIZE= 13860
VALID_DATA_SIZE=3960
TEST_DATA_SIZE=1980

#createFolders(Data_DIR_PATH, GESTURES_Names)
valid_path = os.path.join(Data_DIR_PATH, 'valid')
print(valid_path)
train_path = os.path.join(Data_DIR_PATH, 'train')
print(train_path)
test_path = os.path.join(Data_DIR_PATH, 'test')
print(test_path)

# LIST_TRAIN_DIR= os.listdir(train_path)
# print(LIST_TRAIN_DIR)
# LIST_VALID_DIR= os.listdir(valid_path)
# print(LIST_VALID_DIR)        
# LIST_TEST_DIR= os.listdir(test_path)
# print(LIST_TEST_DIR)
# TRAIN_DATA_SIZE= 210
# VALID_DATA_SIZE=60
# TEST_DATA_SIZE=27
#split_data(Data_DIR_PATH, GESTURES_Names, TRAIN_DATA_SIZE, TEST_DATA_SIZE, VALID_DATA_SIZE)

    #test data visualisation
preprocess_input_vgg= tf.keras.applications.vgg16.preprocess_input

train_batches, valid_batches, test_batches= data_generator.dataGenerator_without_Aug(preprocess_input_vgg,  train_path, valid_path, test_path, 224, 10, GESTURES_Names)
data_generator.testDataGeneratorMethod(train_batches, valid_batches, test_batches, 9, TRAIN_DATA_SIZE, VALID_DATA_SIZE, TEST_DATA_SIZE )

imgs_train, labels = next(train_batches)
#designPlot.visualizing_Training_images(imgs_train)
#print(labels)



model= modelNetVGG16.create_model_VGG16(IMAGE_SIZE=224,NBClasses=9, fine_tune=0)
# Print the model summary
model.summary()
#alpha. Also referred to as the learning rate or step size. The proportion that weights are updated (e.g. 0.001). Larger values (e.g. 0.3) results in faster initial learning before the rate is updated. Smaller values (e.g. 1.0E-5) slow learning right down during training
#beta1. The exponential decay rate for the first moment estimates (e.g. 0.9).
#beta2. The exponential decay rate for the second-moment estimates (e.g. 0.999). This value should be set close to 1.0 on problems with a sparse gradient (e.g. NLP and computer vision problems).
#epsilon. Is a very small number to prevent any division by zero in the implementation (e.g. 10E-8).
adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)
#1e-3
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# Checkpoint
checkpoint_path="weights.best.large_Data.hdf5"

    # Learning Rate Reducer
learn_control = ReduceLROnPlateau(
                            monitor='val_accuracy',
                            patience=5,
#                             verbose=1,factor=0.2, 
                            min_lr=1e-7)
   
    # CSVLoger logs epoch, acc, loss, val_acc, val_loss
los_csv=CSVLogger('VGG16Net_large_logs.csv', separator=',', append=False )

early_stop = EarlyStopping(monitor='val_loss',
                           patience=20,
                           #restore_best_weights=True,
                           mode='min')

checkpoint = ModelCheckpoint(checkpoint_path, 
                                monitor='val_accuracy', 
                                verbose=1, 
                                save_best_only=True, 
                                mode='max')
history=model.fit(x=train_batches,                     
          validation_data=valid_batches, 
          epochs=10, 
          verbose=2,
          callbacks=[ checkpoint, early_stop, los_csv])
show_history(history)
plot_history(history, path="VGG16Net_large.png")
plt.close()

with open('weights/historyVGG16_large.txt', 'a+') as f:
    print(history.history, file=f)

print('All Done!')

    #Grad-Cam

#loss, acc = model.evaluate(testX, testY, verbose=2)
#print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

# predict 
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

test_imgs, test_labels = next(test_batches)
#designPlot.visualizing_Training_images(test_imgs)
#print("test labels {}", test_labels)
#test_batches.classes
#print("test classes {}", test_batches.classes)
predictions= model.predict(test_batches)
print("np round predictions {}", np.round(predictions))
y_true=test_batches.classes
y_pred=np.argmax(predictions, axis=1)

cm=confusion_mtx=confusion_matrix(y_true,y_pred)
test_batches.class_indices
cm_plot_labels=GESTURES_Names
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')  
plt.show()
