import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
# Creating a checkpointer 
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import plotResults, confusionMetrics, dataGenerator, dataDetails, network, hardwarDetect
import warnings
warnings.filterwarnings('ignore')


#  Data 
#train_path = 'D:/DataSubSet/train'
#valid_path = 'D:/DataSubSet/valid'
train_path = 'C:/Users/shouhou/Downloads/TinyHGR/Train'
valid_path = 'C:/Users/shouhou/Downloads/TinyHGR/Valid'
   # "call",
    # "five",
    # "four",
    # "one",
    # "three",
    # "two",
    # "v"
GESTURES_Names = [ 
    "first", 
    "l", 
    "ok", 
    "palm",
    "pointer",
    "thumb_down",
    "thumb_up"
]
# detect hardware
hardwarDetect.detectHardware(tf)

# Visualize data
dataDetails.DataCount_TinyHGR(train_path, valid_path)
nb_train_samples = 2951
nb_validation_samples = 2952

dataDetails.VisualizeImages(train_path)



# Data Preprocessing
num_classes = 7 # number of classes of the dataset contains images of 7 different hand gestures
img_rows, img_cols = 224, 224 # define the image size 224x224
batch_size = 16 #define batch size

# data generator function
train_generator, validation_generator = dataGenerator.dataGenerator_without_Aug(train_path, valid_path, img_rows, img_cols, batch_size, GESTURES_Names)
# create the model
model_transfer_full= network.create_model(img_rows, img_cols, nbclasses=7)
# overview a summary
model_transfer_full.summary()
# plot the model
dot_img_file = 'C:/Users/shouhou/testcode/UGR_Brique_Technologique/Final_Model/images/model_2.png'
tf.keras.utils.plot_model(model_transfer_full, to_file=dot_img_file, show_shapes=True)


checkpoint = ModelCheckpoint("C:/Users/shouhou/testcode/UGR_Brique_Technologique/Final_Model/weights/cp_hand_gesture_cnn_vgg16_TinyHGR.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)


earlystop = EarlyStopping(monitor = 'val_loss', 
                          min_delta = 0, 
                          patience = 3,
                          verbose = 1,
                          restore_best_weights = True)

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                              factor = 0.2,
                              patience = 3,
                              verbose = 1,
                              min_delta = 0.0001)

# CSVLoger logs epoch, acc, loss, val_acc, val_loss
los_csv=CSVLogger('C:/Users/shouhou/testcode/UGR_Brique_Technologique/Final_Model/weights/loss_hand_gesture_cnn_vgg16_TinyHGR.csv', separator=',', append=False )

# we put our call backs into a callback list
callbacks = [earlystop, checkpoint, reduce_lr]

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model_transfer_full.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.001),
              metrics=['accuracy'])


epochs = 20

# start training the whole thing, with a very slow learning rate:

history_vgg16 = model_transfer_full.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples //16, # batch_size,
    epochs = epochs,
    callbacks = callbacks,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // 16) 


with open('C:/Users/shouhou/testcode/UGR_Brique_Technologique/Final_Model/weights/historyVGG16_large_TinyHGR.txt', 'a+') as f:
    print(history_vgg16.history, file=f)

print('All Done!')

model_transfer_full.save('C:/Users/shouhou/testcode/UGR_Brique_Technologique/Final_Model/weights/my_model_TinyHGR.h5')


Y_pred_cnn=confusionMetrics.classif_report(model_transfer_full, validation_generator, nb_validation_samples, batch_size)

true_classes = validation_generator.classes
class_labels = validation_generator.class_indices
class_labels = {v: k for k, v in class_labels.items()}
# Get the names of the ten classes
class_names_CNN = list(class_labels.values())

fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8))

figpath ='C:/Users/shouhou/testcode/UGR_Brique_Technologique/Final_Model/images/heatmap_TinyHGR.png'
confusionMetrics.plot_heatmap(true_classes, Y_pred_cnn, class_names_CNN, ax1, title="CNN without data augmentation", figpath=figpath)  

fig.tight_layout()
fig.subplots_adjust(top=1.25)
plt.show()

acc_title= 'Training and validation accuracy -Pretrained VGG16'
loss_title= 'Training and validation loss -Pretrained VGG16'
plotResults.plotHistory(history_vgg16, acc_title, loss_title)