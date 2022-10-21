
# Dataset from https://laurencemoroney.com/datasets.html

# Images have all been generated using CGI techniques as an experiment in determining if a CGI-based dataset can be used for classification against real images.
import LoadData, predict
import designPlot
import modelNetVGG16
import predict
import tensorflow as tf
import os
from math import ceil
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.optimizers import Adam
from livelossplot.inputs.keras import PlotLossesCallback

# load Data

base_dir="Simples"

train_dir, test_dir, validation_dir= LoadData.load(base_dir)

LoadData.Show_NB_Data(train_dir, test_dir, validation_dir)
# Data preprocessing
BATCH_SIZE = 10
IMAGE_SIZE=224
train_generator, test_generator=LoadData.train_val_generators(train_dir, test_dir, IMAGE_SIZE, BATCH_SIZE, 'categorical')


# configure the network

NBClasses=3
network = modelNetVGG16.create_model_VGG16(IMAGE_SIZE,NBClasses, fine_tune=0)

# Print the model summary
network.summary()

# Define a Callback class that stops training once accuracy reaches 99.9%
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.96 & logs.get('loss')>0.91):
      print("\nReached 96% accuracy so cancelling training!")
      self.model.stop_training = True


# Set the training parameters

def loss_function_1(y_true, y_pred):
    """ Probabilistic output loss """
    a = tf.clip_by_value(y_pred, 1e-20, 1)
    b = tf.clip_by_value(tf.subtract(1.0, y_pred), 1e-20, 1)
    cross_entropy = - tf.multiply(y_true, tf.math.log(a)) - tf.multiply(tf.subtract(1.0, y_true), tf.math.log(b))
    cross_entropy = tf.reduce_mean(cross_entropy, 0)
    loss = tf.reduce_mean(cross_entropy)
    return loss



#alpha. Also referred to as the learning rate or step size. The proportion that weights are updated (e.g. 0.001). Larger values (e.g. 0.3) results in faster initial learning before the rate is updated. Smaller values (e.g. 1.0E-5) slow learning right down during training
#beta1. The exponential decay rate for the first moment estimates (e.g. 0.9).
#beta2. The exponential decay rate for the second-moment estimates (e.g. 0.999). This value should be set close to 1.0 on problems with a sparse gradient (e.g. NLP and computer vision problems).
#epsilon. Is a very small number to prevent any division by zero in the implementation (e.g. 10E-8).
adam = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-10, decay=0.0)
loss_function = {"prob_output": loss_function_1}
network.compile(optimizer=adam, loss=loss_function, metrics=['accuracy'])
#optim_1 = Adam(learning_rate=0.00001)
#optimizer = RMSprop(learning_rate=0.0001)
# Compile
#network.compile(loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True,reduction='auto'), optimizer=adam, metrics=['accuracy'])

# Train the model
epochs = 100
train_set_size = len(os.listdir(train_dir))
test_set_size = len(os.listdir(test_dir)) 
training_steps_per_epoch = ceil(train_set_size / BATCH_SIZE)
validation_steps_per_epoch = ceil(test_set_size / BATCH_SIZE)

# Set the training parameters
checkpoints = ModelCheckpoint('tl_model_v1.weights.best.hdf5', 
                             # monitor='val_acc', 
                              verbose=1, 
                              save_best_only=True)#, 
                             # mode='max')

# Finds the minimun in the validation loss and it goes three epochs after them to confirm that it is exactly the lowest point
#early_stop = EarlyStopping(monitor='val_loss', partience=3, verbose=1)
early_stop = EarlyStopping(monitor='val_loss',
                           patience=3,
                           restore_best_weights=True,
                           mode='min')

# CSVLoger logs epoch, acc, loss, val_acc, val_loss
los_csv=CSVLogger('my_logs.csv', separator=',', append=False )

# loss plot 
plot_loss_1 = PlotLossesCallback()

calback_list= [checkpoints, early_stop, los_csv, plot_loss_1]
history = network.fit(train_generator, 
                   # steps_per_epoch=training_steps_per_epoch,  # total number of steps (batches of samples)
                    epochs=epochs,  # number of epochs to train the model
                    validation_data=test_generator, 
                    #validation_steps=validation_steps_per_epoch,
                    verbose=1, # verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch
                    #use_multiprocessing=False,        # whether to use process-based threading
                   # max_queue_size=512,                # maximum size for the generator queue
                   # shuffle=True,                     # whether to shuffle the order of the batches at the beginning of each epoch
                    #class_weight=classWeights,                # optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function

                    callbacks=calback_list) # keras.callbacks.Callback instances to apply during training

with open('weights/history.txt', 'a+') as f:
    print(history.history, file=f)

print('All Done!')

# design results

designPlot.diagrams(history=history)

#predict.generate_prediction(path='tl_model_v1.weights.best.hdf5', model_history=history, train_generator= train_generator, test_Generator=test_generator)
# Model Prediction
#predict(validation_dir, network, 150 )





