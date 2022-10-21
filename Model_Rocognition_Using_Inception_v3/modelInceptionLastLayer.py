""" Using inception and Using dropout!
dropout Another useful tool to explore at this point is the Dropout. 

The idea behind it is to remove a random number of neurons in your neural network. 
This works very well for two reasons: The first is that neighboring neurons often end up with similar weights, which can lead to overfitting, so dropping some out at random can remove this. 
The second is that often a neuron can over-weigh the input from a neuron in the previous layer, and can over specialize as a result. 
Thus, dropping out can break the neural network out of this potential bad habit!  """

# For more on how to freeze/lock layers, you can explore the https://www.tensorflow.org/tutorials/images/transfer_learning

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras import Model
from sklearn.metrics import accuracy_score
import os
import loadData
import designPlot
# steps in transfer learning are 
# a/ data proparation
# b/ model preparation
# c/ model Training
# d/ testing

# !wget --no-check-certificate \
#     https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
#     -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5

# Set the weights file you downloaded into a variable
local_weights_file = './weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

# Initialize the base model.
# Set the input shape and remove the dense layers.
pre_trained_model = InceptionV3(input_shape = (150, 150, 3), 
                                include_top = False, 
                                weights = None)

# Load the pre-trained weights you downloaded.
pre_trained_model.load_weights(local_weights_file)

# Freeze the weights of the layers.
for layer in pre_trained_model.layers:
    layer.trainable = False

#pre_trained_model.layers.pop()
# Flatten the output layer to 1 dimension
x=layers.Flatten()(pre_trained_model.output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
#x = layers.Dense(1024, activation='relu')(pre_trained_model.output)
# Add a dropout rate of 0.2
#x = layers.Dropout(0.2)(x)
# Add a final sigmoid layer for multi classification
x=layers.Dense(1, activation='sigmoid')(x)

#x=layers.Dense(3, activation='softmax')(pre_trained_model.output)
  

        

# Append the dense network to the base model
model = Model(pre_trained_model.input, x) 

# Print the model summary. See your dense network connected at the end.
model.summary()



# Set the training parameters
model.compile(optimizer = SGD(learning_rate=0.000001), #RMSprop(learning_rate=0.0001)
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])
#model.compile(optimizer = 'adam', 
           #   loss = 'categorical_crossentropy', 
             #  metrics = ['accuracy'])

#Prepare the dataset

base_dir='.\simples'

train_dir, validation_dir= loadData.load(base_dir)
BATCH_SIZE=10
IMAGE_SIZE=150
train_generator, validation_generator=loadData.train_val_generators(train_dir, validation_dir, IMAGE_SIZE, BATCH_SIZE)

history = model.fit(
            train_generator,
            validation_data = validation_generator,
            #steps_per_epoch = 1,
            epochs = 400,
           # validation_steps = 1,
            verbose = 1)

designPlot.diagrams(history)

#Epoch 10/10
#2/2 [==============================] - 14s 7s/step - loss: 0.0000e+00 - accuracy: 0.5000 - val_loss: 0.0000e+00 - val_accuracy: 0 0.5000
