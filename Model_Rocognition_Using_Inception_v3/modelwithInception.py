""" Using inception and Using dropout!
dropout Another useful tool to explore at this point is the Dropout. 

The idea behind it is to remove a random number of neurons in your neural network. This works very well for two reasons: The first is that neighboring neurons often end up with similar weights, which can lead to overfitting, so dropping some out at random can remove this. The second is that often a neuron can over-weigh the input from a neuron in the previous layer, and can over specialize as a result. Thus, dropping out can break the neural network out of this potential bad habit!  """

# For more on how to freeze/lock layers, you can explore the https://www.tensorflow.org/tutorials/images/transfer_learning

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
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

#we will use up to mixed_7 as our base model and add to that. 
# This is because the original last layer might be too specialized in what it has learned so it might not translate well into your application. 
# mixed_7 on the other hand will be more generalized and you can start with that for any application. 

# Choose `mixed_7` as the last layer of your base model
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output


# check that everything works as expected 
#last_output_expected = last_output(pre_trained_model)
# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)                  
# Add a final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')(x)           

# Append the dense network to the base model
model = Model(pre_trained_model.input, x) 

# Print the model summary. See your dense network connected at the end.
model.summary()



# Set the training parameters
model.compile(optimizer = RMSprop(learning_rate=0.00001), 
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])
# model.compile(optimizer = 'adam', 
#               loss = 'categorical_crossentropy', 
#               metrics = ['accuracy'])

#Prepare the dataset

base_dir='.\simples'

train_dir, validation_dir= loadData.load(base_dir)
train_generator, validation_generator=loadData.train_val_generators(train_dir, validation_dir, 150)

#callbacks = myCallback()
history = model.fit(
            train_generator,
            validation_data = validation_generator,
            #teps_per_epoch = 1,
            epochs = 100,
           # validation_steps = 1,
            verbose = 2)
            #callbacks=callbacks)

designPlot.diagrams(history)


#Resultat donnée
#Epoch 10/10
#2/2 [==============================] - 12s 6s/step - loss: 0.3487 - accuracy: 0.8750 - val_loss: 0.8117 - val_accuracy: 0.4750   