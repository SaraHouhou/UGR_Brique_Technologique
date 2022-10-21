import os
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import designPlot
import loadData

def train_val_generators(TRAINING_DIR, VALIDATION_DIR, IMAGE_SIZE):
  """
  Creates the training and validation data generators
  
  Args:
    TRAINING_DIR (string): directory path containing the training images
    VALIDATION_DIR (string): directory path containing the testing/validation images
    
  Returns:
    train_generator, validation_generator - tuple containing the generators
  """
  ### START CODE HERE

  # Instantiate the ImageDataGenerator class with data augmentation
  train_datagen = ImageDataGenerator(rescale=1./255.0,
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True, #  training data only has people facing left, but I want to classify people facing right
                                    fill_mode='nearest' # It attempts to recreate lost information after a transformation like a shear
                                    )

#la méthode flow_from_directory vous permet de normaliser la variété de resolutions en définissant un tuple appelé target_size qui sera utilisé pour convertir chaque image à cette résolution cible.
  # Pass in the appropiate arguments to the flow_from_directory method
  train_generator = train_datagen.flow_from_directory(directory=TRAINING_DIR,
                                                      batch_size=20,
                                                      class_mode='binary',
                                                      target_size=(IMAGE_SIZE, IMAGE_SIZE))

  # Instantiate the ImageDataGenerator class (don't forget to set the rescale argument)
  validation_datagen = ImageDataGenerator( rescale = 1./255.0)

  # Pass in the appropiate arguments to the flow_from_directory method
  validation_generator = validation_datagen.flow_from_directory(directory=VALIDATION_DIR,
                                                                batch_size=20,
                                                                class_mode='binary',
                                                                target_size=(IMAGE_SIZE, IMAGE_SIZE))
  ### END CODE HERE
  return train_generator, validation_generator

# Test your generators


# GRADED FUNCTION: create_model
def create_model():
  # DEFINE A KERAS MODEL TO CLASSIFY CATS V DOGS
  # USE AT LEAST 3 CONVOLUTION LAYERS

  ### START CODE HERE

  model = tf.keras.models.Sequential([ 
      tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2,2), 
      tf.keras.layers.Conv2D(128, (3,3), activation='relu'), 
      tf.keras.layers.MaxPooling2D(2,2),
       # The fourth convolution
      tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2,2),
#     # Flatten the results to feed into a DNN
      tf.keras.layers.Flatten(), 
      # 512 neuron hidden layer
      tf.keras.layers.Dense(512, activation='relu'), 
#     # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
      tf.keras.layers.Dense(1, activation='sigmoid')  
  ])

  model.compile(optimizer=RMSprop(lr= 0.001), loss='binary_crossentropy', metrics=['accuracy'])
   ### END CODE HERE

  return model




    
  # Test your generators

def test():

    base_dir = '.\simples'
    print("Contents of base directory:")
    print(os.listdir(base_dir))
    TRAINING_DIR = os.path.join(base_dir, 'training')
    VALIDATION_DIR = os.path.join(base_dir, 'validation')  
    IMAGE_SIZE=150
    train_generator, validation_generator = train_val_generators(TRAINING_DIR, VALIDATION_DIR, IMAGE_SIZE)
    # Get the untrained model
    model = create_model()
    # Train the model
    # Note that this may take some time.
    history = model.fit(train_generator,
                        epochs=15,  
                        verbose=1,
                        validation_data=validation_generator)  
    designPlot.diagrams(history)


test()

# results : Epoch 10/10
#2/2 [==============================] - 7s 3s/step - loss: 0.6358 - accuracy: 0.7500 - val_loss: 0.6950 - val_accuracy: 0.5000
#No handles with labels found to put in legend.