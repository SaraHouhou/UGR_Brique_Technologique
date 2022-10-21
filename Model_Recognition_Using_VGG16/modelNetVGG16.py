from tensorflow.keras.applications.vgg16 import VGG16
#from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from math import ceil

def create_model_VGG16(IMAGE_SIZE,NBClasses, fine_tune=0):
  
  
    """   Compiles a model integrated with VGG16 pretrained layers
    
    image size: tuple - the shape of input images (width, height)
    NBClasses: int - number of classes for the output layer
    # fine_tune: int - The number of pre-trained layers to unfreeze.
    #               If set to 0, all pretrained layers will freeze during training """


    #local_weights_file = './weights/.h5'

    pre_trained_model = VGG16(include_top=False, 
                                input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                                weights='imagenet')
    
  # Load the pre-trained weights you downloaded.
#pre_trained_model.load_weights(local_weights_file)

  # Freeze the weights of the layers.
  #  for layer in pre_trained_model.layers:
       # layer.trainable = False
  # Defines how many layers to freeze during training.
    # Layers in the convolutional base are switched from trainable to non-trainable
    # depending on the size of the fine-tuning parameter.
    if fine_tune > 0:
        for layer in pre_trained_model.layers[:-fine_tune]:
            layer.trainable = False
    else:
        for layer in pre_trained_model.layers:
            layer.trainable = False
    
    last_output = pre_trained_model.output
    # print([layer.name for layer in pre_trained_model.layers if 'conv' in layer.name])
    # Flatten the output layer to 1 dimension
    x = tf.keras.layers.Flatten()(last_output)
    # Add a fully connected layer with 1,024 hidden units and ReLU activation
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    # Add a dropout rate of 0.2
    x = tf.keras.layers.Dropout(0.2)(x)
    # Add a fully connected layer with 1,024 hidden units and ReLU activation
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    # Add a dropout rate of 0.5
    x = tf.keras.layers.Dropout(0.2)(x)
    # Add a final sigmoid layer for classification
    classification_layer = tf.keras.layers.Dense(NBClasses, activation='softmax', name='prob_output')(x)

    y = tf.keras.layers.UpSampling2D((3, 3))(last_output)
    y = tf.keras.layers.Activation('relu')(y)
    y = tf.keras.layers.Conv2D(1, (3, 3), activation='linear')(y)
    position_layer = tf.keras.layers.Reshape(target_shape=(10, 10), name='pos_output')(y)
    # Append the dense network and the presition network to the base model
    model = tf.keras.models.Model(inputs=pre_trained_model.input, outputs=[classification_layer])

    return model


def get_submodel(layer_name, model):
  return tf.keras.models.Model(
      model.input, 
      model.get_layer(layer_name).output
  )
#get_submodel('block1_conv2').summary()

