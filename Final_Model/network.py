from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, GlobalAveragePooling2D, Conv2D, MaxPooling2D


""" "fine-tune" the last convolutional block of the VGG16 model alongside the top-level classifier. 
Fine-tuning consist in starting from a trained network, then re-training it on a new dataset using very small weight updates. In our case, this can be done in 3 steps:

    - instantiate the convolutional base of VGG16 and load its weights
    - add our previously defined fully-connected model on top, and load its weights
    - freeze the layers of the VGG16 model up to the last convolutional block """




def create_model(img_rows, img_cols, nbclasses):
    print("[INFO loading network...")
    model_vgg = VGG16(weights="imagenet", include_top=False, input_shape=(img_rows, img_cols, 3))
    model_vgg.summary()

    # Freeze all the layers except the last 3 layers
    for layer in model_vgg.layers[:-3]:
        layer.trainable = False
        
    # show the layers
    for layer in model_vgg.layers:
        print(layer, layer.trainable)

    model_transfer_full = Sequential()
    model_transfer_full.add(model_vgg)
    model_transfer_full.add(GlobalAveragePooling2D())
    model_transfer_full.add(Dense(512, activation='relu'))
    model_transfer_full.add(Dropout(0.5))
    model_transfer_full.add(Dense(nbclasses, activation='softmax'))
    return model_transfer_full