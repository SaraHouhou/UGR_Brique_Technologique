from tensorflow.keras.preprocessing.image import ImageDataGenerator

def  dataGenerator_with_Aug(train_data_dir, validation_data_dir, img_rows, img_cols, batch_size, Gestures_Names):
  #creating a data generator
  train_datagen = ImageDataGenerator(
        rescale=1./255, # normalize the image to 0-1 range
        rotation_range=30, #rotate image upto 30 degree to rotated image
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest')
  
  
  train_generator = train_datagen.flow_from_directory(
          train_data_dir,
          target_size=(img_rows, img_cols),
          batch_size =batch_size,
          class_mode='categorical',
          classes=Gestures_Names,
          shuffle=True)
  
  validation_datagen = ImageDataGenerator(rescale=1./255)
  
  validation_generator = validation_datagen.flow_from_directory(
          validation_data_dir,
          target_size=(img_rows, img_cols),
          batch_size=batch_size,
          class_mode='categorical',
          classes=Gestures_Names,
          shuffle=False)
  return train_generator, validation_generator

def  dataGenerator_without_Aug(train_data_dir, validation_data_dir, img_rows, img_cols, batch_size, Gestures_Names):
  #normalize images
  train_datagen = ImageDataGenerator(rescale=1./255)
  
  
  train_generator = train_datagen.flow_from_directory(
          train_data_dir,
          target_size=(img_rows, img_cols),
          batch_size =batch_size,
          class_mode='categorical',
          classes=Gestures_Names,
          shuffle=True)
  
  validation_datagen = ImageDataGenerator(rescale=1./255)
  
  validation_generator = validation_datagen.flow_from_directory(
          validation_data_dir,
          target_size=(img_rows, img_cols),
          batch_size=batch_size,
          class_mode='categorical',
          classes=Gestures_Names,
          shuffle=False)
  return train_generator, validation_generator


#Something important to note is that the images in this dataset come in a variety of resolutions. Luckily, the flow_from_directory method allows us to standarize this by defining a tuple called target_size that will be used to convert each image to this target resolution. As we use the transfere learning with VGG16, we prepocess the image to preprocessing_input of the later and we will use a target_size of (224, 224).

def dataGenerator_with_preporcess_input_and_aug(preprocess_input,  train_path, valid_path, test_path, IMAGE_SIZE, BS, Gestures_Names):
      #la méthode flow_from_directory vous permet de normaliser la variété de resolutions en définissant un tuple appelé target_size qui sera utilisé pour convertir chaque image à cette résolution cible.
    # Pass in the appropiate arguments to the flow_from_directory meth
    train_batches= ImageDataGenerator(preprocessing_function=preprocess_input,
                                    rotation_range = 40,
                                    width_shift_range = 0.2,
                                    height_shift_range = 0.2,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True) # It attempts to recreate lost information after a transformation like a shear) 
    train_gen= train_batches.flow_from_directory(directory=train_path, 
                                    target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                    classes=Gestures_Names,
                                    batch_size=BS)
                                    
    valid_batches= ImageDataGenerator(preprocessing_function=preprocess_input) 
    valid_gen=valid_batches.flow_from_directory(directory=valid_path, 
                                    target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                    classes=Gestures_Names,
                                    batch_size=BS)
    test_batches= ImageDataGenerator(preprocessing_function=preprocess_input)
    #ImageDataGenerator.flow_from_directory() creates a DirectoryIterator, which generates batches of normalized tensor image data from the respective data directories.  
    test_gen=test_batches.flow_from_directory(directory=test_path, 
                                    target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                    classes=Gestures_Names,
                                    batch_size=BS,
                                    shuffle=False) # THE TEST SET MUST BE FALSE
    return train_gen, valid_gen, test_gen

def dataGenerator_with_preporcess_input_and_without_aug(preprocess_input,  train_path, valid_path, test_path, IMAGE_SIZE, BS, Gestures_Names):
    train_batches= ImageDataGenerator(preprocessing_function=preprocess_input) 
    train_gen=train_batches.flow_from_directory(directory=train_path, 
                                    target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                    classes=Gestures_Names,
                                    batch_size=BS)
                                    
    valid_batches= ImageDataGenerator(preprocessing_function=preprocess_input) 
    valid_gen=valid_batches.flow_from_directory(directory=valid_path, 
                                    target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                    classes=Gestures_Names,
                                    batch_size=BS)
    test_batches= ImageDataGenerator(preprocessing_function=preprocess_input)  
    test_gen=test_batches.flow_from_directory(directory=test_path, 
                                    target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                    classes=Gestures_Names,
                                    batch_size=BS)
    return train_gen, valid_gen, test_gen