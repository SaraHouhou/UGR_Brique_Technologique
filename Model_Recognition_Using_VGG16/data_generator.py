from keras.preprocessing.image import ImageDataGenerator
import tensorflow as  tf


def dataGenerator_without_Aug(preprocess_input,  train_path, valid_path, test_path, IMAGE_SIZE, BS, Gestures_Names):
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


def dataGenerator_with_Aug(preprocess_input,  train_path, valid_path, test_path, IMAGE_SIZE, BS, Gestures_Names):
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

# Method for test that the numbers that we specify are taken into account
def testDataGeneratorMethod(train_batches, valid_batches, test_batches, nb_classes, TRAIN_DATA_SIZE, VALID_DATA_SIZE, TEST_DATA_SIZE ):
    assert train_batches.n == TRAIN_DATA_SIZE 
    assert valid_batches.n== VALID_DATA_SIZE
    assert test_batches.n== TEST_DATA_SIZE
    assert train_batches.num_classes== valid_batches.num_classes==test_batches.num_classes==nb_classes


