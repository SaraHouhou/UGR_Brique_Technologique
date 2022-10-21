import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input

def load(base_dir) :
 #base_dir = 'Data/'

    print("Contents of base directory:")
    print(os.listdir(base_dir))

    train_dir = os.path.join(base_dir, 'training')
    test_dir = os.path.join(base_dir, 'test')
    validation_dir = os.path.join(base_dir, 'validation')
 
    return train_dir, test_dir, validation_dir


def train_val_generators(TRAINING_DIR, TEST_DIR, IMAGE_SIZE, BATCH_SIZE, CLASS_MODE):
  """
  Creates the training and validation data generators
  
  Args:
    TRAINING_DIR (string): directory path containing the training images
    VALIDATION_DIR (string): directory path containing the testing/validation images
    
  Returns:
    train_generator, validation_generator - tuple containing the generators
  """
  ### START CODE HERE
  # need to utilize the VGG16 preprocessing function on our image data.
  # Instantiate the ImageDataGenerator class with data augmentation
  train_datagen = ImageDataGenerator(
                                    rotation_range = 40,
                                    width_shift_range = 0.2,
                                    height_shift_range = 0.2,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True, # It attempts to recreate lost information after a transformation like a shear
                                    preprocessing_function=preprocess_input)

  #la méthode flow_from_directory vous permet de normaliser la variété de resolutions en définissant un tuple appelé target_size qui sera utilisé pour convertir chaque image à cette résolution cible.
    # Pass in the appropiate arguments to the flow_from_directory method
  train_generator = train_datagen.flow_from_directory(directory=TRAINING_DIR,
                                                        batch_size=BATCH_SIZE,
                                                        class_mode=CLASS_MODE,
                                                        target_size=(IMAGE_SIZE, IMAGE_SIZE))

    # Instantiate the ImageDataGenerator class (don't forget to set the rescale argument)
  test_datagen = ImageDataGenerator( preprocessing_function=preprocess_input)

    # Pass in the appropiate arguments to the flow_from_directory method
  test_generator = test_datagen.flow_from_directory(directory=TEST_DIR,
                                                                  batch_size=BATCH_SIZE,
                                                                  class_mode=CLASS_MODE,
                                                                  target_size=(IMAGE_SIZE, IMAGE_SIZE))

  
    ### END CODE HERE
  return train_generator, test_generator


def Show_NB_Data(train_dir, test_dir, validation_dir):
  # Directory with training horse pictures
  train_four_dir = os.path.join(train_dir, 'four')
# Directory with training humans pictures
  train_one_dir = os.path.join(train_dir, 'one')
  # Directory with training humans pictures
  train_two_up_dir = os.path.join(train_dir, 'two_up')
# Directory with validation horse pictures
  validation_four_dir = os.path.join(validation_dir, 'four')
# Directory with validation human pictures
  validation_one_dir = os.path.join(validation_dir, 'one')
# Directory with validation human pictures
  validation_two_up_dir = os.path.join(validation_dir, 'two_up')

# Directory with validation horse pictures
  test_four_dir = os.path.join(test_dir, 'four')
# Directory with validation human pictures
  test_one_dir = os.path.join(test_dir, 'one')
# Directory with validation human pictures
  test_two_up_dir = os.path.join(test_dir, 'two_up')

# Check the number of images for each class and set
  print(f"There are {len(os.listdir(train_four_dir))} images of four for training.\n")
  print(f"There are {len(os.listdir(train_one_dir))} images of one for training.\n")
  print(f"There are {len(os.listdir(train_two_up_dir))} images of two_up for training.\n")

  print(f"There are {len(os.listdir(test_four_dir))} images of four for test.\n")
  print(f"There are {len(os.listdir(test_one_dir))} images of one for test.\n")
  print(f"There are {len(os.listdir(test_two_up_dir))} images of two_up for test.\n")

  print(f"There are {len(os.listdir(validation_four_dir))} images of four for validation.\n")
  print(f"There are {len(os.listdir(validation_one_dir))} images of one for validation.\n")
  print(f"There are {len(os.listdir(validation_two_up_dir))} images of two_up for validation.\n")




# image visualisation
def create_image(image_size):
  return tensorflow.random.uniform((image_size, image_size,3), 0.5, 0.5)

def plot_image(image, title='random image'):
  image = image - tensorflow.math.reduce_min(image)
  image = image / tensorflow.math.reduce_max(image)
  plt.imshow(image)
  plt.xticks([])
  plt.yticks([])
  plt.title(title)
  plt.show()


    # print("\nContents of train directory:")
    # print(os.listdir(train_dir))

    # print("\nContents of validation directory:")
    # print(os.listdir(validation_dir))

    # # Directory with training cat/dog pictures
    # train_call_dir = os.path.join(train_dir, 'call')
    # train_menu_dir = os.path.join(train_dir, 'menu')

    # # Directory with validation cat/dog pictures
    # validation_call_dir = os.path.join(validation_dir, 'call')
    # validation_menu_dir = os.path.join(validation_dir, 'menu')

    # train_call_fnames = os.listdir( train_call_dir )
    # train_menu_fnames = os.listdir( train_menu_dir )

    # print(train_call_fnames[:10])
    # print(train_menu_fnames[:10])



    #let's find out the total number of call and menu images in the train and validation directories:
        
        
    # print('total training call images :', len(os.listdir(train_call_dir ) ))
    # print('total validation call images :', len(os.listdir(validation_call_dir ) ))

    # print(' total training menu images :', len(os.listdir(train_menu_dir) ))
    # print('total validation menu images :', len(os.listdir(validation_menu_dir ) ))

    # #take a look at a few pictures to get a better sense of what the call and menu datasets look like. First, configure the matplotlib parameters:
    # # Parameters for our graph; we'll output images in a 4x4 configuration
    # nrows = 4
    # ncols = 4

    # pic_index = 0 # Index for iterating over images

    # #Display a batch of 8 call and 8 menu pictures. You can re-run the cell to see a fresh batch each time:
        
        
    # # Set up matplotlib fig, and size it to fit 4x4 pics
    # fig = plt.gcf()
    # fig.set_size_inches(ncols*4, nrows*4)

    # pic_index+=8

    # next_call_pix = [os.path.join(train_call_dir, fname) 
    #                 for fname in train_call_fnames[ pic_index-8:pic_index] 
    #                 ]

    # next_menu_pix = [os.path.join(train_menu_dir, fname) 
    #                 for fname in train_menu_fnames[ pic_index-8:pic_index]
    #                 ]

    # for i, img_path in enumerate(next_call_pix+next_menu_pix):
    # # Set up subplot; subplot indices start at 1
    #     sp = plt.subplot(nrows, ncols, i + 1)
    #     sp.axis('Off') # Don't show axes (or gridlines)

    #     img = mpimg.imread(img_path)
    #     #print(img.shape)

    #     plt.show()

# Load the first example of a horse
# Directory with training horse pictures
# train_horses_dir = os.path.join(train_dir, 'horses')
# # Directory with training humans pictures
# train_humans_dir = os.path.join(train_dir, 'humans')
# # Directory with validation horse pictures
# validation_horses_dir = os.path.join(validation_dir, 'horses')
# # Directory with validation human pictures
# validation_humans_dir = os.path.join(validation_dir, 'humans')
# sample_image  = load_img(f"{os.path.join(train_horses_dir, os.listdir(train_horses_dir)[0])}")

# # Convert the image into its numpy array representation
# sample_array = img_to_array(sample_image)

# print(f"Each image has shape: {sample_array.shape}")