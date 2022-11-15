
import os, random
import matplotlib.pyplot as plt
from PIL import Image
#Visualize le nombre de données"""
def DataCount(train_path, valid_path):
  # Directory with training one pictures
  train_one_dir = os.path.join(train_path, 'one')
  # Directory with training two pictures
  train_two_dir = os.path.join(train_path, 'two')
  # Directory with training three pictures
  train_three_dir = os.path.join(train_path, 'three')
  # Directory with training for pictures
  train_four_dir = os.path.join(train_path, 'four')
  # Directory with training five pictures
  train_five_dir = os.path.join(train_path, 'five')
  # Directory with training call pictures
  train_call_dir = os.path.join(train_path, 'call')
  # Directory with training v pictures
  train_v_dir = os.path.join(train_path, 'v')


  print(f"There are {len(os.listdir(train_one_dir))} images of one for training.\n")
  print(f"There are {len(os.listdir(train_two_dir))} images of two for training.\n")
  print(f"There are {len(os.listdir(train_three_dir))} images of three for training.\n")
  print(f"There are {len(os.listdir(train_four_dir))} images of four for training.\n")
  print(f"There are {len(os.listdir(train_five_dir))} images of five for training.\n")
  print(f"There are {len(os.listdir(train_call_dir))} images of call for training.\n")
  print(f"There are {len(os.listdir(train_v_dir))} images of v for training.\n")


  # Directory with validation one pictures
  validation_one_dir = os.path.join(valid_path, 'one')
  # Directory with validation two pictures
  validation_two_dir = os.path.join(valid_path, 'two')
  # Directory with validation three pictures
  validation_three_dir = os.path.join(valid_path, 'three')
  # Directory with validation for pictures
  validation_four_dir = os.path.join(valid_path, 'four')
  # Directory with validation five pictures
  validation_five_dir = os.path.join(valid_path, 'five')
  # Directory with validation call pictures
  validation_call_dir = os.path.join(valid_path, 'call')
  # Directory with validation v pictures
  validation_v_dir = os.path.join(valid_path, 'v')

  print(f"There are {len(os.listdir(validation_one_dir))} images of one for validation.\n")
  print(f"There are {len(os.listdir(validation_two_dir))} images of two for validation.\n")
  print(f"There are {len(os.listdir(validation_three_dir))} images of three for validation.\n")
  print(f"There are {len(os.listdir(validation_four_dir))} images of four for validation.\n")
  print(f"There are {len(os.listdir(validation_five_dir))} images of five for validation.\n")
  print(f"There are {len(os.listdir(validation_call_dir))} images of call for validation.\n")
  print(f"There are {len(os.listdir(validation_v_dir))} images of v for validation.\n")



def DataCount_TinyHGR(train_path, valid_path):
  # Directory with training one pictures
  train_first_dir = os.path.join(train_path, 'first')
  
  # Directory with training two pictures
  train_l_dir = os.path.join(train_path, 'l')

  # Directory with training three pictures
  train_ok_dir = os.path.join(train_path, 'ok')

  # Directory with training for pictures
  train_palm_dir = os.path.join(train_path, 'palm')

  # Directory with training five pictures
  train_pointer_dir = os.path.join(train_path, 'pointer')

  # Directory with training call pictures
  train_thumb_down_dir = os.path.join(train_path, 'thumb_down')

  # Directory with training v pictures
  train_thumb_up_dir = os.path.join(train_path, 'thumb_up')



  print(f"There are {len(os.listdir(train_first_dir))} images of first for training.\n")
  print(f"There are {len(os.listdir(train_l_dir))} images of l for training.\n")
  print(f"There are {len(os.listdir(train_ok_dir))} images of ok for training.\n")
  print(f"There are {len(os.listdir(train_palm_dir))} images of palm for training.\n")
  print(f"There are {len(os.listdir(train_pointer_dir))} images of pointer for training.\n")
  print(f"There are {len(os.listdir(train_thumb_down_dir))} images of thumb_down for training.\n")
  print(f"There are {len(os.listdir(train_thumb_up_dir))} images of thumb_up for training.\n")


  # Directory with validation one pictures
  validation_first_dir = os.path.join(valid_path, 'first')
 
  # Directory with validation two pictures
  validation_l_dir = os.path.join(valid_path, 'l')
 

  # Directory with validation three pictures
  validation_ok_dir = os.path.join(valid_path, 'ok')
  

  # Directory with validation for pictures
  validation_palm_dir = os.path.join(valid_path, 'palm')
 

  # Directory with validation five pictures
  validation_pointer_dir = os.path.join(valid_path, 'pointer')
  

  # Directory with validation call pictures
  validation_thumb_down_dir = os.path.join(valid_path, 'thumb_down')
  

  # Directory with validation v pictures
  validation_thumb_up_dir = os.path.join(valid_path, 'thumb_up')
  


  print(f"There are {len(os.listdir(validation_first_dir))} images of first for validation.\n")
  print(f"There are {len(os.listdir(validation_l_dir))} images of l for validation.\n")
  print(f"There are {len(os.listdir(validation_ok_dir))} images of ok for validation.\n")
  print(f"There are {len(os.listdir(validation_palm_dir))} images of palm for validation.\n")
  print(f"There are {len(os.listdir(validation_pointer_dir))} images of pointer for validation.\n")
  print(f"There are {len(os.listdir(validation_thumb_down_dir))} images of thumb_down for validation.\n")
  print(f"There are {len(os.listdir(validation_thumb_up_dir))} images of thumb_up for validation.\n")

  
  
def VisualizeImages(path):
  dirs = os.listdir(path)
  image_random_number = 2
  plt.figure(1, figsize=(20, 20))
  plt.axis('off')
  n = 0
  for folder in os.listdir(path):
      image_lists = os.listdir(path+'\\'+folder)
      for i in range(image_random_number):
          n += 1
          index = random.randint(1,image_random_number)
          image = image_lists[index]
          img = os.path.join(path,folder,image)
          im = Image.open(img)
          plt.subplot(5, 5, n)
          plt.axis('off')
          plt.imshow(im,cmap='gray')
  plt.show()