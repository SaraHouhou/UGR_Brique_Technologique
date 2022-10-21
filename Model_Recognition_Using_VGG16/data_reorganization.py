
import os
import random
import shutil
import glob
import tensorflow as  tf

""" # Data downloads from test set of Hagrid :  """


def createFolders(Data_DIR_PATH, GESTURES_Names):
    os.chdir(Data_DIR_PATH)
    print(os.getcwd())
    for i, gesture_name in enumerate(GESTURES_Names):
#         print(gesture_name)
        if os.path.isdir(os.path.join('train/', gesture_name)) is False:
            os.makedirs(os.path.join('train/', gesture_name))
#     #         #os.makedirs('train/call')
        if os.path.isdir(os.path.join('valid/', gesture_name)) is False:
            os.makedirs(os.path.join('valid/', gesture_name))
        if os.path.isdir(os.path.join('test/', gesture_name)) is False:
            os.makedirs(os.path.join('test/', gesture_name))
        
#     #     # reorganise images """


# Organize dara into train , valid , test Directory
def split_data(Data_DIR_PATH, GESTURES_Names, TRAIN_DATA_SIZE, TEST_DATA_SIZE, VALID_DATA_SIZE):
    #os.chdir(Data_DIR_PATH)  # changes the current working directory of the calling process to the directory specified in path.
    #print(list_images)
    os.chdir(Data_DIR_PATH)

    for i, gesture_name in enumerate(GESTURES_Names):
        #print(gesture_name)
        print(os.getcwd())
            #  glob  returns a list of files or folders that matches the path specified in the pathname
       # print(listfilename)
           # print(glob.glob(os.path.join(Data_DIR_PATH, gesture_name+"*")))
        print(glob.glob(gesture_name+'*.jpg'))
        for c in random.sample(glob.glob(gesture_name+'*.jpg'), TRAIN_DATA_SIZE): #3 si la phto contien dans son nom gest1
            shutil.move(c, os.path.join('train', gesture_name))
        for c in random.sample(glob.glob(gesture_name+'*.jpg'), TEST_DATA_SIZE): #3 si la phto contien dans son nom gest1
            shutil.move(c, os.path.join('test', gesture_name))
        for c in random.sample(glob.glob(gesture_name+'*.jpg'), VALID_DATA_SIZE): #3 si la phto contien dans son nom gest1
            shutil.move(c, os.path.join('valid', gesture_name))
        print(gesture_name + "done")



#""""""""""""""""""""""""""""""""""""test"""""""""""""""""""""""""
GESTURES_Names = [
     "one",
     "two.",
     "two_inverted",
    "four",
    "five",
     "rock",
     "like",
    "mute",
    "ok",
   # "three", 
   # "call"
]
Data_DIR_PATH='D:\Data'

TRAIN_DATA_SIZE= 1540
VALID_DATA_SIZE=440
TEST_DATA_SIZE=220

createFolders(Data_DIR_PATH, GESTURES_Names)
split_data(Data_DIR_PATH, GESTURES_Names, TRAIN_DATA_SIZE, TEST_DATA_SIZE, VALID_DATA_SIZE)