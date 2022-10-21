import tensorflow as tf
print(tf.__version__)



physical_devices=tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available:", len(physical_devices))
if len(physical_devices)==0 : 
    raise SystemError('GPU devise not found')

print(len(tf.config.list_physical_devices('GPU'))>0)
