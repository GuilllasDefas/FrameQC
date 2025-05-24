import os

os.environ['TF_DETERMINISTIC_OPS'] = '0'

import tensorflow as tf

print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))
