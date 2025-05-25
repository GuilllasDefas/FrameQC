import os
import warnings

# Mostrar logs CUDA para diagnóstico
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Mostrar todos os logs incluindo CUDA
os.environ['TF_DETERMINISTIC_OPS'] = '0'

import tensorflow as tf
from tensorflow.python.client import device_lib

print('TensorFlow version:', tf.__version__)
print('CUDA Version built with TF:', tf.test.is_built_with_cuda())
print('GPU built with TF:', tf.test.is_built_with_gpu_support())

# Verificar detalhes CUDA
print('Dispositivos locais:')
print(device_lib.list_local_devices())

print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))
print('GPU devices:', tf.config.list_physical_devices('GPU'))

# Teste de compatibilidade CUDA
if tf.test.is_built_with_cuda():
    print('CUDA build info:', tf.sysconfig.get_build_info())
