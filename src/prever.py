import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import sys
import os

# Garante que 'tf' está disponível para qualquer Lambda layer
sys.modules['tf'] = tf

# Configuração para detecção e uso da GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Permitir crescimento de memória (evita alocar toda VRAM)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU(s) detectada(s): {len(gpus)}")
    except RuntimeError as e:
        print(f"Erro na configuração da GPU: {e}")
else:
    print("Nenhuma GPU detectada. Usando CPU.")

# ====== Adicione as camadas customizadas usadas no modelo ======
@tf.keras.utils.register_keras_serializable()
class RGBToGrayscale(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RGBToGrayscale, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.image.rgb_to_grayscale(inputs)

    def get_config(self):
        return super().get_config()

@tf.keras.utils.register_keras_serializable()
class GrayscaleToRGB(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GrayscaleToRGB, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.concat([inputs, inputs, inputs], axis=-1)

    def get_config(self):
        return super().get_config()
# ===============================================================

# Variável global para armazenar o modelo carregado
_modelo_global = None
_modelo_predicao = None

def carregar_modelo(caminho_modelo='src/models/FrameQC_model.keras'):
    # Tratar diferentes versões do TensorFlow
    try:
        # Para TensorFlow 2.11+
        tf.keras.saving.enable_unsafe_deserialization()
    except AttributeError:
        try:
            # Para TensorFlow 2.9-2.10
            tf.keras.utils.enable_unsafe_deserialization()
        except AttributeError:
            pass

    # Verifica se o caminho é absoluto, senão ajusta
    if not os.path.isabs(caminho_modelo):
        caminho_modelo = os.path.abspath(caminho_modelo)
        
    # Carrega o modelo salvo com as camadas customizadas registradas
    modelo = load_model(
        caminho_modelo,
        custom_objects={
            'RGBToGrayscale': RGBToGrayscale,
            'GrayscaleToRGB': GrayscaleToRGB,
        },
        compile=False  # Não compilar o modelo para evitar avisos do otimizador
    )
    
    # Verificar dispositivo onde o modelo vai executar
    dispositivo = '/GPU:0' if len(tf.config.list_physical_devices('GPU')) > 0 else '/CPU:0'
    print(f"Modelo será executado em: {dispositivo}")
    
    return modelo

def classificar_imagem(caminho_img, modelo=None, caminho_modelo='src/models/FrameQC_model.keras'):
    global _modelo_global
    
    # Carrega o modelo apenas uma vez e reutiliza
    if modelo is None:
        if _modelo_global is None:
            _modelo_global = carregar_modelo(caminho_modelo)
        modelo = _modelo_global
    
    # Carrega a imagem no tamanho esperado pelo modelo
    img = image.load_img(caminho_img, target_size=(320, 320))
    x = image.img_to_array(img)
    
    # Normaliza a imagem EXATAMENTE como no treinamento
    # Escala para [-1, 1]
    x = x * (1.0/127.5) - 1.0
    
    # Expande para formato de batch
    x = np.expand_dims(x, axis=0)
    
    # Faz a predição usando o modelo completo
    # O modelo já inclui as camadas de pré-processamento
    prob = modelo.predict(x, verbose=0)[0][0]
    
    return prob